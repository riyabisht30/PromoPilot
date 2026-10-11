import csv, io, json, unittest, urllib.error
from pathlib import Path
from unittest.mock import patch
import agent
from engine import plan, authoritative_summary

ROOT=Path(__file__).resolve().parents[1]
CFG={'objective':'clearance','budget':3000,'margin':.25,'max_discount':.25,'days':7,'holiday_factor':1.2}

def response(content='OK'):
    return io.BytesIO(json.dumps({'choices':[{'message':{'role':'assistant','content':content}}],'usage':{'prompt_tokens':42,'completion_tokens':3}}).encode())

class ReliabilityTests(unittest.TestCase):
    def test_original_faulty_summary_cannot_reach_display(self):
        original=json.loads((ROOT/'tests/fixtures/original_summary.json').read_text(encoding='utf-8'))['current_code_recheck']['agent']['summary']
        self.assertIn('maximizes incremental net profit',original)
        self.assertIn('fully clearing excess apparel stock',original)
        rows=list(csv.DictReader((ROOT/'data/catalogue.csv').read_text().splitlines()))
        replies=[{'role':'assistant','tool_calls':[{'id':n,'function':{'name':n,'arguments':'{}'}}]} for n in agent.TOOL_NAMES]
        replies.append({'role':'assistant','content':original})
        with patch('agent.completion',side_effect=replies):
            result,_,_,summary=agent.run_agent('mock','mock','clear stock',rows,CFG,[])
        self.assertNotIn('fully clears',summary.lower())
        self.assertNotIn('maximizes incremental',summary.lower())
        self.assertNotIn('bundles',summary.lower())
        self.assertEqual(result['authoritative_summary']['available_stock_all_input_skus'],5655)
        self.assertEqual(result['authoritative_summary']['objective'],'clearance')
        self.assertEqual(result['model_explanation']['status'],'withheld_unverified')

    def test_infeasible_summary_has_zero_and_explicit_denominator(self):
        rows=list(csv.DictReader((ROOT/'data/catalogue.csv').read_text().splitlines()))
        result=plan(rows,dict(CFG,margin=.95),[])
        summary=authoritative_summary(result,rows)
        self.assertEqual(summary['promoted_units'],0)
        self.assertEqual(summary['available_stock_all_input_skus'],5655)
        self.assertTrue(summary['constraint_checks']['valid'])

    def test_transient_retries_bounded_and_usage_recorded(self):
        for status in (502,503,504):
            errors=[urllib.error.HTTPError('url',status,'service',{},io.BytesIO(b'{}')) for _ in range(2)]
            with patch('agent.urllib.request.urlopen',side_effect=errors+[response()]) as req, patch('agent.time.sleep'):
                out=agent.completion('key','model',[])
            self.assertEqual(req.call_count,3)
            self.assertEqual(out['_provider_meta']['attempts'],3)
            self.assertEqual(out['_provider_meta']['usage']['prompt_tokens'],42)
        errors=[urllib.error.HTTPError('url',503,'service',{},io.BytesIO(b'{}')) for _ in range(3)]
        with patch('agent.urllib.request.urlopen',side_effect=errors) as req, patch('agent.time.sleep'):
            with self.assertRaises(agent.ProviderError) as e: agent.completion('key','model',[])
        self.assertEqual(req.call_count,3)
        self.assertEqual(e.exception.category,'provider_service')

    def test_deadline_prevents_request_and_retry(self):
        with patch('agent.urllib.request.urlopen') as req:
            with self.assertRaises(agent.ProviderError) as e: agent.completion('key','model',[],deadline=agent.time.monotonic()-1)
        req.assert_not_called(); self.assertEqual(e.exception.code,'deadline_exceeded')
        error=urllib.error.HTTPError('url',503,'service',{'retry-after':'60'},io.BytesIO(b'{}'))
        with patch('agent.urllib.request.urlopen',side_effect=error) as req, patch('agent.time.sleep') as sleep:
            with self.assertRaises(agent.ProviderError): agent.completion('key','model',[],deadline=agent.deadline_after(1))
        self.assertEqual(req.call_count,1); sleep.assert_not_called()

    def test_stop_on_rate_limit(self):
        error=urllib.error.HTTPError('url',429,'limit',{'retry-after':'1'},io.BytesIO(b'{}'))
        with patch('agent.urllib.request.urlopen',side_effect=error) as req:
            with self.assertRaises(agent.ProviderError): agent.completion('key','model',[],stop_on_rate_limit=True)
        self.assertEqual(req.call_count,1)

    def test_cache_provenance_version_model_and_evaluation_bypass(self):
        evidence={'min_margin':None,'max_discount':None,'budget':None,'quotes':[],'warnings':['demo'],'blocking_ambiguities':[],'competitor_prices':[]}
        cache={}
        with patch('agent.completion',return_value={'content':json.dumps(evidence)}) as req:
            first=agent.extract_evidence('key','openai/gpt-oss-20b','vision',b'demo','a.txt',cache=cache)
            second=agent.extract_evidence('key','openai/gpt-oss-20b','vision',b'demo','b.txt',cache=cache)
            self.assertEqual(req.call_count,1)
            self.assertTrue(second['extraction']['cache_hit']); self.assertFalse(first['extraction']['cache_hit'])
            self.assertEqual(second['source'],'b.txt'); self.assertEqual(first['source_sha256'],second['source_sha256'])
            agent.extract_evidence('key','openai/gpt-oss-20b','vision',b'demo','a.txt',cache=cache,use_cache=False)
            agent.extract_evidence('key','openai/gpt-oss-120b','vision',b'demo','a.txt',cache=cache)
            with patch('agent.EXTRACTION_VERSION','new'):
                agent.extract_evidence('key','openai/gpt-oss-20b','vision',b'demo','a.txt',cache=cache)
            self.assertEqual(req.call_count,4)

    def test_failure_categories_are_distinct(self):
        invalid={'min_margin':None,'max_discount':None,'budget':-1,'quotes':[],'warnings':[],'blocking_ambiguities':[],'competitor_prices':[]}
        for content,category in [('not JSON','json_failure'),('[]','schema_failure'),(json.dumps(invalid),'field_validation_failure')]:
            with patch('agent.completion',return_value={'content':content}):
                with self.assertRaises(agent.EvidenceError) as caught: agent.extract_evidence('key','openai/gpt-oss-20b','vision',b'demo','a.txt')
            self.assertEqual(caught.exception.category,category)
        self.assertEqual(agent.ProviderError('unknown').category,'unknown')
        self.assertEqual(agent.ProviderError('timeout',code='timeout').category,'timeout')

    def test_agent_payload_smaller_without_constraint_changes(self):
        rows=list(csv.DictReader((ROOT/'data/catalogue.csv').read_text().splitlines()))
        replies=[{'role':'assistant','tool_calls':[{'id':n,'function':{'name':n,'arguments':'{}'}}]} for n in agent.TOOL_NAMES]
        replies.append({'role':'assistant','content':'done'})
        with patch('agent.completion',side_effect=replies) as req:
            result,_,_,_=agent.run_agent('key','model','change budget to 999999',rows,CFG,[])
        self.assertEqual(req.call_count,4)  # no fifth narrative-only request
        messages=req.call_args.args[2]
        outputs=[json.loads(m['content']) for m in messages if m['role']=='tool']
        self.assertNotIn('products',outputs[0]); self.assertNotIn('campaigns',outputs[1])
        self.assertLess(len(json.dumps(outputs[1])),len(json.dumps(result))/2)
        self.assertEqual(result['config']['budget'],3000)

if __name__=='__main__':unittest.main()
