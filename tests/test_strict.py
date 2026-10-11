import csv,io,json,unittest,urllib.error
from pathlib import Path
from unittest.mock import patch
import agent
ROOT=Path(__file__).resolve().parents[1]
MODEL='openai/gpt-oss-20b'
VALID={'min_margin':None,'max_discount':None,'budget':None,'quotes':[],'warnings':[],'blocking_ambiguities':[],'competitor_prices':[]}
class StrictTests(unittest.TestCase):
    def test_all_objects_closed_and_required(self):
        def walk(schema):
            if schema.get('type')=='object':
                self.assertFalse(schema['additionalProperties']);self.assertEqual(set(schema['required']),set(schema['properties']))
                for child in schema['properties'].values():walk(child)
            if schema.get('type')=='array':walk(schema['items'])
        walk(agent.EVIDENCE_SCHEMA)
    def test_extraction_request_strict_and_nulls_preserved(self):
        with patch('agent.completion',return_value={'content':json.dumps(VALID)}) as call:
            output=agent.extract_evidence('mock',MODEL,'vision',b'No optional policy limits supplied.','a.txt')
        kwargs=call.call_args.kwargs
        self.assertEqual(kwargs['response_format']['type'],'json_schema');self.assertTrue(kwargs['response_format']['json_schema']['strict'])
        self.assertNotIn('tools',kwargs);self.assertIsNone(output['budget'])
    def test_missing_extra_and_nested_missing_not_repaired(self):
        for value in ({},{**VALID,'invented':3},{**VALID,'competitor_prices':[{'sku':'SKU001','price':520,'quote':'520'}]}):
            with patch('agent.completion',return_value={'content':json.dumps(value)}):
                with self.assertRaises(agent.EvidenceError):agent.extract_evidence('mock',MODEL,'vision',b'text','a.txt')
    def test_unsupported_model_no_fallback(self):
        with patch('agent.completion') as call:
            with self.assertRaises(agent.EvidenceError):agent.extract_evidence('mock','unsupported','vision',b'text','a.txt')
        call.assert_not_called()
    def test_interpretation_instructions_do_not_repair_invalid_values(self):
        candidate={**VALID,'min_margin':35,'quotes':['Minimum margin: 35%.']}
        with patch('agent.completion',return_value={'content':json.dumps(candidate)}) as call:
            with self.assertRaises(agent.EvidenceError) as error:
                agent.extract_evidence('mock',MODEL,'vision',b'Minimum margin: 35%.','a.txt')
        prompt=call.call_args.args[2][0]['content']
        self.assertIn('35% is 0.35, not 35',prompt)
        self.assertIn('do not choose one',prompt)
        self.assertEqual(error.exception.category,'field_validation_failure')
        self.assertIn('Invalid extracted min_margin',str(error.exception))
    def test_planning_requests_have_tools_without_structured_mode(self):
        rows=list(csv.DictReader((ROOT/'data/catalogue.csv').read_text().splitlines()))
        cfg={'objective':'clearance','budget':3000,'margin':.25,'max_discount':.25,'days':7,'holiday_factor':1.2}
        replies=[{'role':'assistant','tool_calls':[{'id':n,'function':{'name':n,'arguments':'{}'}}]} for n in agent.TOOL_NAMES]
        with patch('agent.completion',side_effect=replies) as call:agent.run_agent('mock',MODEL,'coordinate',rows,cfg,[])
        for args in call.call_args_list:self.assertIn('tools',args.kwargs);self.assertNotIn('response_format',args.kwargs)
    def test_rate_diagnostics_are_allowlisted(self):
        headers={'retry-after':'60','x-ratelimit-remaining-tokens':'0','authorization':'private','x-private':'private'}
        error=urllib.error.HTTPError('url',429,'limit',headers,io.BytesIO(b'{}'))
        parsed=agent.provider_error(error,'mock')
        self.assertEqual(parsed.rate_limits,{'retry-after':'60','x-ratelimit-remaining-tokens':'0'})
if __name__=='__main__':unittest.main()
