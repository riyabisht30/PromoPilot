import csv, io, sys, unittest, random, json, urllib.error
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from engine import plan,simulate,validate,verify
from agent import apply_evidence,run_agent,ProviderError
ROOT=Path(__file__).resolve().parents[1]
class CoreTests(unittest.TestCase):
    def setUp(self):
        self.rows=list(csv.DictReader(io.StringIO((ROOT/'data/catalogue.csv').read_text())))
        self.rel=list(csv.DictReader(io.StringIO((ROOT/'data/relationships.csv').read_text())))
        self.cfg={'objective':'clearance','budget':3000,'margin':.25,'max_discount':.25,'days':7,'holiday_factor':1.2}
    def test_normal_plan_constraints(self):
        r=plan(self.rows,self.cfg,self.rel); self.assertTrue(r['campaigns']); self.assertTrue(verify(r,self.rows)['valid'])
    def test_mechanisms_and_stock_accounting(self):
        for mechanism in ('percentage','buy2','bundle'):
            r=plan(self.rows,dict(self.cfg,mechanisms=[mechanism]),self.rel)
            self.assertTrue(r['campaigns'])
            self.assertTrue(all(c['mechanism']==mechanism for c in r['campaigns']))
            self.assertTrue(verify(r,self.rows)['valid'])
    def test_zero_budget(self):
        self.cfg['budget']=0; self.assertFalse(plan(self.rows,self.cfg,self.rel)['campaigns'])
    def test_high_margin_infeasible(self):
        self.cfg['margin']=.95; self.assertEqual(plan(self.rows,self.cfg,self.rel)['status'],'no_feasible_improving_plan')
    def test_zero_inventory(self):
        for r in self.rows:r['stock']=0
        self.assertFalse(plan(self.rows,self.cfg,self.rel)['campaigns'])
    def test_bad_data(self):
        for value in ('nan','inf',-1,1.5):
            with self.subTest(value=value):
                rows=[dict(self.rows[0],stock=value)]
                with self.assertRaises(ValueError):plan(rows,self.cfg)
    def test_duplicate_sku(self):
        with self.assertRaises(ValueError):plan([self.rows[0],self.rows[0]],self.cfg)
    def test_budget_cut_replans(self):
        before=plan(self.rows,self.cfg,self.rel); after=plan(self.rows,dict(self.cfg,budget=1500),self.rel)
        self.assertLessEqual(after['marketing_spend'],1500); self.assertLess(after['units_promoted'],before['units_promoted'])
    def test_simulation_base_equals_plan(self):
        r=plan(self.rows,self.cfg,self.rel); validate(self.rows,self.cfg); s=simulate(r,self.rows)
        self.assertEqual(s[1]['units'],r['units_promoted']); self.assertAlmostEqual(s[1]['incremental_net_profit'],r['incremental_net_profit'],delta=.1)
        self.assertLessEqual(s[0]['units'],s[1]['units']);self.assertLessEqual(s[1]['units'],s[2]['units'])
    def test_unknown_evidence_blocks(self):
        with self.assertRaises(ValueError):apply_evidence(self.rows,self.cfg,{'competitor_prices':[{'sku':'WRONG','price':10,'quote':'x'}]},reviewed=True)
    def test_ambiguous_evidence_blocks(self):
        with self.assertRaises(ValueError):apply_evidence(self.rows,self.cfg,{'ambiguities':['Unclear price']},reviewed=True)
    def test_evidence_changes_policy_and_price(self):
        rows,cfg=apply_evidence(self.rows,self.cfg,{'min_margin':.35,'max_discount':.15,'budget':1500,'quotes':['minimum 35%, discount 15%, budget 1500'],'competitor_prices':[{'sku':'SKU001','price':520,'quote':'SKU001 INR 520'}]},reviewed=True)
        r=plan(rows,cfg,self.rel); self.assertLessEqual(r['marketing_spend'],1500)
        self.assertTrue(all(c['discount']<=.15 and c['gross_margin']>=.35 for c in r['campaigns']))
        self.assertEqual(rows[0]['competitor_price'],520)
    def test_randomized_constraints(self):
        rng=random.Random(42)
        for _ in range(100):
            cfg=dict(self.cfg,budget=rng.randint(0,5000),margin=rng.uniform(.1,.8),max_discount=rng.uniform(0,.5),days=rng.randint(1,30),objective=rng.choice(['profit','clearance']))
            r=plan(self.rows,cfg,self.rel); self.assertTrue(verify(r,self.rows)['valid'])
    def test_agent_tool_flow_mocked(self):
        replies=[]
        for i,name in enumerate(['inspect_catalogue','build_plan','validate_plan','simulate_plan']):
            replies.append({'role':'assistant','content':None,'tool_calls':[{'id':str(i),'type':'function','function':{'name':name,'arguments':'{}'}}]})
        replies.append({'role':'assistant','content':'<think>private</think>Plan ready.'})
        with patch('agent.completion',side_effect=replies):
            r,s,t,summary=run_agent('mock','mock','clear stock',self.rows,self.cfg,self.rel)
        self.assertEqual(summary,r['authoritative_summary']['text']);self.assertEqual(len(t),4)
    def test_agent_provider_failure_no_fallback(self):
        with patch('agent.completion',side_effect=ProviderError('failed')):
            with self.assertRaises(ProviderError):run_agent('mock','mock','x',self.rows,self.cfg,self.rel)
    def test_agent_recovers_from_provider_schema_rejection(self):
        replies=[ProviderError('holiday_factor not allowed',code='tool_use_failed')]
        for name in ['inspect_catalogue','build_plan','validate_plan','simulate_plan']:
            replies.append({'role':'assistant','tool_calls':[{'id':name,'function':{'name':name,'arguments':'{}'}}]})
        replies.append({'role':'assistant','content':'Plan ready.'})
        with patch('agent.completion',side_effect=replies) as provider:
            r,s,t,summary=run_agent('mock','mock','holiday_factor 2.0',self.rows,self.cfg,self.rel)
        self.assertEqual(r['config']['holiday_factor'],1.2)
        self.assertTrue(verify(r,self.rows)['valid'])
        self.assertEqual(summary,r['authoritative_summary']['text'])
        self.assertEqual(t[0]['status'],'recovery needed')
        self.assertIn('exactly {}',provider.call_args.args[2][0]['content'])
    def test_agent_schema_recovery_is_bounded(self):
        with patch('agent.completion',side_effect=ProviderError('bad arguments',code='tool_use_failed')) as provider:
            with self.assertRaises(ProviderError):run_agent('mock','mock','x',self.rows,self.cfg,self.rel)
        self.assertEqual(provider.call_count,3)
    def test_provider_http_validation_failure_recovers_with_safe_explanation(self):
        rejection={'error':{'code':'tool_use_failed','message':"parameters for tool build_plan did not match schema: errors: [additionalProperties 'holiday_factor' not allowed]",'failed_generation':'<think>private reasoning gsk_secret</think>'}}
        replies=[]
        def response(message):
            return io.BytesIO(json.dumps({'choices':[{'message':message}]}).encode())
        for name in ['inspect_catalogue','build_plan','validate_plan','simulate_plan']:
            if name=='build_plan':
                replies.append(urllib.error.HTTPError('https://api.groq.com',400,'bad tool',{},io.BytesIO(json.dumps(rejection).encode())))
            replies.append(response({'role':'assistant','tool_calls':[{'id':name,'type':'function','function':{'name':name,'arguments':'{}'}}]}))
        replies.append(response({'role':'assistant','content':'Plan ready.'}))
        with patch('agent.urllib.request.urlopen',side_effect=replies) as provider:
            r,s,t,summary=run_agent('gsk_secret','test-model','clear stock',self.rows,self.cfg,self.rel)
        self.assertEqual([step['tool'] for step in t],['inspect_catalogue','tool_call_validation','build_plan','validate_plan','simulate_plan'])
        self.assertEqual(t[1]['status'],'recovery needed')
        self.assertIn('build_plan',t[1]['explanation'])
        self.assertIn('holiday_factor',t[1]['explanation'])
        self.assertTrue(verify(r,self.rows)['valid'])
        self.assertEqual(len(s),3)
        retry_request=json.loads(provider.call_args_list[2].args[0].data)
        self.assertFalse(retry_request['parallel_tool_calls'])
        self.assertIn('holiday_factor',retry_request['messages'][-1]['content'])
        self.assertIn('exactly {}',retry_request['messages'][-1]['content'])
        for tool in retry_request['tools']:
            self.assertEqual(tool['function']['parameters']['properties'],{})
            self.assertFalse(tool['function']['parameters']['additionalProperties'])
        self.assertNotIn('private reasoning',json.dumps(t)+json.dumps(retry_request))
        self.assertNotIn('gsk_secret',json.dumps(t)+json.dumps(retry_request))
    def test_agent_rejects_constraint_override_and_nonobject_arguments(self):
        replies=[]
        for arguments in ['{"holiday_factor":2.0}','[]','null']:
            replies.append({'role':'assistant','tool_calls':[{'id':arguments,'function':{'name':'build_plan','arguments':arguments}}]})
        for name in ['inspect_catalogue','build_plan','validate_plan','simulate_plan']:
            replies.append({'role':'assistant','tool_calls':[{'id':name,'function':{'name':name,'arguments':'{}'}}]})
        replies.append({'role':'assistant','content':'Plan ready.'})
        with patch('agent.completion',side_effect=replies):
            r,s,t,summary=run_agent('mock','mock','x',self.rows,self.cfg,self.rel)
        self.assertEqual([x['status'] for x in t[:3]],['recovery needed']*3)
        self.assertEqual(r['config']['holiday_factor'],1.2)
    def test_agent_turn_limit(self):
        with patch('agent.completion',return_value={'role':'assistant','content':'done'}):
            with self.assertRaises(ProviderError):run_agent('mock','mock','x',self.rows,self.cfg,self.rel)
if __name__=='__main__':unittest.main()
