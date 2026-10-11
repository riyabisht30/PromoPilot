import csv,json,unittest
from pathlib import Path
from unittest.mock import patch
import agent
import streamlit
from engine import plan,simulate
from streamlit.testing.v1 import AppTest

ROOT=Path(__file__).resolve().parents[1]
CFG={'objective':'clearance','budget':3000,'margin':.25,'max_discount':.25,'days':7,'holiday_factor':1.2,'mechanisms':['percentage','buy2','bundle']}
def calls(names):
    return [{'role':'assistant','tool_calls':[{'id':n,'function':{'name':n,'arguments':'{}'}}]} for n in names]

class D2Tests(unittest.TestCase):
    def setUp(self):
        self.rows=list(csv.DictReader((ROOT/'data/catalogue.csv').read_text().splitlines()))
        self.rel=list(csv.DictReader((ROOT/'data/relationships.csv').read_text().splitlines()))
    def test_final_infeasible_result_never_requests_later_provider_narrative(self):
        replies=calls(agent.TOOL_NAMES)+[agent.ProviderError('later service failure',http_status=503)]
        with patch('agent.completion',side_effect=replies) as provider:
            p,s,trace,_=agent.run_agent('mock','mock','goal',self.rows,dict(CFG,margin=.95),self.rel)
        self.assertEqual(provider.call_count,4);self.assertEqual(p['execution_status'],'completed')
        self.assertFalse(p['campaigns']);self.assertEqual(s[1]['units'],0)
    def test_verified_infeasible_survives_failure_before_remaining_required_step(self):
        replies=calls(('build_plan','validate_plan','simulate_plan'))+[agent.ProviderError('service unavailable',http_status=503)]
        with patch('agent.completion',side_effect=replies),patch('agent.plan',wraps=agent.plan) as build:
            with self.assertRaises(agent.ProviderError) as caught:
                agent.run_agent('mock','mock','goal',self.rows,dict(CFG,margin=.95),self.rel)
        partial=caught.exception.partial_run
        self.assertEqual(build.call_count,1);self.assertEqual(partial['execution_status'],'incomplete')
        self.assertFalse(partial['result']['campaigns']);self.assertEqual(partial['scenarios'][1]['units'],0)
        self.assertNotIn('inspect_catalogue',partial['required_tools_succeeded'])
    def test_unvalidated_calculation_not_retained_as_verified(self):
        with patch('agent.completion',side_effect=calls(('build_plan',))+[agent.ProviderError('failed')]):
            with self.assertRaises(agent.ProviderError) as caught:agent.run_agent('mock','mock','goal',self.rows,CFG,self.rel)
        self.assertFalse(hasattr(caught.exception,'partial_run'))
    def test_ui_exports_retained_work_with_incomplete_status(self):
        app=AppTest.from_file(str(ROOT/'app.py'),default_timeout=10).run()
        app.radio[0].set_value('Live Groq agent').run()
        p=plan(self.rows,dict(CFG,margin=.95),self.rel);scenarios=simulate(p,self.rows)
        p['execution_status']='incomplete'
        error=agent.ProviderError('service unavailable',http_status=503)
        error.partial_run={'result':p,'scenarios':scenarios,'trace':[],'summary':'verified calculation','execution_status':'incomplete','failure_message':'service unavailable'}
        exports=[];real_download=streamlit.download_button
        def download(label,data,*args,**kwargs):
            if label=='Export complete plan and evidence':exports.append(json.loads(data))
            return real_download(label,data,*args,**kwargs)
        with patch('agent.run_agent',side_effect=error),patch('streamlit.download_button',side_effect=download):
            next(b for b in app.button if b.label=='Generate / replan').click().run()
        self.assertFalse(app.exception)
        self.assertEqual(app.session_state['run']['execution_status'],'incomplete')
        self.assertTrue(any('Incomplete agent execution' in w.value for w in app.warning))
        self.assertEqual(exports[-1]['run']['execution_status'],'incomplete')
        self.assertEqual(exports[-1]['run']['result']['execution_status'],'incomplete')
    def test_invalid_relationship_csv_blocks_before_planning(self):
        class Upload:
            def getvalue(self):return b'sku_a,sku_b,reason\nSKU001,UNKNOWN,Complementary\n'
        def uploader(label,*args,**kwargs):
            return Upload() if label.startswith('Optional relationship CSV') else None
        with patch('streamlit.file_uploader',side_effect=uploader):
            app=AppTest.from_file(str(ROOT/'app.py'),default_timeout=10).run()
        self.assertFalse(app.exception)
        self.assertTrue(any('Invalid relationships' in e.value for e in app.error))
        self.assertNotIn('run',app.session_state)
    def test_missing_optional_policy_preserves_settings_and_critical_blocker_rejects(self):
        evidence={'budget':None,'min_margin':.35,'max_discount':None,'quotes':['Minimum margin 35%'],'competitor_prices':[]}
        _,cfg=agent.apply_evidence(self.rows,CFG,evidence,reviewed=True)
        self.assertEqual(cfg['budget'],3000);self.assertEqual(cfg['max_discount'],.25)
        with self.assertRaises(ValueError):agent.apply_evidence(self.rows,CFG,dict(evidence,blocking_ambiguities=['Required budget is unreadable']),reviewed=True)

if __name__=='__main__':unittest.main()
