import csv,unittest
from pathlib import Path
from unittest.mock import patch
from agent import ProviderError
from workflow import plan_with_quota_fallback
from streamlit.testing.v1 import AppTest
ROOT=Path(__file__).resolve().parents[1]
CFG={'objective':'clearance','budget':3000,'margin':.25,'max_discount':.25,'days':7,'holiday_factor':1.2,'mechanisms':['percentage','buy2','bundle']}
class QuotaFallbackTests(unittest.TestCase):
    def setUp(self):
        self.rows=list(csv.DictReader((ROOT/'data/catalogue.csv').read_text().splitlines()))
        self.rel=list(csv.DictReader((ROOT/'data/relationships.csv').read_text().splitlines()))
    def test_quota_fallback_one_attempt_and_explicit_labels(self):
        error=ProviderError('quota',http_status=429,code='rate_limit_exceeded',rate_limits={'retry-after':'60'})
        with patch('agent.completion') as api,patch('agent.run_agent',side_effect=error) as live:
            result,_,trace,_=plan_with_quota_fallback(lambda:live(),self.rows,CFG,self.rel,enabled=True)
        live.assert_called_once();api.assert_not_called()
        self.assertEqual(result['execution_route'],'local_engine');self.assertEqual(result['units_promoted'],2600)
        self.assertFalse(result['fallback']['ai_workflow_completed']);self.assertEqual(trace[0]['status'],'failed')
    def test_disabled_and_nonquota_failures_do_not_fallback(self):
        for error,enabled in [(ProviderError('quota',http_status=429),False),(ProviderError('service',http_status=503),True),(ProviderError('schema',http_status=400),True)]:
            def fail():raise error
            with self.assertRaises(ProviderError):plan_with_quota_fallback(fail,self.rows,CFG,self.rel,enabled=enabled)
    def test_verified_partial_work_preserved_in_failure_record(self):
        error=ProviderError('quota',http_status=429);error.partial_run={'execution_status':'incomplete','trace':[]}
        def fail():raise error
        result=plan_with_quota_fallback(fail,self.rows,dict(CFG,margin=.95),self.rel,enabled=True)[0]
        self.assertFalse(result['campaigns']);self.assertEqual(result['fallback']['failed_attempt']['verified_partial_run'],error.partial_run)
    def test_ui_route_is_local_and_ai_not_completed(self):
        with patch('agent.run_agent',side_effect=ProviderError('quota',http_status=429)):
            app=AppTest.from_file(str(ROOT/'app.py'),default_timeout=10).run()
            app.radio[0].set_value('Live Groq agent').run()
            next(b for b in app.button if b.label=='Generate / replan').click().run()
        self.assertFalse(app.exception)
        run=app.session_state['run'];self.assertEqual(run['execution_route'],'local_engine');self.assertFalse(run['ai_workflow_completed'])
        self.assertIn('quota fallback',run['mode']);self.assertTrue(any('AI attempt did not complete' in w.value for w in app.warning))
