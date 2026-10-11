import csv,json,unittest
from pathlib import Path
from unittest.mock import patch
from streamlit.testing.v1 import AppTest
import streamlit
from workflow import local_plan,review_evidence,apply_reviewed
from agent import EXTRACTION_VERSION
from engine import verify
ROOT=Path(__file__).resolve().parents[1]
CFG={'objective':'clearance','budget':3000,'margin':.25,'max_discount':.25,'days':7,'holiday_factor':1.2,'mechanisms':['percentage','buy2','bundle']}
class LocalTests(unittest.TestCase):
    def setUp(self):
        self.rows=list(csv.DictReader((ROOT/'data/catalogue.csv').read_text().splitlines()));self.rel=list(csv.DictReader((ROOT/'data/relationships.csv').read_text().splitlines()))
        self.ev={'source':'policy.txt','source_sha256':'a'*64,'min_margin':.35,'max_discount':.15,'budget':None,'quotes':['Minimum margin 35%; max discount 15%'],'warnings':[],'blocking_ambiguities':[],'competitor_prices':[],'extraction':{'model':'openai/gpt-oss-20b','schema_version':EXTRACTION_VERSION,'cache_hit':False}}
    def test_local_normal_replan_infeasible_zero_provider_calls(self):
        with patch('agent.completion') as api:
            normal=local_plan(self.rows,CFG,self.rel)[0]
            changed=local_plan(self.rows,dict(CFG,budget=1500),self.rel)[0]
            infeasible=local_plan(self.rows,dict(CFG,margin=.95),self.rel)[0]
        api.assert_not_called();self.assertEqual(normal['units_promoted'],2600);self.assertEqual(changed['units_promoted'],1637);self.assertFalse(infeasible['campaigns'])
        for r in (normal,changed,infeasible):self.assertTrue(verify(r,self.rows)['valid']);self.assertEqual(r['execution_route'],'local_engine')
    def test_reviewed_ai_evidence_and_budget_changes_local(self):
        reviewed=review_evidence(self.ev,reviewed=True)
        with patch('agent.completion') as api:
            rows,cfg=apply_reviewed(self.rows,CFG,reviewed)
            first=local_plan(rows,cfg,self.rel)[0]
            _,cfg2=apply_reviewed(self.rows,dict(CFG,budget=1500),reviewed)
            second=local_plan(rows,cfg2,self.rel)[0]
        api.assert_not_called();self.assertEqual(first['config']['margin'],.35);self.assertEqual(second['config']['budget'],1500);self.assertEqual(reviewed['source_sha256'],'a'*64)
    def test_review_changes_invalidate_and_cache_hit_does_not_grant_review(self):
        ev=review_evidence(self.ev,reviewed=True)
        for field,value in [('budget',100),('source_sha256','b'*64),('quotes',['changed quote'])]:
            changed=json.loads(json.dumps(ev));changed[field]=value
            with self.assertRaises(ValueError):apply_reviewed(self.rows,CFG,changed)
        changed=json.loads(json.dumps(ev));changed['extraction']['schema_version']='changed'
        with self.assertRaises(ValueError):apply_reviewed(self.rows,CFG,changed)
        cached=json.loads(json.dumps(self.ev));cached['extraction']['cache_hit']=True
        with self.assertRaises(ValueError):apply_reviewed(self.rows,CFG,cached)
    def test_conflict_or_missing_required_blocks_even_reviewed(self):
        for blocker in ('Conflicting margin values','Required budget unreadable'):
            ev=review_evidence(dict(self.ev,blocking_ambiguities=[blocker]),reviewed=True)
            with self.assertRaises(ValueError):apply_reviewed(self.rows,CFG,ev)
    def test_two_sku_upload_keeps_relevant_default_relationship(self):
        class Upload:
            def getvalue(self):return (ROOT/'tests/fixtures/catalogue_two.csv').read_bytes()
        def uploader(label,*args,**kwargs):return Upload() if label.startswith('Upload catalogue CSV') else None
        with patch('streamlit.file_uploader',side_effect=uploader),patch('agent.completion') as api:
            app=AppTest.from_file(str(ROOT/'app.py'),default_timeout=10).run()
            next(b for b in app.button if b.label=='Generate / replan').click().run()
        api.assert_not_called();self.assertFalse(app.exception);self.assertFalse(app.error)
        self.assertEqual(app.session_state['run']['result']['units_promoted'],170)
    def test_ui_local_labels_and_evidence_change_requires_review(self):
        exports=[];real_download=streamlit.download_button
        def capture(label,data,*args,**kwargs):
            if label=='Export complete plan and evidence':exports.append(json.loads(data))
            return real_download(label,data,*args,**kwargs)
        with patch('agent.completion') as api,patch('streamlit.download_button',side_effect=capture):
            app=AppTest.from_file(str(ROOT/'app.py'),default_timeout=10).run()
            self.assertEqual(app.radio[0].value,'Local engine (no API)')
            app.session_state['evidence']=self.ev;app.run();app.checkbox[0].check().run()
            next(b for b in app.button if b.label=='Apply reviewed evidence').click().run()
            next(b for b in app.button if b.label=='Generate / replan').click().run()
            run=app.session_state['run'];self.assertEqual(run['execution_route'],'local_engine');self.assertEqual(run['execution_status'],'completed');self.assertTrue(run['source_evidence']['review']['reviewed'])
            self.assertEqual(exports[-1]['run']['execution_route'],'local_engine');self.assertEqual(exports[-1]['run']['result']['execution_route'],'local_engine')
            changed=dict(self.ev,budget=1000);app.session_state['evidence']=changed;app.run()
            self.assertFalse(app.checkbox[0].value)
        api.assert_not_called();self.assertFalse(app.exception)
if __name__=='__main__':unittest.main()
