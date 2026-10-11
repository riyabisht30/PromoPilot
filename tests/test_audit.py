import csv
import io
import json
import unittest
import urllib.error
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

import agent
from engine import plan,simulate,verify,compare_discounts
from streamlit.testing.v1 import AppTest
import streamlit

ROOT=Path(__file__).resolve().parents[1]
CFG={'objective':'clearance','budget':3000,'margin':.25,'max_discount':.25,'days':7,
     'holiday_factor':1.2,'mechanisms':['percentage','buy2','bundle']}

class AuditTests(unittest.TestCase):
    def setUp(self):
        with (ROOT/'data/catalogue.csv').open() as f: self.rows=list(csv.DictReader(f))
        with (ROOT/'data/relationships.csv').open() as f: self.rel=list(csv.DictReader(f))

    def test_discount_candidates_show_stock_saturation_and_economics(self):
        options=compare_discounts(self.rows,CFG,'SKU002')
        self.assertEqual([c['discount'] for c in options],[.05,.1,.15,.2,.25])
        self.assertEqual(options[-1]['units'],options[-2]['units'])
        self.assertGreater(options[-2]['incremental_net_profit'],options[-1]['incremental_net_profit'])
        selected=plan(self.rows[1:2],dict(CFG,objective='profit'),[])['campaigns'][0]
        self.assertEqual(selected['discount'],.2)

    def test_declared_relationship_and_each_bundle_component(self):
        cfg=dict(CFG,mechanisms=['bundle'])
        self.assertFalse(plan(self.rows[:2],cfg,[])['campaigns'])
        rows=deepcopy(self.rows[:2]);rows[0]['stock']=5;rows[1]['stock']=3
        p=plan(rows,cfg,self.rel[:1]);self.assertEqual(p['campaigns'][0]['units'],{'SKU001':3,'SKU002':3})
        self.assertTrue(verify(p,rows)['valid'])
        corrupted=deepcopy(p);corrupted['campaigns'][0]['units']['SKU001']=4
        with self.assertRaisesRegex(ValueError,'equal quantities'):verify(corrupted,rows)

    def test_product_selection_responds_to_objective_with_fixed_budget(self):
        clearance=plan(self.rows[:2],dict(CFG,budget=200),self.rel[:1])
        profit=plan(self.rows[:2],dict(CFG,budget=200,objective='profit'),self.rel[:1])
        self.assertEqual(set(clearance['campaigns'][0]['units']),{'SKU001'})
        self.assertEqual(set(profit['campaigns'][0]['units']),{'SKU002'})

    def test_odd_stock_multi_buy_reserves_complete_pairs(self):
        rows=deepcopy(self.rows[:1]);rows[0]['stock']=5
        percentage=plan(rows,dict(CFG,mechanisms=['percentage']),[])['campaigns'][0]
        buy2=plan(rows,dict(CFG,mechanisms=['buy2']),[])['campaigns'][0]
        self.assertEqual(percentage['units']['SKU001'],5)
        self.assertEqual(buy2['units']['SKU001'],4)
        self.assertEqual(buy2['revenue'],4*650*(1-buy2['discount']))
        self.assertTrue(verify({'campaigns':[buy2],'config':dict(CFG,mechanisms=['buy2'])},rows)['valid'])

    def test_rejected_provider_attempt_executes_no_calculation_tool(self):
        with patch('agent.completion',side_effect=agent.ProviderError('rejected',code='tool_use_failed')),patch('agent.plan') as build,patch('agent.verify') as check,patch('agent.simulate') as simulate_tool:
            with self.assertRaises(agent.ProviderError):agent.run_agent('mock','mock','goal',self.rows,CFG,self.rel)
        build.assert_not_called();check.assert_not_called();simulate_tool.assert_not_called()

    def test_timeout_is_explicit_and_no_request_retry_or_fallback(self):
        with patch('agent.urllib.request.urlopen',side_effect=TimeoutError) as request:
            with self.assertRaises(agent.ProviderError):agent.completion('mock','mock',[])
        self.assertEqual(request.call_count,1)
        self.assertEqual(request.call_args.kwargs['timeout'],45)

    def test_disallowed_tool_cannot_execute(self):
        reply={'role':'assistant','tool_calls':[{'id':'bad','function':{'name':'launch_campaign','arguments':'{}'}}]}
        with patch('agent.completion',return_value=reply),patch('agent.plan') as build:
            with self.assertRaises(agent.ProviderError):agent.run_agent('mock','mock','goal',self.rows,CFG,self.rel)
        build.assert_not_called()

    def test_goal_does_not_silently_override_numeric_objective(self):
        plans=[]
        for goal in ('Clear stock','Maximize profit and use a larger budget'):
            replies=[{'role':'assistant','tool_calls':[{'id':name,'function':{'name':name,'arguments':'{}'}}]} for name in agent.TOOL_NAMES]
            replies.append({'role':'assistant','content':'Plan ready.'})
            with patch('agent.completion',side_effect=replies):
                p,_,_,_=agent.run_agent('mock','mock',goal,self.rows,CFG,self.rel)
            plans.append(p)
        self.assertEqual(plans[0],plans[1])
        self.assertEqual(plans[1]['config']['objective'],'clearance')

class AuditUITests(unittest.TestCase):
    def button(self,app,label):return next(b for b in app.button if b.label==label)
    def input(self,app,kind,label):return next(x for x in getattr(app,kind) if x.label==label)
    def upload_patch(self,data):
        class Upload:
            def getvalue(self):return data
        return patch('streamlit.file_uploader',side_effect=lambda label,**kwargs:Upload() if label.startswith('Upload catalogue') else None)

    def test_zero_budget_ui(self):
        app=AppTest.from_file(str(ROOT/'app.py'),default_timeout=10).run()
        self.input(app,'number_input','Marketing budget (₹)').set_value(0).run()
        self.button(app,'Generate / replan').click().run()
        self.assertEqual(len(app.exception),0)
        self.assertEqual(app.session_state['run']['result']['marketing_spend'],0)
        self.assertEqual(app.session_state['run']['result']['campaigns'],[])
        self.assertTrue(any('No improving feasible' in w.value for w in app.warning))

    def test_impossible_margin_ui(self):
        app=AppTest.from_file(str(ROOT/'app.py'),default_timeout=10).run()
        self.input(app,'slider','Minimum gross margin (%)').set_value(90).run()
        self.button(app,'Generate / replan').click().run()
        self.assertEqual(len(app.exception),0)
        self.assertEqual(app.session_state['run']['result']['status'],'no_feasible_improving_plan')

    def test_zero_stock_uploaded_catalogue_ui(self):
        with (ROOT/'data/catalogue.csv').open() as f:rows=list(csv.DictReader(f))
        for r in rows:r['stock']=0
        data=io.StringIO();writer=csv.DictWriter(data,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
        with self.upload_patch(data.getvalue().encode()):
            app=AppTest.from_file(str(ROOT/'app.py'),default_timeout=10).run()
            self.button(app,'Generate / replan').click().run()
        self.assertEqual(len(app.exception),0)
        self.assertEqual(app.session_state['run']['result']['units_promoted'],0)

    def test_invalid_csv_stops_before_planning(self):
        with self.upload_patch(b'sku,price\nSKU001,nan\n'):
            app=AppTest.from_file(str(ROOT/'app.py'),default_timeout=10).run()
        self.assertEqual(len(app.exception),0)
        self.assertTrue(any('Missing column' in e.value for e in app.error))
        self.assertNotIn('run',app.session_state)

    def test_live_failure_does_not_masquerade_as_sample(self):
        app=AppTest.from_file(str(ROOT/'app.py'),default_timeout=10).run()
        app.radio[0].set_value('Live Groq agent').run()
        with patch('agent.completion',side_effect=agent.ProviderError('explicit provider failure')):
            self.button(app,'Generate / replan').click().run()
        self.assertEqual(len(app.exception),0)
        self.assertNotIn('run',app.session_state)
        self.assertTrue(any('No silent fallback' in w.value for w in app.warning))

    def test_export_exact_effective_inputs_and_removed_evidence_new_run(self):
        from test_evidence import fixture_evidence
        captured=[];real_download=streamlit.download_button
        # Preserve positional arguments in the app's existing download calls.
        def capture(label,data,*args,**kwargs):
            if label=='Export complete plan and evidence':captured.append(json.loads(data))
            return real_download(label,data,*args,**kwargs)
        with patch('streamlit.download_button',side_effect=capture):
            app=AppTest.from_file(str(ROOT/'app.py'),default_timeout=10).run()
            app.session_state['evidence']=fixture_evidence();app.run()
            app.checkbox[0].check().run();self.button(app,'Apply reviewed evidence').click().run()
            self.button(app,'Generate / replan').click().run()
            payload=captured[-1]
            self.assertEqual(payload['source_evidence'],payload['run']['source_evidence'])
            self.assertEqual([r['competitor_price'] for r in payload['run']['effective_catalogue'][:2]],[520,640])
            self.assertEqual(payload['run']['result']['config']['margin'],.25)
            self.button(app,'Remove applied evidence').click().run()
            self.assertIsNotNone(captured[-1]['source_evidence'])
            self.button(app,'Generate / replan').click().run()
            self.assertIsNone(captured[-1]['source_evidence'])
            self.assertEqual([r['competitor_price'] for r in captured[-1]['run']['effective_catalogue'][:2]],[572,768])
        self.assertEqual(len(app.exception),0)

if __name__=='__main__':unittest.main()
