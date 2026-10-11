import unittest
from pathlib import Path

from streamlit.testing.v1 import AppTest
from agent import LEGACY_DEMO_DISCLOSURE
from test_evidence import fixture_evidence

ROOT=Path(__file__).resolve().parents[1]

class EvidenceUITests(unittest.TestCase):
    def button(self,app,label):
        return next(button for button in app.button if button.label==label)

    def test_review_warning_and_plan_provenance_snapshot(self):
        app=AppTest.from_file(str(ROOT/'app.py'),default_timeout=10).run()
        evidence=fixture_evidence()
        del evidence['warnings']; del evidence['blocking_ambiguities']
        evidence['ambiguities']=[LEGACY_DEMO_DISCLOSURE]
        app.session_state['evidence']=evidence
        app.run()
        self.assertEqual(len(app.exception),0)
        self.assertTrue(self.button(app,'Apply reviewed evidence').disabled)
        self.assertTrue(any(LEGACY_DEMO_DISCLOSURE in warning.value for warning in app.warning))
        app.checkbox[0].check().run()
        self.button(app,'Apply reviewed evidence').click().run()
        self.assertEqual(len(app.error),0)
        applied=app.session_state['applied']
        self.assertEqual(applied['warnings'],[LEGACY_DEMO_DISCLOSURE])
        self.button(app,'Generate / replan').click().run()
        self.assertEqual(len(app.exception),0)
        self.assertEqual(len(app.error),0)
        run=app.session_state['run']
        self.assertEqual(run['source_evidence']['source'],'competitor.png')
        self.assertEqual(run['source_evidence']['warnings'],[LEGACY_DEMO_DISCLOSURE])
        self.assertEqual(run['source_evidence']['competitor_prices'][1]['price'],640)
        # Removing active evidence must not change an existing plan's export provenance.
        self.button(app,'Remove applied evidence').click().run()
        self.assertEqual(app.session_state['run']['source_evidence'],run['source_evidence'])
        self.assertTrue(any('Applied evidence warning:' in warning.value for warning in app.warning))

    def test_review_does_not_bypass_price_conflict(self):
        app=AppTest.from_file(str(ROOT/'app.py'),default_timeout=10).run()
        evidence=fixture_evidence()
        evidence['blocking_ambiguities']=['SKU001 has conflicting prices']
        app.session_state['evidence']=evidence
        app.run()
        app.checkbox[0].check().run()
        self.button(app,'Apply reviewed evidence').click().run()
        self.assertEqual(len(app.exception),0)
        self.assertTrue(any('Resolve blocking extraction ambiguities' in error.value for error in app.error))
        self.assertNotIn('applied',app.session_state)

if __name__=='__main__': unittest.main()
