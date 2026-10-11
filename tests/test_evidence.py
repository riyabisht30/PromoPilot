import csv
import json
import math
import unittest
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

from agent import (LEGACY_DEMO_DISCLOSURE, apply_evidence, extract_evidence,
                   normalize_evidence)

ROOT=Path(__file__).resolve().parents[1]

def fixture_evidence():
    return {
        'min_margin':None,'max_discount':None,'budget':None,
        'competitor_prices':[
            {'sku':'SKU001','price':520,'observed_date':'2026-10-11','quote':'SKU001 INR 520'},
            {'sku':'SKU002','price':640,'observed_date':'2026-10-11','quote':'SKU002 INR 640'},
        ],
        'quotes':[], 'warnings':['Demo fixture; not a live retailer listing'],
        'blocking_ambiguities':[], 'source':'competitor.png',
    }

class EvidenceTests(unittest.TestCase):
    def setUp(self):
        with (ROOT/'data/catalogue.csv').open() as f: self.rows=list(csv.DictReader(f))
        self.cfg={'margin':.25,'max_discount':.25,'budget':3000}
        self.evidence=fixture_evidence()

    def test_demo_fixture_applies_only_after_review_and_retains_provenance(self):
        with self.assertRaisesRegex(ValueError,'human review'):
            apply_evidence(self.rows,self.cfg,self.evidence)
        original=deepcopy(self.evidence)
        rows,cfg=apply_evidence(self.rows,self.cfg,self.evidence,reviewed=True)
        self.assertEqual([r['competitor_price'] for r in rows[:2]],[520,640])
        self.assertEqual(cfg,self.cfg)
        self.assertEqual(self.evidence,original)
        exported=json.loads(json.dumps({'source_evidence':normalize_evidence(self.evidence)}))['source_evidence']
        self.assertEqual(exported['warnings'],self.evidence['warnings'])
        self.assertEqual(exported['source'],'competitor.png')
        self.assertEqual(exported['competitor_prices'][0]['observed_date'],'2026-10-11')
        self.assertEqual(exported['competitor_prices'][0]['quote'],'SKU001 INR 520')

    def test_demo_with_genuine_conflict_still_blocks(self):
        for message in ('SKU001 price unreadable','SKU002 has conflicting prices','SKU001 price depends on unclear bundle conditions'):
            with self.subTest(message=message):
                evidence=deepcopy(self.evidence)
                evidence['blocking_ambiguities']=[message]
                with self.assertRaisesRegex(ValueError,'blocking extraction'):
                    apply_evidence(self.rows,self.cfg,evidence,reviewed=True)

    def test_legacy_only_known_disclosure_becomes_warning(self):
        evidence=deepcopy(self.evidence)
        del evidence['warnings']; del evidence['blocking_ambiguities']
        evidence['ambiguities']=[LEGACY_DEMO_DISCLOSURE]
        normalized=normalize_evidence(evidence)
        self.assertEqual(normalized['warnings'],[LEGACY_DEMO_DISCLOSURE])
        self.assertEqual(normalized['ambiguities'],[LEGACY_DEMO_DISCLOSURE])
        apply_evidence(self.rows,self.cfg,normalized,reviewed=True)
        for message in ('Unclear price','Synthetic document also has conflicting SKU001 prices','Some unknown warning'):
            with self.subTest(message=message):
                mixed=deepcopy(evidence); mixed['ambiguities'].append(message)
                self.assertIn(message,normalize_evidence(mixed)['blocking_ambiguities'])
                with self.assertRaises(ValueError): apply_evidence(self.rows,self.cfg,mixed,reviewed=True)

    def test_unknown_and_duplicate_skus_block(self):
        for sku in ('UNKNOWN','SKU001'):
            with self.subTest(sku=sku):
                evidence=deepcopy(self.evidence)
                evidence['competitor_prices'][1]['sku']=sku
                with self.assertRaisesRegex(ValueError,'Unknown or duplicate'):
                    apply_evidence(self.rows,self.cfg,evidence,reviewed=True)

    def test_absent_policy_fields_remain_null_and_do_not_override(self):
        for omitted in (False,True):
            evidence=deepcopy(self.evidence)
            if omitted:
                for field in ('min_margin','max_discount','budget'): del evidence[field]
            normalized=normalize_evidence(evidence)
            for field in ('min_margin','max_discount','budget'): self.assertIsNone(normalized[field])
            _,cfg=apply_evidence(self.rows,self.cfg,normalized,reviewed=True)
            self.assertEqual(cfg,self.cfg)

    def test_invalid_prices_and_missing_quotes_block(self):
        for value in (0,-1,float('nan'),float('inf'),-float('inf'),True,'520',None):
            with self.subTest(value=value):
                evidence=deepcopy(self.evidence); evidence['competitor_prices'][0]['price']=value
                with self.assertRaises(ValueError): apply_evidence(self.rows,self.cfg,evidence,reviewed=True)
        for quote in ('','   ',None,12):
            evidence=deepcopy(self.evidence); evidence['competitor_prices'][0]['quote']=quote
            with self.assertRaises(ValueError): apply_evidence(self.rows,self.cfg,evidence,reviewed=True)

    def test_malformed_messages_are_not_silently_dropped(self):
        for field in ('warnings','blocking_ambiguities','ambiguities','quotes'):
            evidence=deepcopy(self.evidence); evidence[field]='not a list'
            with self.assertRaises(ValueError): normalize_evidence(evidence)

    def test_extraction_uses_new_contract_and_hashes_original_source(self):
        response=deepcopy(self.evidence); del response['source']
        with patch('agent.completion',return_value={'content':json.dumps(response)}) as provider:
            evidence=extract_evidence('mock','text','vision',(ROOT/'data/competitor.png').read_bytes(),'competitor.png')
        prompt=provider.call_args.args[2][0]['content'][0]['text']
        self.assertIn('blocking_ambiguities',prompt)
        self.assertIn('Synthetic documents can still contain genuine conflicts',prompt)
        self.assertEqual(evidence['source'],'competitor.png')
        self.assertEqual(len(evidence['source_sha256']),64)
        self.assertIsNone(evidence['min_margin'])
        self.assertEqual(evidence['warnings'],response['warnings'])

    def test_policy_quotes_and_finite_values_still_required(self):
        evidence=deepcopy(self.evidence); evidence['budget']=1500
        with self.assertRaisesRegex(ValueError,'source quotes'): normalize_evidence(evidence)
        evidence['quotes']=['Marketing budget INR 1500']
        _,cfg=apply_evidence(self.rows,self.cfg,evidence,reviewed=True)
        self.assertEqual(cfg['budget'],1500)
        evidence['budget']=math.inf
        with self.assertRaises(ValueError): normalize_evidence(evidence)

if __name__=='__main__': unittest.main()
