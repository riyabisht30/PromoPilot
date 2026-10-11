"""Explicit engine route and immutable-content human review binding."""
import hashlib,json
from copy import deepcopy
from datetime import datetime,timezone
from engine import plan,validate,verify,simulate,authoritative_summary
from agent import normalize_evidence,apply_evidence,ProviderError

def plan_with_quota_fallback(live_runner,rows,cfg,relationships,*,enabled=False):
    """Explicitly authorized local continuity; never claim the failed AI run completed."""
    try:
        return live_runner()
    except ProviderError as error:
        if not enabled or not (error.http_status==429 or error.code in ('rate_limit_exceeded','insufficient_quota')):
            raise
        result,scenarios,trace,summary=local_plan(rows,cfg,relationships)
        failure={'execution_route':'live_ai_agent','execution_status':'incomplete','category':error.category,'http_status':error.http_status,'code':error.code,'rate_limits':error.rate_limits}
        if hasattr(error,'partial_run'):failure['verified_partial_run']=deepcopy(error.partial_run)
        result['fallback']={'reason':'Groq rate/quota limit','failed_attempt':failure,'ai_workflow_completed':False}
        trace.insert(0,{'step':0,'tool':'live_ai_agent','status':'failed','execution_route':'live_ai_agent','explanation':'Groq rate/quota limit. Explicit local fallback executed; AI workflow did not complete.'})
        return result,scenarios,trace,summary

def evidence_identity(evidence):
    evidence=normalize_evidence(evidence)
    fields=('source','source_sha256','min_margin','max_discount','budget','competitor_prices','quotes','warnings','blocking_ambiguities','ambiguities')
    body={k:evidence.get(k) for k in fields}
    meta=evidence.get('extraction',{})
    body['extraction_model']=meta.get('model')
    body['schema_version']=meta.get('schema_version','legacy-normalized-v1')
    return hashlib.sha256(json.dumps(body,sort_keys=True,allow_nan=False).encode()).hexdigest()

def review_evidence(evidence,*,reviewed):
    if not reviewed:raise ValueError('Explicit human review is required.')
    result=normalize_evidence(evidence)
    result['review']={'reviewed':True,'evidence_sha256':evidence_identity(result),'source_sha256':result.get('source_sha256'),'schema_version':result.get('extraction',{}).get('schema_version','legacy-normalized-v1'),'extraction_model':result.get('extraction',{}).get('model'),'reviewed_at':datetime.now(timezone.utc).isoformat()}
    return result

def apply_reviewed(rows,cfg,evidence):
    review=evidence.get('review',{})
    if not review.get('reviewed') or review.get('evidence_sha256')!=evidence_identity(evidence):
        raise ValueError('Evidence or its source/schema/model changed: review again before applying.')
    return apply_evidence(rows,cfg,evidence,reviewed=True)

def local_plan(rows,cfg,relationships):
    """All four steps execute locally; no provider or implicit extraction calls."""
    rows=deepcopy(rows);cfg=deepcopy(cfg);validate(rows,cfg)
    known={r['sku'] for r in rows}
    if any(not {'sku_a','sku_b','reason'}.issubset(r) or r['sku_a']==r['sku_b'] or r['sku_a'] not in known or r['sku_b'] not in known for r in relationships):raise ValueError('Invalid relationship input.')
    result=plan(rows,cfg,relationships);checked=verify(result,rows)
    if not checked['valid']:raise ValueError('Plan did not pass constraint validation.')
    scenarios=simulate(result,rows)
    result['execution_status']='completed';result['execution_route']='local_engine'
    result['authoritative_summary']=authoritative_summary(result,rows)
    trace=[{'step':i+1,'tool':name,'status':'success','execution_route':'local_engine'} for i,name in enumerate(('inspect_catalogue','build_plan','validate_plan','simulate_plan'))]
    return result,scenarios,trace,result['authoritative_summary']['text']
