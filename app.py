import csv, io, json, os
from copy import deepcopy
from pathlib import Path
import streamlit as st
import pandas as pd
from engine import plan, simulate, validate, compare_discounts, authoritative_summary
from agent import run_agent, extract_evidence, apply_evidence, normalize_evidence, ProviderError
from workflow import local_plan,review_evidence,apply_reviewed,evidence_identity,plan_with_quota_fallback

ROOT=Path(__file__).parent
st.set_page_config(page_title='PromoPilot',page_icon='🛍️',layout='wide')
st.markdown('''<style>.stApp{background:#f7f9fc}h1,h2,h3{color:#152a46}.stButton>button{border-radius:9px}div[data-testid="stMetric"]{background:white;padding:18px;border-radius:12px;border:1px solid #e0e7ef}</style>''',unsafe_allow_html=True)
st.caption('PROMOPILOT / AGENTIC RETAIL PLANNING')
st.title('Better promotions. Clearer decisions.')
st.write('Plan offers, check commercial constraints, and adapt when conditions change.')
st.info('Hackathon prototype • Sample data and demand assumptions are synthetic • Plans are simulated, never launched')

def secret(name,default=''):
    try: return st.secrets.get(name,os.environ.get(name,default))
    except Exception: return os.environ.get(name,default)

with st.sidebar:
    st.header('Planning controls')
    mode=st.radio('Execution',['Local engine (no API)','Live Groq agent'])
    st.caption('Local engine accepts structured inputs and reviewed AI evidence. No API request or autonomous AI execution is claimed for this route.')
    objective=st.selectbox('Objective',['clearance','profit'])
    mechanisms=st.multiselect('Allowed offer mechanisms',['percentage','buy2','bundle'],default=['percentage','buy2','bundle'])
    budget=st.number_input('Marketing budget (₹)',min_value=0,value=3000,step=100)
    margin=st.slider('Minimum gross margin (%)',0,90,25)/100
    discount=st.slider('Maximum discount (%)',0,60,25)/100
    days=st.slider('Maximum campaign duration (days)',1,30,7)
    holiday=st.slider('Assumed holiday demand factor',.5,2.0,1.2,.1)
    st.caption('Budget is campaign advertising spend. Discounts reduce revenue; they are not counted as advertising spend.')
    with st.expander('Provider configuration'):
        text_model=st.text_input('Text/tool model',secret('GROQ_MODEL','openai/gpt-oss-20b'))
        vision_model=st.text_input('Vision model',secret('GROQ_VISION_MODEL','qwen/qwen3.8-27b'))
        run_deadline=st.number_input('Run deadline (seconds)',min_value=10,max_value=600,value=240)
        st.caption('Model access can vary by account. Key is read only from environment or Streamlit secrets.')
        quota_behavior=st.selectbox('If Groq planning quota runs out',['Use local engine and label the failed AI attempt','Stop and retain verified work'])
        st.caption('Quota fallback calculates from current inputs and reviewed evidence. It does not extract new documents or complete the AI workflow.')
key=secret('GROQ_API_KEY')
cfg={'objective':objective,'budget':float(budget),'margin':margin,'max_discount':discount,'days':days,'holiday_factor':holiday,'mechanisms':mechanisms}
base=list(csv.DictReader(io.StringIO((ROOT/'data/catalogue.csv').read_text(encoding='utf-8'))))
relationships=list(csv.DictReader(io.StringIO((ROOT/'data/relationships.csv').read_text(encoding='utf-8'))))
tabs=st.tabs(['Plan & adapt','Data & evidence','About & validation'])
with tabs[1]:
    st.subheader('Catalogue')
    st.caption('Catalogue CSV and sidebar values are structured/manual inputs. CSV data may be edited manually before upload; these inputs are not AI extraction.')
    uploaded=st.file_uploader('Upload catalogue CSV (or use sample below)',type=['csv'])
    try:
        rows=list(csv.DictReader(io.StringIO(uploaded.getvalue().decode('utf-8-sig')))) if uploaded else base
        validate(rows,cfg)
    except Exception as e:
        st.error(str(e)); st.stop()
    st.dataframe(pd.DataFrame(rows),use_container_width=True,hide_index=True)
    # Sample pairs outside an uploaded catalogue are not declarations for that scope.
    known_skus={r['sku'] for r in rows}
    relationships=[r for r in relationships if r['sku_a'] in known_skus and r['sku_b'] in known_skus]
    st.download_button('Download sample catalogue',(ROOT/'data/catalogue.csv').read_bytes(),'catalogue.csv','text/csv')
    st.caption('Relationships are explicitly supplied complementary pairs, not learned purchase associations.')
    st.dataframe(pd.DataFrame(relationships),hide_index=True)
    rel_upload=st.file_uploader('Optional relationship CSV: sku_a,sku_b,reason',type=['csv'])
    if rel_upload:
        relationships=list(csv.DictReader(io.StringIO(rel_upload.getvalue().decode('utf-8-sig'))))
        if any(not {'sku_a','sku_b','reason'}.issubset(r) or r['sku_a']==r['sku_b'] or r['sku_a'] not in {x['sku'] for x in rows} or r['sku_b'] not in {x['sku'] for x in rows} for r in relationships):
            st.error('Invalid relationships: use two distinct catalogue SKUs and a reason.'); st.stop()
    st.subheader('Document & image evidence')
    st.write('Extract policy limits or competitor prices. Review source quotes before applying. Applying evidence changes the next plan.')
    evidence_file=st.file_uploader('Policy PDF/TXT or competitor/scanned-page PNG/JPG',type=['pdf','txt','png','jpg','jpeg'])
    if st.button('Extract evidence',disabled=evidence_file is None):
        if not key: st.error('Set GROQ_API_KEY in secrets. Extraction requires a live model; no sample extraction is substituted.')
        else:
            try:
                with st.spinner('Extracting source evidence…'):
                    st.session_state['evidence']=extract_evidence(key,text_model,vision_model,evidence_file.getvalue(),evidence_file.name,deadline_seconds=run_deadline,cache=st.session_state.setdefault('evidence_cache',{}),stop_on_rate_limit=True)
                st.session_state['reviewed_evidence']=False
                st.session_state.pop('applied',None)
            except Exception as e:
                st.error(f"Extraction failed ({getattr(e,'category','unknown')}): {e}")
                if getattr(e,'rate_limits',{}):st.json({'provider_rate_limits':e.rate_limits})
                st.info('Local planning remains available. Enter structured settings/CSV manually or reuse previously reviewed evidence; failed extraction is not applied.')
    if 'evidence' in st.session_state:
        evidence=st.session_state['evidence']; evidence_ready=True
        try:
            evidence=normalize_evidence(evidence)
            st.session_state['evidence']=evidence
            identity=evidence_identity(evidence)
            if st.session_state.get('evidence_identity')!=identity:
                st.session_state['reviewed_evidence']=False
                st.session_state['evidence_identity']=identity
        except ValueError as e:
            st.error(str(e)); evidence_ready=False
        st.json(evidence)
        for warning in evidence.get('warnings',[]) if evidence_ready else []: st.warning('Evidence warning: '+warning)
        for ambiguity in evidence.get('blocking_ambiguities',[]) if evidence_ready else []: st.error('Blocking ambiguity: '+ambiguity)
        reviewed=st.checkbox('I reviewed the quotes, product matches, dates, and policy values',key='reviewed_evidence')
        if st.button('Apply reviewed evidence',disabled=not reviewed or not evidence_ready):
            try:
                apply_evidence(rows,cfg,evidence,reviewed=reviewed)
                st.session_state['applied']=review_evidence(evidence,reviewed=reviewed)
                st.success('Evidence will override the relevant inputs when the next plan runs.')
            except Exception as e: st.error(str(e))
    if 'applied' in st.session_state:
        st.warning('Reviewed evidence is active and overrides relevant sidebar values. Remove it to use sidebar values alone.')
        if st.button('Remove applied evidence'): st.session_state.pop('applied'); st.rerun()
    st.download_button('Download competitor image fixture',(ROOT/'data/competitor.png').read_bytes(),'competitor.png','image/png')
    st.download_button('Download policy fixture',(ROOT/'data/policy.txt').read_bytes(),'policy.txt','text/plain')
with tabs[0]:
    goal=st.text_area('Business goal','Clear excess apparel stock before the holiday. Compare offers within the configured margin and marketing budget.')
    st.caption('Sidebar/reviewed policy settings control the calculations. In live mode, the goal guides tool coordination; it does not change the numeric objective, product scope, or constraints. Numerical summaries come from the engine. Use sidebar controls and catalogue data for those choices.')
    change=st.selectbox('Disruption to apply before planning',['None','Marketing budget cut by 50%','Competitor prices cut by 20%','First product becomes unavailable'])
    if st.button('Generate / replan',type='primary'):
        try:
            actual_rows=rows; actual_cfg=cfg; actual_evidence=None
            if 'applied' in st.session_state:
                actual_evidence=normalize_evidence(st.session_state['applied'])
                actual_rows,actual_cfg=apply_reviewed(rows,cfg,actual_evidence)
            from copy import deepcopy
            actual_rows=deepcopy(actual_rows); actual_cfg=deepcopy(actual_cfg)
            if change=='Marketing budget cut by 50%': actual_cfg['budget']*=.5
            elif change=='Competitor prices cut by 20%':
                for r in actual_rows: r['competitor_price']*=.8
            elif change=='First product becomes unavailable': actual_rows[0]['stock']=0
            with st.spinner('Building and checking promotion plan…'):
                if mode=='Live Groq agent':
                    if not key: raise ValueError('Set GROQ_API_KEY in Streamlit secrets before running live mode.')
                    result,scenarios,trace,summary=plan_with_quota_fallback(lambda:run_agent(key,text_model,goal,actual_rows,actual_cfg,relationships,deadline_seconds=run_deadline,stop_on_rate_limit=True),actual_rows,actual_cfg,relationships,enabled=quota_behavior.startswith('Use local'))
                else:
                    result,scenarios,trace,summary=local_plan(actual_rows,actual_cfg,relationships)
            result['authoritative_summary']=authoritative_summary(result,actual_rows)
            summary=result['authoritative_summary']['text']
            if 'run' in st.session_state: st.session_state['previous']=st.session_state['run']['result']
            route=result.get('execution_route') or ('live_ai_agent' if mode=='Live Groq agent' else 'local_engine')
            display_mode='Local engine (Groq quota fallback)' if result.get('fallback') else mode
            result['execution_route']=route
            st.session_state['run']={'result':result,'scenarios':scenarios,'trace':trace,'summary':summary,'mode':display_mode,'requested_mode':mode,'execution_route':route,'execution_status':'completed','ai_workflow_completed':False if result.get('fallback') else route=='live_ai_agent','input_origin':'structured/manual inputs with reviewed AI evidence' if actual_evidence else 'structured/manual inputs','evidence_origin':'AI-extracted reviewed evidence' if actual_evidence else None,'disruption':change,'source_evidence':deepcopy(actual_evidence),'effective_catalogue':deepcopy(actual_rows)}
        except Exception as e:
            partial=getattr(e,'partial_run',None)
            if partial:
                from copy import deepcopy
                st.session_state['run']={**partial,'mode':mode,'execution_route':'live_ai_agent','disruption':change,'source_evidence':deepcopy(actual_evidence),'effective_catalogue':deepcopy(actual_rows)}
            else:
                if 'run' in st.session_state:st.session_state['last_verified_run']=deepcopy(st.session_state['run'])
                st.session_state.pop('run',None)
            st.error(f"Run failed ({getattr(e,'category','unknown')}): {e}"); st.warning('This run did not complete. No silent fallback was used. Select Local engine explicitly to continue without the provider.')
            if getattr(e,'rate_limits',{}):st.json({'provider_rate_limits':e.rate_limits})
    if 'last_verified_run' in st.session_state:
        with st.expander('Previous verified result retained separately'):
            st.caption('This is a previous run, not a successful result for the failed attempt.')
            st.json(st.session_state['last_verified_run'])
            st.download_button('Export previous verified run',json.dumps(st.session_state['last_verified_run'],indent=2),'previous_verified_plan.json','application/json')
    if 'run' in st.session_state:
        run=st.session_state['run']; r=run['result']
        st.caption(f'Execution: {run["mode"]} | Disruption: {run["disruption"]}')
        if r.get('fallback'):st.warning('Groq reached its rate/quota limit. The AI attempt did not complete; the displayed plan was calculated and validated by the local engine. New document extraction still requires AI access.')
        st.caption('Inputs: '+run.get('input_origin','Previously executed inputs')+' | Evidence: '+str(run.get('evidence_origin') or 'No AI evidence applied'))
        if run.get('execution_status')=='incomplete':
            st.warning('Incomplete agent execution. Previously executed, verified calculations are retained below; this is not a completed workflow. '+run.get('failure_message',''))
        a,b,c,d=st.columns(4)
        a.metric('Campaigns',len(r['campaigns'])); b.metric('Units promoted',r['units_promoted']); c.metric('Marketing spend',f'₹{r["marketing_spend"]:,.0f}'); d.metric('Est. incremental net profit',f'₹{r["incremental_net_profit"]:,.0f}')
        st.write(authoritative_summary(r,run['effective_catalogue'])['text'])
        if r.get('model_explanation'): st.caption(r['model_explanation']['reason'])
        for step in run['trace']:
            if step.get('status')=='recovery needed': st.warning('Tool recovery: '+step.get('explanation','A tool call needed recovery; no diagnostic was retained for this earlier run.'))
        for warning in (run.get('source_evidence') or {}).get('warnings',[]): st.warning('Applied evidence warning: '+warning)
        for warning in r['warnings']: st.warning(warning)
        st.subheader('Promotion portfolio')
        table=[{k:c[k] for k in ('products','offer','days','segment','units','gross_margin','marketing_spend','incremental_net_profit')} for c in r['campaigns']]
        st.dataframe(pd.DataFrame(table),hide_index=True,use_container_width=True)
        if run.get('effective_catalogue'):
            with st.expander('Feasible discount comparisons'):
                comparison_sku=st.selectbox('Compare individual offers for SKU',[row['sku'] for row in run['effective_catalogue']],index=1 if len(run['effective_catalogue'])>1 else 0)
                choices=compare_discounts(run['effective_catalogue'],r['config'],comparison_sku)
                st.caption('Same configured duration, percentage offers. Individual feasibility does not guarantee inclusion in the portfolio. Bundles and buy-two are compared by changing allowed mechanisms and replanning.')
                st.dataframe(pd.DataFrame([{k:c[k] for k in ('discount','days','units','revenue','gross_profit','baseline_gross_profit','marketing_spend','incremental_net_profit','score')} for c in choices]),hide_index=True)
        st.subheader('Demand sensitivity / same selected products')
        st.dataframe(pd.DataFrame(run['scenarios']),hide_index=True,use_container_width=True)
        if 'previous' in st.session_state:
            prev=st.session_state['previous']; st.write(f'Previous run: {prev["units_promoted"]} units, ₹{prev["marketing_spend"]:,.0f} advertising. Current: {r["units_promoted"]} units, ₹{r["marketing_spend"]:,.0f}.')
        with st.expander('Tool activity and effective inputs'):
            st.dataframe(pd.DataFrame(run['trace']),hide_index=True); st.json(r['config']); st.caption(r['assumptions'])
        st.download_button('Export complete plan and evidence',json.dumps({'run':run,'source_evidence':run.get('source_evidence')},indent=2),'promotion_plan.json','application/json')
with tabs[2]:
    st.subheader('How PromoPilot works')
    st.write('Choose products and offers within stock, margin, discount and advertising constraints. Compare estimated outcomes with a no-promotion baseline. AI evidence must be validated and explicitly reviewed before use.')
    st.caption('Local calculation and live AI execution have separate labels. Seven audited features support F3; D2 remains provisional and D3 is unverified. Demand is assumed and selection is heuristic.')
    st.write('To try the prototype: generate a local sample plan, change the budget, inspect feasible discounts and sensitivity, then export the result. AI document extraction requires your own provider key and explicit review.')
    st.caption('Architecture, installation and environment instructions are included in the repository README. No campaigns are launched automatically.')
