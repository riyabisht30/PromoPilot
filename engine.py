"""Finite offer search + deterministic constrained portfolio heuristic. Currency: INR."""
import math
from copy import deepcopy

FIELDS = ('sku','name','stock','price','cost','daily_sales','elasticity','competitor_price','segment','ad_cost')
NUMERIC = FIELDS[2:8] + ('ad_cost',)

def validate(rows, cfg):
    if not rows: raise ValueError('Catalogue is empty.')
    if len(rows)>500: raise ValueError('Maximum 500 products.')
    seen=set()
    for r in rows:
        for f in FIELDS:
            if f not in r: raise ValueError(f'Missing column: {f}')
        if not r['sku'] or r['sku'] in seen: raise ValueError('SKU must be nonempty and unique.')
        seen.add(r['sku'])
        for f in NUMERIC:
            v=float(r[f])
            if not math.isfinite(v) or v<0: raise ValueError(f'{r["sku"]}: invalid {f}')
            r[f]=v
        if r['stock'] != int(r['stock']): raise ValueError('Stock must be a whole number.')
        if r['price']<=0 or r['competitor_price']<=0: raise ValueError('Prices must be positive.')
    for f in ('budget','margin','max_discount','days','holiday_factor'):
        if f not in cfg or not math.isfinite(float(cfg[f])): raise ValueError(f'Invalid {f}')
    if cfg['budget']<0 or not 0<=cfg['margin']<1 or not 0<=cfg['max_discount']<1: raise ValueError('Invalid budget/margin/discount.')
    if not 1<=cfg['days']<=30 or cfg['holiday_factor']<=0: raise ValueError('Duration must be 1–30 days; holiday factor positive.')
    if cfg.get('objective') not in ('profit','clearance'): raise ValueError('Choose profit or clearance objective.')

def forecast(r, days, discount, holiday, factor=1):
    # Assumed constant elasticity and bounded competitor sensitivity, not learned causal uplift.
    competitor=max(.8,min(1.2,r['competitor_price']/max(.01,r['price']*(1-discount))))
    return r['daily_sales']*days*holiday*((1-discount)**(-r['elasticity']))*competitor*factor

def plan(catalogue, config, relationships=()):
    rows=deepcopy(catalogue); cfg=deepcopy(config); validate(rows,cfg)
    by={r['sku']:r for r in rows}; candidates=[]
    discounts=[d for d in (.05,.10,.15,.20,.25,.30) if d<=cfg['max_discount']+1e-9]
    durations=sorted(set((min(3,int(cfg['days'])), min(7,int(cfg['days'])),int(cfg['days']))))
    for r in rows:
        for days in durations:
            for mechanism in cfg.get('mechanisms', ('percentage','buy2')):
                if mechanism not in ('percentage','buy2'): continue
                for d in discounts:
                    price=r['price']*(1-d)
                    if (price-r['cost'])/price+1e-9<cfg['margin']: continue
                    unit_count=min(int(r['stock']),int(forecast(r,days,d,cfg['holiday_factor'])))
                    if mechanism=='buy2': unit_count=(unit_count//2)*2
                    if unit_count<=0: continue
                    candidates.append(_candidate([r],{r['sku']:unit_count},d,days,mechanism,cfg))
    for rel in relationships if 'bundle' in cfg.get('mechanisms', ('percentage','buy2','bundle')) else ():
        a,b=rel.get('sku_a'),rel.get('sku_b')
        if a not in by or b not in by or a==b: continue
        products=[by[a],by[b]]
        for days in durations:
            for d in discounts:
                # Require each component's margin too, not just aggregate bundle margin.
                if any((r['price']*(1-d)-r['cost'])/(r['price']*(1-d))+1e-9<cfg['margin'] for r in products): continue
                pairs=min(min(int(r['stock']),int(forecast(r,days,d,cfg['holiday_factor']))) for r in products)
                if pairs>0:
                    c=_candidate(products,{a:pairs,b:pairs},d,days,'bundle',cfg)
                    c['relationship']=rel.get('reason','Declared complementary products')
                    candidates.append(c)
    # At most one campaign per SKU: avoids overlapping-stock and demand double counting.
    candidates.sort(key=lambda c:(c['score']/max(1,c['marketing_spend']),c['score']),reverse=True)
    selected=[]; used=set(); spent=0
    for c in candidates:
        if c['score']<=0 or used.intersection(c['units']): continue
        if spent+c['marketing_spend']>cfg['budget']+1e-9: continue
        selected.append(c); used.update(c['units']); spent+=c['marketing_spend']
    result={'status':'feasible' if selected else 'no_feasible_improving_plan','config':cfg,'campaigns':selected,
            'candidate_count':len(candidates),'marketing_spend':round(spent,2),
            'units_promoted':sum(sum(c['units'].values()) for c in selected),
            'incremental_net_profit':round(sum(c['incremental_net_profit'] for c in selected),2),
            'method':'Finite offer enumeration; greedy portfolio heuristic. Global optimality is not claimed.',
            'assumptions':'Synthetic constant-elasticity demand; no measured causal uplift, cannibalization, or replenishment.',
            'warnings':[]}
    if not selected: result['warnings'].append('No improving feasible offer. Increase budget, relax constraints, or retain no-promotion baseline.')
    verify(result,rows)
    return result

def authoritative_summary(result, rows):
    """Engine-backed facts only; inventory denominator includes the entire input catalogue."""
    checked=verify(result,rows)
    available=sum(int(r['stock']) for r in rows)
    return {'objective':result['config']['objective'],'selection_method':result['method'],
            'global_optimality_proven':False,'promoted_units':result['units_promoted'],
            'available_stock_all_input_skus':available,
            'estimated_incremental_net_profit':result['incremental_net_profit'],
            'demand_assumptions':result['assumptions'],'constraint_checks':checked,
            'text':f"Configured objective: {result['config']['objective']}. {result['method']} "
                   f"Estimated promoted units: {result['units_promoted']:,} / {available:,} available units across all input SKUs. "
                   f"Estimated incremental net profit versus the same-products no-promotion baseline: INR {result['incremental_net_profit']:,.2f}. "
                   f"{result['assumptions']} Constraint checks passed: {', '.join(checked['checks'])}."}

def _candidate(products,units,d,days,mechanism,cfg):
    revenue=sum(r['price']*(1-d)*units[r['sku']] for r in products)
    cost=sum(r['cost']*units[r['sku']] for r in products)
    baseline=sum((r['price']-r['cost'])*min(r['stock'],int(forecast(r,days,0,cfg['holiday_factor']))) for r in products)
    spend=sum(r['ad_cost'] for r in products)
    incremental=revenue-cost-spend-baseline
    score=incremental if cfg['objective']=='profit' else sum(units.values())
    return {'products':', '.join(r['name'] for r in products),'units':units,'mechanism':mechanism,
            'discount':d,'days':days,'segment':', '.join(sorted(set(r['segment'] for r in products))),
            'revenue':round(revenue,2),'gross_profit':round(revenue-cost,2),'gross_margin':(revenue-cost)/revenue,
            'baseline_gross_profit':round(baseline,2),'marketing_spend':spend,'incremental_net_profit':round(incremental,2),
            'score':score,'competitor_prices':{r['sku']:r['competitor_price'] for r in products},
            'offer':f'{d:.0%} off' if mechanism=='percentage' else (f'Buy 2, {d:.0%} off the pair' if mechanism=='buy2' else f'Bundle, {d:.0%} off both'),
            'stock_pressure':{r['sku']:round(r['stock']/max(.01,r['daily_sales']),1) for r in products}}

def verify(result,rows):
    rows=deepcopy(rows); validate(rows,result['config'])
    by={r['sku']:r for r in rows}; cfg=result['config']; used=set(); total=0
    for c in result['campaigns']:
        if used.intersection(c['units']): raise ValueError('Overlapping campaigns.')
        used.update(c['units']); total+=c['marketing_spend']
        if c['discount']>cfg['max_discount']+1e-9: raise ValueError('Discount cap violated.')
        if c['mechanism'] not in cfg.get('mechanisms',('percentage','buy2','bundle')): raise ValueError('Offer mechanism not allowed.')
        if c['gross_margin']+1e-9<cfg['margin']: raise ValueError('Margin violated.')
        for sku,n in c['units'].items():
            if n<0 or n>by[sku]['stock'] or n!=int(n): raise ValueError('Stock violated.')
        if c['mechanism']=='buy2' and any(n%2 for n in c['units'].values()): raise ValueError('Multi-buy quantity invalid.')
        if c['mechanism']=='bundle' and (len(c['units'])!=2 or len(set(c['units'].values()))!=1):
            raise ValueError('Bundle must consume equal quantities of two components.')
        for sku in c['units']:
            price=by[sku]['price']*(1-c['discount'])
            if (price-by[sku]['cost'])/price+1e-9<cfg['margin']: raise ValueError('Component margin violated.')
    if total>cfg['budget']+1e-9: raise ValueError('Marketing budget violated.')
    return {'valid':True,'checks':['stock','margin','component margins','budget','discount cap','allowed mechanisms','campaign overlap','multi-buy quantities','bundle component quantities']}

def compare_discounts(catalogue,config,sku,days=None,mechanism='percentage'):
    """Expose feasible individual offers, not a claim of portfolio optimality."""
    rows=deepcopy(catalogue);cfg=deepcopy(config);validate(rows,cfg)
    if mechanism not in ('percentage','buy2') or mechanism not in cfg.get('mechanisms',('percentage','buy2','bundle')):
        return []
    product=next((r for r in rows if r['sku']==sku),None)
    if product is None: raise ValueError('Unknown comparison SKU.')
    days=int(cfg['days'] if days is None else days)
    if not 1<=days<=cfg['days']: raise ValueError('Invalid comparison duration.')
    output=[]
    for discount in (.05,.10,.15,.20,.25,.30):
        price=product['price']*(1-discount)
        if discount>cfg['max_discount']+1e-9 or (price-product['cost'])/price+1e-9<cfg['margin'] or product['ad_cost']>cfg['budget']: continue
        units=min(int(product['stock']),int(forecast(product,days,discount,cfg['holiday_factor'])))
        if mechanism=='buy2': units=units//2*2
        if units>0: output.append(_candidate([product],{sku:units},discount,days,mechanism,cfg))
    return output

def simulate(result,rows):
    rows=deepcopy(rows); validate(rows,result['config'])
    by={r['sku']:r for r in rows}; cfg=result['config']; output=[]
    for label,factor in [('Conservative',.7),('Base',1),('Optimistic',1.3)]:
        revenue=cost=baseline=units=0
        for c in result['campaigns']:
            expected={s:min(int(by[s]['stock']),int(forecast(by[s],c['days'],c['discount'],cfg['holiday_factor'],factor))) for s in c['units']}
            if c['mechanism']=='bundle': expected={s:min(expected.values()) for s in expected}
            if c['mechanism']=='buy2': expected={s:(n//2)*2 for s,n in expected.items()}
            for s,n in expected.items():
                r=by[s]; units+=n; revenue+=n*r['price']*(1-c['discount']); cost+=n*r['cost']
                baseline+=min(int(r['stock']),int(forecast(r,c['days'],0,cfg['holiday_factor'],factor)))*(r['price']-r['cost'])
        output.append({'scenario':label,'units':units,'revenue':round(revenue,2),'gross_profit':round(revenue-cost,2),
                       'no_promotion_gross_profit_same_products':round(baseline,2),
                       'incremental_net_profit':round(revenue-cost-result['marketing_spend']-baseline,2)})
    return output
