#!/usr/bin/env python3
from __future__ import annotations
import argparse,json,itertools
from pathlib import Path
import numpy as np,pandas as pd

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--exposure',required=True);ap.add_argument('--model-cells',required=True);ap.add_argument('--chem',required=True);ap.add_argument('--out',required=True);a=ap.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
 e=pd.read_csv(a.exposure,low_memory=False);m=pd.read_csv(a.model_cells,low_memory=False);c=pd.read_csv(a.chem,low_memory=False)
 m=m[m.mle_status.astype(str).str.startswith('identified') & m.effect_family.isin(['mortality_survival','immobilization_intoxication','growth','reproduction','development_morphology'])]
 model_chems=set(m.dtxsid.astype(str)); valid=e[e.DTXSID.astype(str).isin(model_chems)&e.mapping_status.isin(['exact_norm_match','ctx_unique'])&e.exposure_domain.eq('specific_chemical')].copy()
 states=sorted(valid.state_code.dropna().unique()); nstates=len(states)
 g=valid.groupby('DTXSID',as_index=False).agg(n_states=('state_code','nunique'),sum_state_weight=('pre_hc5_exposure_weight','sum'),max_state_weight=('pre_hc5_exposure_weight','max'),mean_detection_frequency=('detection_frequency','mean'),total_sites=('n_sites','sum'),total_records=('n_result_records','sum'))
 g['mean_weight_all_states']=g.sum_state_weight/max(nstates,1);g['state_prevalence']=g.n_states/max(nstates,1)
 # all three components are already 0-1-ish; normalize robustly to [0,1]
 for x in ['mean_weight_all_states','state_prevalence','max_state_weight']:
  mx=g[x].max();g[x+'_scaled']=g[x]/mx if mx>0 else 0
 scenarios=[]
 components=['mean_weight_all_states_scaled','state_prevalence_scaled','max_state_weight_scaled']
 perms=list(itertools.permutations([.5,.3,.2]))
 for w in perms:
  name='w'+''.join(str(int(x*10)) for x in w)
  raw=sum(wi*g[ci] for wi,ci in zip(w,components));norm=raw/raw.sum()
  g[name+'_raw']=raw;g[name]=norm; scenarios.append({'scenario':name,'mean_weight':w[0],'prevalence_weight':w[1],'hotspot_weight':w[2]})
 g['national_priority_raw']=g['w532_raw'];g['national_weight']=g['w532'];g=g.sort_values('national_weight',ascending=False).reset_index(drop=True);g['cumulative_weight']=g.national_weight.cumsum();g['priority_tier']=np.where(g.cumulative_weight<=.80,'national_core_80pct','national_surveillance_tail')
 names=c[['DTXSID','PREFERRED_NAME','primary_moa','chemical_form_class']].drop_duplicates('DTXSID');g=g.merge(names,on='DTXSID',how='left');g.to_csv(out/'national_priority_chemicals.csv',index=False)
 pd.DataFrame(scenarios).to_csv(out/'national_weight_scenarios.csv',index=False)
 # rank sensitivity summary
 rs=[]
 base_rank=g.set_index('DTXSID')['national_weight'].rank(ascending=False)
 for sc in [x['scenario'] for x in scenarios]:
  r=g.set_index('DTXSID')[sc].rank(ascending=False);rho=base_rank.corr(r,method='spearman');top20=len(set(base_rank.nsmallest(20).index)&set(r.nsmallest(20).index));top50=len(set(base_rank.nsmallest(50).index)&set(r.nsmallest(50).index));rs.append({'scenario':sc,'spearman_vs_base':rho,'top20_overlap':top20,'top50_overlap':top50})
 pd.DataFrame(rs).to_csv(out/'national_weight_sensitivity.csv',index=False)
 # all-chemicals baseline is all eligible protective chemicals, equal per chemical.
 allc=pd.DataFrame({'DTXSID':sorted(model_chems)});allc=allc.merge(names,on='DTXSID',how='left');allc['all_chemical_weight']=1/len(allc);allc.to_csv(out/'all_protective_chemicals_uniform.csv',index=False)
 # State weights normalized within state for state panels.
 valid['state_weight']=valid.groupby('state_code').pre_hc5_exposure_weight.transform(lambda x:x/x.sum() if x.sum()>0 else 1/len(x));valid.to_csv(out/'state_priority_chemical_weights.csv.gz',index=False,compression='gzip')
 man={'priority_chemicals':len(g),'priority_core_80pct':int((g.priority_tier=='national_core_80pct').sum()),'all_protective_chemicals':len(allc),'states_with_eligible_priority_chemicals':nstates,'base_formula':'0.50 national mean state weight + 0.30 state prevalence + 0.20 maximum state hotspot; then normalize across chemicals','sensitivity_scenarios':len(scenarios),'mapping_filter':['exact_norm_match','ctx_unique']}
 (out/'national_weights_manifest.json').write_text(json.dumps(man,indent=2,ensure_ascii=False));print(json.dumps(man,indent=2,ensure_ascii=False))
if __name__=='__main__':main()
