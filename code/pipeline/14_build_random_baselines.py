#!/usr/bin/env python3
from __future__ import annotations
import argparse,json,math
from pathlib import Path
import numpy as np
import pandas as pd
from numpy.polynomial.hermite import hermgauss
from scipy.stats import norm

KS=[1,3,5,10,20]

def copula_union_sequence(p,rho,nodes=12):
    p=np.clip(np.asarray(p,float),1e-7,1-1e-7)
    if p.ndim==1:p=p[None,:]
    x,w=hermgauss(nodes);z=np.sqrt(2)*x;w=w/np.sqrt(np.pi);thr=norm.ppf(1-p);sr=math.sqrt(max(rho,0));sd=math.sqrt(max(1-rho,1e-8));no=np.zeros_like(p)
    for zz,ww in zip(z,w):no += ww*np.cumprod(norm.cdf((thr-sr*zz)/sd),axis=0)
    return np.clip(1-no,0,1)

def support_matched_sequence(rng,tiers,target_idx,kmax=20):
    selected=[]
    for pos in range(kmax):
        j=target_idx[pos] if pos<len(target_idx) else target_idx[-1]
        pool=np.flatnonzero(tiers==tiers[j]);pool=np.setdiff1d(pool,np.asarray(selected,int),assume_unique=False)
        if len(pool)==0:pool=np.setdiff1d(np.arange(len(tiers)),np.asarray(selected,int),assume_unique=False)
        selected.append(int(rng.choice(pool)))
    return np.asarray(selected,int)

def bootstrap_intervals(values,qmean,w,rng,nboot):
    n=len(values);m=len(w);panel=np.empty(nboot);chem=np.empty(nboot)
    for b in range(nboot):
        panel[b]=np.mean(rng.choice(values,n,replace=True))
        counts=rng.multinomial(m,w);chem[b]=(counts/m)@qmean
    return np.quantile(panel,[.025,.975]),np.quantile(chem,[.025,.975])

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',required=True);ap.add_argument('--replicates',type=int,default=300);ap.add_argument('--bootstrap',type=int,default=500);ap.add_argument('--seed',type=int,default=20260622);ap.add_argument('--targets',default='0.95');ap.add_argument('--universes',default='priority,all_chemicals');a=ap.parse_args()
    root=Path(a.root);out=root/'results/panels';out.mkdir(parents=True,exist_ok=True);manifest=json.loads((out/'panel_probability_manifest.json').read_text(encoding='utf-8'))
    cand=pd.read_csv(root/'results/probability/candidate_universe_locked.csv');cand['tier']=pd.cut(cand.n_contexts,bins=[0,4,19,49,np.inf],labels=False,include_lowest=True).astype(int)
    pri=pd.read_csv(root/'results/weights/national_priority_chemicals.csv');pri['DTXSID']=pri.DTXSID.astype(str);seq=pd.read_csv(out/'national_panel_sequences.csv');rng=np.random.default_rng(a.seed);rows=[]
    curves=pd.read_csv(out/'national_k1_20_coverage_curves.csv')
    for x in [float(v) for v in a.targets.split(',') if v.strip()]:
        xx=int(round(x*100));z=np.load(root/f'results/probability/species_chemical_tail_probability_x{xx}.npz',allow_pickle=True);p=z['p'].astype(np.float32);species=z['species'].astype(str);chems=z['chemicals'].astype(str);ci={c:i for i,c in enumerate(chems)}
        dep=manifest.get('dependence_audit_by_x',{}).get(str(x));rho=float(dep['rho_working']) if dep else float(curves[curves.protection_target_x.eq(x)].rho.iloc[0])
        c2=cand.set_index('latin_name').reindex(species).reset_index();tiers=c2.tier.to_numpy(int);sidx={s:i for i,s in enumerate(species)}
        for universe in [v.strip() for v in a.universes.split(',') if v.strip()]:
            if universe=='priority':
                ws=pri[pri.DTXSID.isin(ci)].set_index('DTXSID').national_weight.astype(float);ws/=ws.sum()
            else:ws=pd.Series(1/len(chems),index=chems)
            names=ws.index.tolist();idx=np.array([ci[c] for c in names]);w=ws.to_numpy(float);pu=p[:,idx]
            data_names=seq[(seq.universe==universe)&seq.protection_target_x.eq(x)&seq.method.eq('data_driven')].sort_values('rank').latin_name.tolist();data_idx=np.array([sidx[s] for s in data_names if s in sidx],int)
            methods=['random_all_species_probability','random_support_matched_all_species']
            for method in methods:
                vals={k:np.empty(a.replicates) for k in KS};qsum={k:np.zeros(len(idx),float) for k in KS}
                for r in range(a.replicates):
                    selected=rng.choice(p.shape[0],20,replace=False) if method=='random_all_species_probability' else support_matched_sequence(rng,tiers,data_idx,20)
                    qs=copula_union_sequence(pu[selected],rho)
                    for k in KS:
                        q=qs[k-1];vals[k][r]=w@q;qsum[k]+=q
                for k in KS:
                    qmean=qsum[k]/a.replicates;panel_ci,chem_ci=bootstrap_intervals(vals[k],qmean,w,rng,a.bootstrap)
                    rows.append({'universe':universe,'protection_target_x':x,'method':method,'k':k,'mean_expected_weighted_coverage':float(vals[k].mean()),'sd_between_random_panels':float(vals[k].std(ddof=1)),'panel_composition_q025':float(np.quantile(vals[k],.025)),'panel_composition_q975':float(np.quantile(vals[k],.975)),'bootstrap_ci_mean_q025':float(panel_ci[0]),'bootstrap_ci_mean_q975':float(panel_ci[1]),'chemical_bootstrap_q025':float(chem_ci[0]),'chemical_bootstrap_q975':float(chem_ci[1]),'n_random_panels':a.replicates,'n_bootstrap':a.bootstrap,'candidate_pool_n':p.shape[0],'n_chemicals':len(idx),'copula_rho':rho,'seed':a.seed,'interpretation':'panel-composition interval is the 2.5-97.5 percentile across randomly drawn species sets; bootstrap intervals describe uncertainty in the mean baseline, not full cell-level posterior uncertainty'})
    df=pd.DataFrame(rows);df.to_csv(out/'random_baselines.csv',index=False);print(df.to_string(index=False))
if __name__=='__main__':main()
