#!/usr/bin/env python3
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score, average_precision_score
from sklearn.model_selection import GroupKFold

PROTECTIVE=['mortality_survival','immobilization_intoxication','growth','reproduction','development_morphology']
TARGETS=[.80,.90,.95]
TAX_LEVELS=[('latin_name',4.0,.80),('genus',2.0,.50),('family',1.0,.30),('class',.5,.15)]

def calibration(raw,y,groups):
    raw=np.clip(np.asarray(raw,float),1e-6,1-1e-6);X=np.log(raw/(1-raw)).reshape(-1,1);out=np.full(len(y),np.nan)
    n=min(5,len(np.unique(groups)))
    if n<2:return raw
    for tr,te in GroupKFold(n_splits=n).split(X,y,groups):
        fit=LogisticRegression(C=1.0,solver='liblinear',max_iter=300);fit.fit(X[tr],y[tr]);out[te]=fit.predict_proba(X[te])[:,1]
    return out

def loco_probability(m,q0,group_col=None):
    n=len(m);vsum=np.zeros(n);wsum=np.zeros(n)
    for level,bw,cap in TAX_LEVELS:
        keys=[level,'effect_family'];ck=['dtxsid',level,'effect_family']
        if group_col:keys.insert(1,group_col);ck.insert(2,group_col)
        total=m.groupby(keys,dropna=False).p_context.agg(total_sum='sum',total_count='count').reset_index();byc=m.groupby(ck,dropna=False).p_context.agg(chemical_sum='sum',chemical_count='count').reset_index()
        fr=m[ck].copy();fr['_id']=np.arange(n);fr=fr.merge(total,on=keys,how='left').merge(byc,on=ck,how='left').sort_values('_id')
        sm=fr.total_sum.fillna(0).to_numpy(float)-fr.chemical_sum.fillna(0).to_numpy(float);nn=fr.total_count.fillna(0).to_numpy(int)-fr.chemical_count.fillna(0).to_numpy(int)
        ok=nn>0;val=np.full(n,q0);val[ok]=(sm[ok]+10*q0)/(nn[ok]+10);wt=np.zeros(n);wt[ok]=bw*np.minimum(nn[ok],20);vsum+=val*wt;wsum+=wt
    out=np.full(n,q0);ok=wsum>0;out[ok]=vsum[ok]/wsum[ok];return out

def metric(y,p):
    p=np.clip(p,1e-6,1-1e-6);return {'brier':float(brier_score_loss(y,p)),'log_loss':float(log_loss(y,p,labels=[0,1])),'roc_auc':float(roc_auc_score(y,p)) if len(np.unique(y))==2 else np.nan,'average_precision':float(average_precision_score(y,p)) if y.sum()>0 else np.nan}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',required=True);a=ap.parse_args();root=Path(a.root);out=root/'results/validation';out.mkdir(parents=True,exist_ok=True)
    model=pd.read_csv(root/'results/toxicity/censored_model_cells.csv.gz',low_memory=False);chem=pd.read_csv(root/'results/chemistry/chemical_moa_hierarchy.csv.gz',low_memory=False);chem['DTXSID']=chem.DTXSID.astype(str)
    cand=set(pd.read_csv(root/'results/probability/candidate_universe_locked.csv').latin_name.astype(str))
    valid=model.mle_status.astype(str).str.startswith('identified')&np.isfinite(model.mu_log10_umol_L);b=model[valid&model.effect_family.isin(PROTECTIVE)&model.latin_name.isin(cand)].copy();b['dtxsid']=b.dtxsid.astype(str);b['context_id']=b[['dtxsid','effect_family','endpoint_band_v16','duration_window_v16','medium_family']].astype(str).agg('|'.join,axis=1)
    cs=b.groupby('context_id').agg(n_species=('latin_name','nunique'),mean=('mu_log10_umol_L','mean'),sd=('mu_log10_umol_L','std')).reset_index();cs=cs[(cs.n_species>=5)&cs.sd.notna()&(cs.sd>.05)];b=b[b.context_id.isin(cs.context_id)].merge(cs,on='context_id').merge(chem[['DTXSID','moa_level_functional','moa_level_form']],left_on='dtxsid',right_on='DTXSID',how='left');b['functional_moa']=b.moa_level_functional.fillna('MOAFAM::unresolved');b['chemical_form']=b.moa_level_form.fillna('FORM::unresolved')
    predictions=[];metrics=[];gate=[]
    for x in TARGETS:
        q0=1-x;m=b.copy();m['hc']=m['mean']+norm.ppf(q0)*m.sd;m['p_context']=np.clip(norm.cdf((m.hc-m.mu_log10_umol_L)/np.sqrt(m.se_mu_log10.fillna(.35)**2+(m.sd/np.sqrt(m.n_species))**2).clip(lower=.12)),1e-6,1-1e-6);m['tail_binary']=(m.mu_log10_umol_L<=m.hc).astype(int);y=m.tail_binary.to_numpy(int);groups=m.dtxsid.to_numpy(str)
        probs={'taxonomy':calibration(loco_probability(m,q0,None),y,groups),'functional_moa':calibration(loco_probability(m,q0,'functional_moa'),y,groups),'chemical_form':calibration(loco_probability(m,q0,'chemical_form'),y,groups)}
        pf=m[['dtxsid','latin_name','effect_family','context_id','tail_binary','functional_moa','chemical_form']].copy();pf['protection_target_x']=x
        for k,v in probs.items():pf[k+'_oof']=v
        predictions.append(pf)
        for eff,g in m.groupby('effect_family'):
            idx=g.index.to_numpy();base=metric(y[idx],probs['taxonomy'][idx]);support={'n_rows':len(idx),'n_positive':int(y[idx].sum()),'n_chemicals':int(g.dtxsid.nunique())}
            metrics.append({'protection_target_x':x,'effect_family':eff,'layer':'taxonomy',**support,**base})
            for layer in ['functional_moa','chemical_form']:
                mm=metric(y[idx],probs[layer][idx]);metrics.append({'protection_target_x':x,'effect_family':eff,'layer':layer,**support,**mm})
    met=pd.DataFrame(metrics);met.to_csv(out/'moa_simplified_oof_metrics.csv',index=False);pd.concat(predictions,ignore_index=True).to_csv(out/'moa_simplified_oof_predictions.csv.gz',index=False,compression='gzip')
    for (x,eff),g in met.groupby(['protection_target_x','effect_family']):
        tax=g[g.layer.eq('taxonomy')].iloc[0]
        for layer in ['functional_moa','chemical_form']:
            r=g[g.layer.eq(layer)].iloc[0];imp=tax.brier-r.brier;rel=imp/tax.brier if tax.brier>0 else np.nan
            adjacent=met[(met.effect_family.eq(eff))&met.protection_target_x.isin([.90,.95])]
            diffs=[]
            for xx in [.90,.95]:
                z=adjacent[adjacent.protection_target_x.eq(xx)];t=z[z.layer.eq('taxonomy')];c=z[z.layer.eq(layer)];
                if len(t)&len(c):diffs.append(float(t.iloc[0].brier-c.iloc[0].brier))
            nonworse=all(z>=-1e-5 for z in diffs);enough=r.n_rows>=500 and r.n_positive>=20 and r.n_chemicals>=30;material=imp>=1e-4 and rel>=.002;secondary=(r.log_loss<=tax.log_loss+1e-4 and (np.isnan(r.roc_auc) or np.isnan(tax.roc_auc) or r.roc_auc>=tax.roc_auc-.01));core=bool(enough and material and nonworse and secondary)
            gate.append({'protection_target_x':x,'effect_family':eff,'candidate_layer':layer,'n_rows':int(r.n_rows),'n_positive':int(r.n_positive),'n_chemicals':int(r.n_chemicals),'taxonomy_brier':tax.brier,'candidate_brier':r.brier,'absolute_brier_improvement':imp,'relative_brier_improvement':rel,'taxonomy_log_loss':tax.log_loss,'candidate_log_loss':r.log_loss,'taxonomy_roc_auc':tax.roc_auc,'candidate_roc_auc':r.roc_auc,'adjacent_hc10_hc5_nonworsening':nonworse,'minimum_support_pass':enough,'material_improvement_pass':material,'secondary_metrics_pass':secondary,'core_matrix_eligible':core})
    gate=pd.DataFrame(gate);gate.to_csv(out/'moa_core_entry_gate.csv',index=False)
    decisions=[]
    for (x,eff),g in gate.groupby(['protection_target_x','effect_family']):
        ok=g[g.core_matrix_eligible]
        if len(ok):best=ok.sort_values('candidate_brier').iloc[0];rec='taxonomy_plus_'+best.candidate_layer;status='core_eligible'
        else:
            pos=g[g.absolute_brier_improvement>0];best=(pos if len(pos) else g).sort_values('candidate_brier').iloc[0];rec='taxonomy_only';status='provisional_not_core' if best.absolute_brier_improvement>0 else 'taxonomy_fallback'
        decisions.append({'protection_target_x':x,'effect_family':eff,'recommended_core_layer':rec,'best_exploratory_layer':best.candidate_layer,'status':status,'best_absolute_brier_improvement':best.absolute_brier_improvement,'n_rows':int(best.n_rows),'n_positive':int(best.n_positive),'n_chemicals':int(best.n_chemicals)})
    dec=pd.DataFrame(decisions);dec.to_csv(out/'moa_simplified_core_decision.csv',index=False)
    man={'version':'v16.5','core_candidate_layers':['functional MOA family','chemical-form class'],'taxonomy_direct_evidence_always_present':True,'annotation_only':['dominant MIE','MIE signature','AOP route','AOP chain'],'requirements':{'whole_chemical_GroupKFold_OOF':True,'n_rows_min':500,'positive_events_min':20,'chemicals_min':30,'absolute_Brier_improvement_min':.0001,'relative_Brier_improvement_min':.002,'HC10_HC5_nonworsening':True,'log_loss_and_AUC_nonmaterial_worsening':True},'rule':'Only validated effect-specific layers may modify the core matrix. Otherwise MOA remains interpretation and testing-priority evidence.'}
    (out/'moa_simplified_manifest.json').write_text(json.dumps(man,indent=2,ensure_ascii=False),encoding='utf-8');print(dec.to_string(index=False));print(json.dumps(man,indent=2,ensure_ascii=False))
if __name__=='__main__':main()
