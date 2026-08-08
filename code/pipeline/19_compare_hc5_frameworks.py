#!/usr/bin/env python3
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import norm, spearmanr, pearsonr

PROTECTIVE=['mortality_survival','immobilization_intoxication','growth','reproduction','development_morphology']

def hc5(v):
    a=np.asarray(v,float);a=a[np.isfinite(a)]
    if len(a)<3:return np.nan
    sd=np.std(a,ddof=1)
    return float(np.mean(a)+norm.ppf(.05)*sd) if np.isfinite(sd) and sd>.05 else np.nan

def flags(g):
    ph=g.phylum_division.fillna('').str.lower();cl=g['class'].fillna('').str.lower();order=g.tax_order.fillna('').str.lower();fam=g.family.fillna('').str.lower();king=g.kingdom.fillna('').str.lower()
    fish=ph.eq('chordata')&cl.isin(['actinopterygii','actinopteri','teleostei','osteichthyes'])
    producer=king.isin(['plantae','chromista'])|cl.isin(['chlorophyceae','bacillariophyceae','cyanophyceae','trebouxiophyceae','zygnematophyceae'])
    return {'fish':fish.to_numpy(),'salmonid':(fish&fam.eq('salmonidae')).to_numpy(),'non_salmonid_fish':(fish&~fam.eq('salmonidae')).to_numpy(),'chordate':ph.eq('chordata').to_numpy(),'invert':(king.eq('animalia')&~ph.eq('chordata')).to_numpy(),'producer':producer.to_numpy(),'plank_crust':(cl.isin(['branchiopoda','maxillopoda','copepoda'])|order.isin(['cladocera','diplostraca','calanoida','cyclopoida'])).to_numpy(),'benthic_crust':(cl.eq('malacostraca')|order.isin(['amphipoda','isopoda','decapoda'])).to_numpy(),'insect':cl.eq('insecta').to_numpy(),'other_phylum':(~ph.isin(['arthropoda','chordata',''])).to_numpy(),'family':fam.to_numpy(),'reg_group':np.select([producer,fish,cl.eq('amphibia'),cl.isin(['branchiopoda','maxillopoda','copepoda']),cl.eq('malacostraca'),cl.eq('insecta'),ph.eq('mollusca')],['producer','fish','amphibian','planktonic_crustacean','benthic_crustacean','insect','mollusk'],default='other')}

def choose_slots(g, masks, score, distinct_family=True):
    fl=flags(g);used=set();families=set();sel=[]
    # Most constrained slots first.
    order=sorted(range(len(masks)),key=lambda i:int(np.sum(masks[i])))
    for slot in order:
        ix=np.flatnonzero(masks[slot]);ix=[int(i) for i in ix if int(i) not in used and (not distinct_family or fl['family'][i] not in families)]
        if not ix:return None
        j=max(ix,key=lambda z:(score[z],g.iloc[z].latin_name))
        used.add(j);families.add(fl['family'][j]);sel.append((slot,j))
    return [j for _,j in sorted(sel)]

def diverse_select(g,k,min_groups,score):
    fl=flags(g);groups=fl['reg_group'];sel=[];seen=set()
    # First obtain group breadth with the best-scoring representative of each group.
    for _ in range(min(min_groups,k)):
        cand=[i for i in range(len(g)) if i not in sel and groups[i] not in seen]
        if not cand:return None
        j=max(cand,key=lambda z:(score[z],g.iloc[z].latin_name));sel.append(j);seen.add(groups[j])
    while len(sel)<k:
        cand=[i for i in range(len(g)) if i not in sel]
        if not cand:return None
        j=max(cand,key=lambda z:(score[z],g.iloc[z].latin_name));sel.append(j)
    return sel

def taxonomy_select(g,k):
    rem=list(range(len(g)));sel=[];seen={x:set() for x in ['phylum_division','class','tax_order','family']}
    strict_class_slots=min(k,g['class'].nunique(dropna=True))
    for rank in range(min(k,len(g))):
        vals=[]
        for j in rem:
            r=g.iloc[j];nov=4*(r.phylum_division not in seen['phylum_division'])+3*(r['class'] not in seen['class'])+2*(r.tax_order not in seen['tax_order'])+(r.family not in seen['family'])
            if rank<strict_class_slots and r['class'] in seen['class']:continue
            vals.append((nov,float(r.oof_probability),r.latin_name,j))
        if not vals:
            for j in rem:
                r=g.iloc[j];nov=4*(r.phylum_division not in seen['phylum_division'])+3*(r['class'] not in seen['class'])+2*(r.tax_order not in seen['tax_order'])+(r.family not in seen['family'])
                vals.append((nov,float(r.oof_probability),r.latin_name,j))
        *_,j=max(vals);sel.append(j);rem.remove(j);r=g.iloc[j]
        for c in seen:seen[c].add(r[c])
    return sel

def summary_metrics(d):
    x=d.estimated_hc5_log10.to_numpy(float);y=d.reference_hc5_log10.to_numpy(float);delta=x-y
    return {'n_contexts':int(d.context_id.nunique()),'n_estimates':len(d),'n_selected':int(d.n_selected.iloc[0]) if d.n_selected.nunique()==1 else np.nan,'median_ratio':float(np.median(10**delta)),'ratio_q025':float(np.quantile(10**delta,.025)),'ratio_q975':float(np.quantile(10**delta,.975)),'median_abs_log10_error':float(np.median(np.abs(delta))),'rmse_log10':float(np.sqrt(np.mean(delta**2))),'spearman_r':float(spearmanr(x,y).statistic),'pearson_r':float(pearsonr(x,y).statistic),'within_factor2':float(np.mean(np.abs(delta)<=np.log10(2))),'false_safe_factor2_rate':float(np.mean(delta>np.log10(2))),'overconservative_factor2_rate':float(np.mean(delta<-np.log10(2)))}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',required=True);ap.add_argument('--random-replicates',type=int,default=10);ap.add_argument('--bootstrap',type=int,default=500);ap.add_argument('--seed',type=int,default=20260622);a=ap.parse_args();root=Path(a.root);out=root/'results/hc5_framework_comparison';out.mkdir(parents=True,exist_ok=True);rng=np.random.default_rng(a.seed)
    m=pd.read_csv(root/'results/toxicity/censored_model_cells.csv.gz',low_memory=False);valid=m.mle_status.astype(str).str.startswith('identified')&np.isfinite(m.mu_log10_umol_L);m=m[valid&m.effect_family.isin(PROTECTIVE)].copy();m['context_id']=m[['dtxsid','effect_family','endpoint_band_v16','duration_window_v16','medium_family']].astype(str).agg('|'.join,axis=1)
    pred=pd.read_csv(root/'results/validation/moa_simplified_oof_predictions.csv.gz',low_memory=False);pred=pred[pred.protection_target_x.eq(.95)][['context_id','latin_name','taxonomy_oof']].groupby(['context_id','latin_name'],as_index=False).mean().rename(columns={'taxonomy_oof':'oof_probability'})
    m=m.merge(pred,on=['context_id','latin_name'],how='left');m['oof_probability']=m.oof_probability.fillna(.05)
    cs=m.groupby('context_id').agg(n_species=('latin_name','nunique'),dtxsid=('dtxsid','first'),effect_family=('effect_family','first')).reset_index();cs=cs[cs.n_species>=15]
    rows=[]
    for num,r in enumerate(cs.itertuples(index=False),1):
        g=m[m.context_id.eq(r.context_id)].sort_values('latin_name').drop_duplicates('latin_name').reset_index(drop=True);ref=hc5(g.mu_log10_umol_L)
        if not np.isfinite(ref):continue
        score=g.oof_probability.to_numpy(float);fl=flags(g)
        methods={
          'probability_sentinel_k5':g.sort_values(['oof_probability','latin_name'],ascending=[False,True]).head(5).index.tolist(),
          'taxonomy_diversity_k5':taxonomy_select(g,5),
          'EPA_1985_8_slots':choose_slots(g,[fl['salmonid'],fl['non_salmonid_fish'],fl['chordate'],fl['plank_crust'],fl['benthic_crust'],fl['insect'],fl['other_phylum'],np.ones(len(g),bool)],score),
          'CCME_Type_A_7':choose_slots(g,[fl['salmonid'],fl['non_salmonid_fish'],fl['fish'],fl['plank_crust'],fl['invert'],fl['invert'],fl['producer']],score),
          'ANZG_minimum_6_4groups':diverse_select(g,6,4,score),
          'EU_WFD_10_8groups':diverse_select(g,10,8,score),
          'ANZG_preferred_15_4groups':diverse_select(g,15,4,score),
        }
        for name,sel in methods.items():
            if sel is None:continue
            est=hc5(g.iloc[sel].mu_log10_umol_L)
            if np.isfinite(est):rows.append({'context_id':r.context_id,'dtxsid':r.dtxsid,'effect_family':r.effect_family,'method':name,'replicate':0,'n_reference_species':len(g),'n_selected':len(sel),'reference_hc5_log10':ref,'estimated_hc5_log10':est,'selected_species':' | '.join(g.iloc[sel].latin_name.astype(str))})
        # Secondary random k=5 sensitivity only; not the main regulatory comparison.
        for rep in range(a.random_replicates):
            sel=rng.choice(len(g),5,replace=False);est=hc5(g.iloc[sel].mu_log10_umol_L)
            if np.isfinite(est):rows.append({'context_id':r.context_id,'dtxsid':r.dtxsid,'effect_family':r.effect_family,'method':'random_k5_secondary','replicate':rep,'n_reference_species':len(g),'n_selected':5,'reference_hc5_log10':ref,'estimated_hc5_log10':est,'selected_species':' | '.join(g.iloc[sel].latin_name.astype(str))})
    d=pd.DataFrame(rows);d['log10_ratio']=d.estimated_hc5_log10-d.reference_hc5_log10;d['ratio_est_to_reference']=10**d.log10_ratio;d.to_csv(out/'hc5_framework_context_estimates.csv.gz',index=False,compression='gzip')
    summaries=[]
    for method,g in d.groupby('method'):summaries.append({'method':method,**summary_metrics(g)})
    summ=pd.DataFrame(summaries).sort_values('n_selected');summ.to_csv(out/'hc5_framework_summary.csv',index=False)
    # Context bootstrap confidence intervals for summary statistics.
    boots=[]
    for method,g in d.groupby('method'):
        grouped=[]
        for _,gg in g.groupby('context_id',sort=False):
            grouped.append((gg.estimated_hc5_log10.to_numpy(float),gg.reference_hc5_log10.to_numpy(float),int(gg.n_selected.iloc[0])))
        ng=len(grouped)
        for b in range(a.bootstrap):
            ids=rng.integers(0,ng,size=ng)
            x=np.concatenate([grouped[i][0] for i in ids]);y=np.concatenate([grouped[i][1] for i in ids]);delta=x-y
            mm={'n_contexts':ng,'n_estimates':len(x),'n_selected':grouped[0][2] if all(z[2]==grouped[0][2] for z in grouped) else np.nan,'median_ratio':float(np.median(10**delta)),'ratio_q025':float(np.quantile(10**delta,.025)),'ratio_q975':float(np.quantile(10**delta,.975)),'median_abs_log10_error':float(np.median(np.abs(delta))),'rmse_log10':float(np.sqrt(np.mean(delta**2))),'spearman_r':float(spearmanr(x,y).statistic),'pearson_r':float(pearsonr(x,y).statistic),'within_factor2':float(np.mean(np.abs(delta)<=np.log10(2))),'false_safe_factor2_rate':float(np.mean(delta>np.log10(2))),'overconservative_factor2_rate':float(np.mean(delta<-np.log10(2)))}
            boots.append({'method':method,'bootstrap':b,**mm})
    bd=pd.DataFrame(boots);bd.to_csv(out/'hc5_framework_bootstrap.csv.gz',index=False,compression='gzip')
    ci=[]
    for method,g in bd.groupby('method'):
        rec={'method':method}
        for col in ['median_ratio','median_abs_log10_error','spearman_r','within_factor2','false_safe_factor2_rate','overconservative_factor2_rate']:
            rec[col+'_ci025']=float(g[col].quantile(.025));rec[col+'_ci975']=float(g[col].quantile(.975))
        ci.append(rec)
    summ.merge(pd.DataFrame(ci),on='method',how='left').to_csv(out/'hc5_framework_summary_with_ci.csv',index=False)
    man={'version':'v16.5','reference':'Lognormal HC5 fitted from all measured species in each context with at least 15 species.','primary_comparison':'Deterministic context-specific panels: probability sentinel uses 5 measured species with highest whole-chemical OOF measured-tail probabilities; regulatory frameworks use their documented minimum species/group counts and choose the highest-probability feasible representatives. All selected panels estimate HC5 from their measured toxicity values.','panel_sizes':{'probability_sentinel':5,'taxonomy_diversity':5,'EPA_1985':8,'CCME_Type_A':7,'ANZG_minimum':6,'EU_WFD':10,'ANZG_preferred':15},'secondary_random_simulation':f'{a.random_replicates} random k=5 sets per context plus {a.bootstrap} context bootstraps. The earlier 50-replicate constrained sampler attempt was not used because of performance bottlenecks; optimize locally before publication.','caution':'A five-species lognormal HC5 is an exploratory screening estimate with wide uncertainty, not a regulatory SSD. The comparison tests whether sensitive probability-guided selection can approximate or conservatively track the all-species reference using fewer species.'}
    (out/'hc5_framework_manifest.json').write_text(json.dumps(man,indent=2,ensure_ascii=False),encoding='utf-8')
    print(summ.to_string(index=False));print(json.dumps(man,indent=2,ensure_ascii=False))
if __name__=='__main__':main()
