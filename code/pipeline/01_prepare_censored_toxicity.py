#!/usr/bin/env python3
from __future__ import annotations
import argparse, glob, json, math
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import ndtr, log_ndtr

PROTECTIVE={'mortality_survival','immobilization_intoxication','growth','reproduction','development_morphology'}
LIFE_GROUPS={
'unknown_missing':{'NR'},
'mixed_or_unspecified':{'--','NC','MX','NOINT'},
'egg_embryo':{'BL','CS','EG','EM','EY','GA','ML','MN','NL','PC','PY','ZY','GE'},
'larva_juvenile':{'AL','CO','CP','EL','FI','FT','FY','GL','IM','IN','IT','JV','LP','LR','LV','ME','MY','NB','ND','NE','NH','NU','NY','PA','PHT','PJ','PK','PL','PN','PQ','PS','PT','PU','PV','PZ','SA','SC','SF','SI','SMT','SU','TA','UY','VE','WN','YA','YE','YO','YY','ZO','PPU'},
'adult_mature':{'AD','FB','IG','MA','MD','PB','PH','RP','SG','SM','VG','VI'},
'reproductive_gamete':{'F0','F1','F11','F2','F3','F6','F7','FG','GM','LE','MG','OO','PG','PO','PP','PW','SP','SW','LC'},
'plant_seed_spore_germination':{'BD','BS','BT','CC','CM','CY','FO','GPS','GS','HD','IB','IE','JN','PE','PRB','RC','RH','RST','SB','SCS','SD','SE','SL','SN','SO','SR','ST','TLS','TN','TU','ZS'},
'microbial_or_cell_growth_phase':{'EX','GP','SY','TC','TZ'},
'other_defined_stage':{'MO','PD','PI'}
}
LIFE_MAP={c:g for g,codes in LIFE_GROUPS.items() for c in codes}
LOG2PI=math.log(2*math.pi)

def phi(z): return np.exp(-0.5*z*z-0.5*LOG2PI)

def censored_normal_em(types,point,lower,upper,sigma,max_iter=100,tol=1e-8):
    exact=np.isin(types,['exact','approximate_exact'])&np.isfinite(point)
    left=(types=='left_censored')&np.isfinite(upper)
    right=(types=='right_censored')&np.isfinite(lower)
    inter=(types=='interval_censored')&np.isfinite(lower)&np.isfinite(upper)
    if exact.sum()==0 and inter.sum()==0:
        if left.sum()>0 and right.sum()==0: return np.nan,'upper_bound_only',np.nan,float(np.nanmin(upper[left])),0
        if right.sum()>0 and left.sum()==0: return np.nan,'lower_bound_only',np.nan,float(np.nanmax(lower[right])),0
    init=[]
    if exact.any(): init.extend(point[exact].tolist())
    if inter.any(): init.extend(((lower[inter]+upper[inter])/2).tolist())
    if left.any(): init.extend((upper[left]-0.5*sigma).tolist())
    if right.any(): init.extend((lower[right]+0.5*sigma).tolist())
    if not init: return np.nan,'no_numeric_support',np.nan,np.nan,0
    mu=float(np.median(init))
    n=int(exact.sum()+left.sum()+right.sum()+inter.sum())
    for it in range(max_iter):
        vals=[]
        if exact.any(): vals.append(point[exact])
        if left.any():
            z=(upper[left]-mu)/sigma
            ratio=np.exp(-0.5*z*z-0.5*LOG2PI-log_ndtr(z))
            vals.append(mu-sigma*ratio)
        if right.any():
            z=(lower[right]-mu)/sigma
            ratio=np.exp(-0.5*z*z-0.5*LOG2PI-log_ndtr(-z))
            vals.append(mu+sigma*ratio)
        if inter.any():
            a=(lower[inter]-mu)/sigma; b=(upper[inter]-mu)/sigma
            den=np.maximum(ndtr(b)-ndtr(a),1e-300)
            vals.append(mu+sigma*(phi(a)-phi(b))/den)
        new=float(np.mean(np.concatenate(vals)))
        if abs(new-mu)<tol: mu=new; break
        mu=new
    info_n=exact.sum()+inter.sum()+0.5*(left.sum()+right.sum())
    se=float(sigma/math.sqrt(max(info_n,0.25)))
    return mu,'identified_censored_mle',se,np.nan,it+1

def aggregate(df,keys,sigma_lookup,stratum_cols,study_col='reference_number'):
    """All groups simultaneously: fixed-sigma censored-normal EM MLE."""
    d=df.copy()
    d['_stratum']=d[stratum_cols].astype(str).agg('|'.join,axis=1)
    sig=d['_stratum'].map(sigma_lookup).fillna(0.65).to_numpy(float)
    mi=pd.MultiIndex.from_frame(d[keys]); codes,uniques=pd.factorize(mi,sort=False); codes=codes.astype(np.int64); G=len(uniques)
    typ=d.censoring_class.astype(str).to_numpy(); pt=d._log_point.to_numpy(float); lo=d._log_lower.to_numpy(float); hi=d._log_upper.to_numpy(float)
    exact=np.isin(typ,['exact','approximate_exact'])&np.isfinite(pt); left=(typ=='left_censored')&np.isfinite(hi); right=(typ=='right_censored')&np.isfinite(lo); inter=(typ=='interval_censored')&np.isfinite(lo)&np.isfinite(hi); approx=(typ=='approximate_exact')&np.isfinite(pt)
    bc=lambda m:np.bincount(codes,weights=m.astype(float),minlength=G)
    n=bc(np.ones(len(d),bool)); ne=bc(exact); nl=bc(left); nr=bc(right); ni=bc(inter); na=bc(approx)
    sigma_g=np.bincount(codes,weights=sig,minlength=G)/np.maximum(n,1)
    upper_only=(ne==0)&(ni==0)&(nl>0)&(nr==0); lower_only=(ne==0)&(ni==0)&(nr>0)&(nl==0)
    identifiable=(n>0)&~upper_only&~lower_only
    pseudo=np.full(len(d),np.nan)
    pseudo[exact]=pt[exact]; pseudo[inter]=(lo[inter]+hi[inter])/2; pseudo[left]=hi[left]-.5*sig[left]; pseudo[right]=lo[right]+.5*sig[right]
    ok=np.isfinite(pseudo); mu=np.bincount(codes[ok],weights=pseudo[ok],minlength=G)/np.maximum(np.bincount(codes[ok],minlength=G),1)
    mu[~identifiable]=np.nan
    iterations=0
    for it in range(80):
        mrec=mu[codes]; ey=np.full(len(d),np.nan); ey[exact]=pt[exact]
        if left.any():
            z=(hi[left]-mrec[left])/sig[left]; ratio=np.exp(np.clip(-.5*z*z-.5*LOG2PI-log_ndtr(z),-700,700)); ey[left]=mrec[left]-sig[left]*ratio
        if right.any():
            z=(lo[right]-mrec[right])/sig[right]; ratio=np.exp(np.clip(-.5*z*z-.5*LOG2PI-log_ndtr(-z),-700,700)); ey[right]=mrec[right]+sig[right]*ratio
        if inter.any():
            aa=(lo[inter]-mrec[inter])/sig[inter]; bb=(hi[inter]-mrec[inter])/sig[inter]; den=np.maximum(ndtr(bb)-ndtr(aa),1e-300); ey[inter]=mrec[inter]+sig[inter]*(phi(aa)-phi(bb))/den
        good=np.isfinite(ey)&np.isfinite(mrec); new=np.bincount(codes[good],weights=ey[good],minlength=G)/np.maximum(np.bincount(codes[good],minlength=G),1); new[~identifiable]=np.nan
        delta=np.nanmax(np.abs(new-mu)); mu=new; iterations=it+1
        if delta<1e-8: break
    # Bounds for one-sided-only groups.
    ub=np.full(G,np.inf); lb=np.full(G,-np.inf)
    if left.any(): np.minimum.at(ub,codes[left],hi[left])
    if right.any(): np.maximum.at(lb,codes[right],lo[right])
    bound=np.full(G,np.nan); bound[upper_only]=ub[upper_only]; bound[lower_only]=lb[lower_only]
    info=ne+ni+.5*(nl+nr); se=sigma_g/np.sqrt(np.maximum(info,.25)); se[~identifiable]=np.nan
    status=np.full(G,'identified_censored_mle',object); status[(n==ne)&(ne>0)]='identified_exact_only'; status[upper_only]='upper_bound_only'; status[lower_only]='lower_bound_only'; status[~(identifiable|upper_only|lower_only)]='no_numeric_support'
    # Exact-only mean should be direct, avoiding trivial EM drift.
    exact_sum=np.bincount(codes[exact],weights=pt[exact],minlength=G); exact_only=(n==ne)&(ne>0); mu[exact_only]=exact_sum[exact_only]/ne[exact_only]
    # Unique study counts without per-group Python callbacks.
    tmp=pd.DataFrame({'gid':codes,'ref':d[study_col].astype(str).to_numpy()}).drop_duplicates(['gid','ref']); ns=tmp.groupby('gid',sort=False).size().reindex(range(G),fill_value=0).to_numpy()
    base=uniques.to_frame(index=False); base.columns=keys
    base['mu_log10_umol_L']=mu; base['se_mu_log10']=se; base['pooled_sigma_log10']=sigma_g; base['mle_status']=status; base['bound_log10_umol_L']=bound
    base['n_records']=n.astype(int); base['n_exact']=ne.astype(int); base['n_left']=nl.astype(int); base['n_right']=nr.astype(int); base['n_interval']=ni.astype(int); base['n_approx']=na.astype(int); base['n_studies']=ns.astype(int); base['em_iterations']=iterations
    return base

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--parts',required=True); ap.add_argument('--life-codes',required=True); ap.add_argument('--out',required=True); ap.add_argument('--cache',default=''); a=ap.parse_args(); out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    lc=pd.read_csv(a.life_codes,sep='|',dtype=str).fillna(''); official=set(lc.code); miss=sorted(official-set(LIFE_MAP));
    if miss: raise RuntimeError(miss)
    lc['broad_life_stage_v16']=lc.code.map(LIFE_MAP); lc['information_status']=np.select([lc.code.eq('NR'),lc.code.isin({'--','NC','MX','NOINT'})],['unknown_missing','defined_but_unspecified'],default='mapped_defined');lc.to_csv(out/'lifestage_code_mapping_complete.csv',index=False)
    use=['result_id','test_id','reference_number','dtxsid','latin_name','common_name','kingdom','phylum_division','class','tax_order','family','genus','effect','measurement','effect_family','effect_evidence_layer','endpoint','endpoint_family','endpoint_level_band','endpoint_group','exposure_duration_h','organism_lifestage','medium_family','censoring_class','concentration_point_umol_L','concentration_lower_umol_L','concentration_upper_umol_L','waterborne_main','publication_year']
    frames=[];raw=0
    for p in sorted(glob.glob(a.parts)):
        x=pd.read_csv(p,usecols=lambda c:c in use,low_memory=False);raw+=len(x);frames.append(x)
    d=pd.concat(frames,ignore_index=True)
    for c in ['concentration_point_umol_L','concentration_lower_umol_L','concentration_upper_umol_L','exposure_duration_h']:d[c]=pd.to_numeric(d[c],errors='coerce')
    d['dtxsid']=d.dtxsid.astype('string').str.strip().str.upper();d['latin_name']=d.latin_name.astype('string').str.strip();d['waterborne_main']=d.waterborne_main.fillna(False).astype(bool)
    valid=d.dtxsid.str.startswith('DTXSID',na=False)&d.latin_name.notna()&(d.latin_name.str.len()>2)&d.waterborne_main&((d.concentration_point_umol_L>0)|(d.concentration_lower_umol_L>0)|(d.concentration_upper_umol_L>0));d=d[valid].copy()
    code=d.organism_lifestage.fillna('').astype(str).str.strip().str.upper();d['broad_life_stage_v16']=code.map(LIFE_MAP);d.loc[code.eq(''),'broad_life_stage_v16']='unknown_missing';d.loc[d.broad_life_stage_v16.isna(),'broad_life_stage_v16']='unknown_unmapped';d['life_stage_information_status']=np.select([code.eq('')|code.eq('NR'),code.isin({'--','NC','MX','NOINT'}),code.isin(official)],['unknown_missing','defined_but_unspecified','mapped_defined'],default='unknown_unmapped')
    h=d.exposure_duration_h;d['duration_window_v16']=np.select([h.notna()&(h>0)&(h<=2),h.notna()&(h>2)&(h<=6),h.notna()&(h>6)&(h<=24),h.notna()&(h>24)&(h<=48),h.notna()&(h>48)&(h<=96),h.notna()&(h>96)&(h<=168),h.notna()&(h>168)&(h<=336),h.notna()&(h>336)],['h00_02_ultra_early','h02_06_early','h06_24_same_day','h24_48','h48_96','h96_168','d07_14','gt14d'],default='duration_unknown')
    ef=d.effect_family.fillna('').astype(str);lay=d.effect_evidence_layer.fillna('').astype(str);fam=d.endpoint_family.fillna('').astype(str);lev=d.endpoint_level_band.fillna('').astype(str)
    d['endpoint_band_v16']=np.select([ef.eq('heartbeat'),lay.eq('mechanism_support'),lay.eq('early_warning'),lay.ne('protective_apical'),fam.isin({'no_observed_effect','lowest_observed_effect','derived_chronic_threshold'})|lev.isin({'no_effect','lowest_effect','low_effect_1_20','derived_threshold','zero_effect'}),lev.eq('median_effect_40_60')|fam.isin({'lethal_concentration','lethal_dose'})],['heartbeat_early_warning','mechanistic_support','other_early_warning','other_context','apical_low_effect_threshold','apical_median_effect'],default='apical_other_effect_level')
    cl=d['class'].fillna('').astype(str);sp=d.latin_name.str.lower();kg=d.kingdom.fillna('').astype(str);branch=ef.isin({'mortality_survival','immobilization_intoxication'})&(cl.eq('Branchiopoda')|sp.str.contains('daphnia|ceriodaphnia',regex=True,na=False));fish=ef.eq('mortality_survival')&cl.eq('Actinopterygii');algae=ef.eq('growth')&cl.isin({'Chlorophyceae','Trebouxiophyceae','Cyanophyceae','Cyanobacteriia'});plant=ef.eq('growth')&kg.eq('Plantae')
    d['acute_standard_window_status']=np.select([h.isna()|(h<=0),branch&(h>24)&(h<=48),branch,fish&(h>48)&(h<=96),fish,algae&(h>48)&(h<=96),algae,plant&(h>96)&(h<=240),plant],['not_assessable','standard_48h','nonstandard_for_taxon_endpoint','standard_96h','nonstandard_for_taxon_endpoint','standard_72_96h','nonstandard_for_taxon_endpoint','standard_7d','nonstandard_for_taxon_endpoint'],default='context_specific_no_single_standard')
    d['_log_point']=np.where(d.concentration_point_umol_L>0,np.log10(d.concentration_point_umol_L),np.nan);d['_log_lower']=np.where(d.concentration_lower_umol_L>0,np.log10(d.concentration_lower_umol_L),np.nan);d['_log_upper']=np.where(d.concentration_upper_umol_L>0,np.log10(d.concentration_upper_umol_L),np.nan);ex=d.censoring_class.isin(['exact','approximate_exact'])&np.isfinite(d._log_point);d.loc[ex,'_log_lower']=d.loc[ex,'_log_point'];d.loc[ex,'_log_upper']=d.loc[ex,'_log_point']
    strata=['effect_family','endpoint_band_v16','duration_window_v16','medium_family'];sx=d[ex].copy();sx['_stratum']=sx[strata].astype(str).agg('|'.join,axis=1);st=sx.groupby('_stratum')._log_point.agg(['count','std']).reset_index();gs=float(np.nanmedian(st.loc[st['count']>=10,'std']));gs=gs if np.isfinite(gs) else .65;st['sigma_raw']=st['std'].fillna(gs);st['pooled_sigma_log10']=((st['count']*st.sigma_raw+20*gs)/(st['count']+20)).clip(.15,1.5);st.to_csv(out/'censored_mle_stratum_sigma.csv',index=False);lookup=dict(zip(st._stratum,st.pooled_sigma_log10))
    # audits before aggregation
    d.groupby(['organism_lifestage','broad_life_stage_v16','life_stage_information_status'],dropna=False).size().rename('n_records').reset_index().to_csv(out/'lifestage_mapping_audit.csv',index=False);d.groupby(['effect_family','duration_window_v16']).size().rename('n_records').reset_index().to_csv(out/'duration_window_effect_support.csv',index=False);d[d.effect_family.eq('heartbeat')].groupby('duration_window_v16').size().rename('n_records').reset_index().to_csv(out/'heartbeat_hourly_window_support.csv',index=False);d.groupby('censoring_class').size().rename('n_records').reset_index().to_csv(out/'censoring_summary_eligible.csv',index=False)
    print('prepared eligible records',len(d),flush=True)
    study_keys=['dtxsid','latin_name','effect_family','endpoint_band_v16','duration_window_v16','broad_life_stage_v16','medium_family','reference_number','test_id']
    print('summarizing study cells',flush=True)
    ds=d.copy(); ds['_exact']=ds.censoring_class.isin(['exact','approximate_exact'])&np.isfinite(ds._log_point); ds['_left']=ds.censoring_class.eq('left_censored')&np.isfinite(ds._log_upper); ds['_right']=ds.censoring_class.eq('right_censored')&np.isfinite(ds._log_lower); ds['_interval']=ds.censoring_class.eq('interval_censored')&np.isfinite(ds._log_lower)&np.isfinite(ds._log_upper); ds['_approx']=ds.censoring_class.eq('approximate_exact')
    study=ds.groupby(study_keys,dropna=False,sort=False,observed=True).agg(n_records=('censoring_class','size'),n_exact=('_exact','sum'),n_left=('_left','sum'),n_right=('_right','sum'),n_interval=('_interval','sum'),n_approx=('_approx','sum'),exact_mean_log10_umol_L=('_log_point','mean'),min_lower_log10_umol_L=('_log_lower','min'),max_upper_log10_umol_L=('_log_upper','max')).reset_index()
    study['study_evidence_status']=np.select([(study.n_records==study.n_exact)&(study.n_exact>0),study.n_exact>0,study.n_interval>0,(study.n_left>0)&(study.n_right==0),(study.n_right>0)&(study.n_left==0)],['exact_only','mixed_exact_and_censored','interval_identifiable','upper_bound_only','lower_bound_only'],default='mixed_censored')
    print('study cells',len(study),flush=True);study.to_csv(out/'censored_study_level_cells.csv.gz',index=False,compression='gzip')
    model_keys=['dtxsid','latin_name','effect_family','endpoint_band_v16','duration_window_v16','medium_family'];print('aggregating model cells by censored-normal MLE',flush=True);model=aggregate(d,model_keys,lookup,strata);print('model cells',len(model),flush=True)
    tax=d[['latin_name','kingdom','phylum_division','class','tax_order','family','genus']].drop_duplicates('latin_name').set_index('latin_name');model=model.join(tax,on='latin_name');ln=d.groupby(model_keys).broad_life_stage_v16.nunique().rename('n_broad_life_stages').reset_index();model=model.merge(ln,on=model_keys,how='left');model.to_csv(out/'censored_model_cells.csv.gz',index=False,compression='gzip')
    # Do not duplicate all raw text; save a reconstructable compact record layer.
    compact=['result_id','test_id','reference_number','dtxsid','latin_name','effect_family','endpoint_band_v16','duration_window_v16','broad_life_stage_v16','life_stage_information_status','medium_family','censoring_class','_log_point','_log_lower','_log_upper','acute_standard_window_status','publication_year'];d[compact].to_csv(out/'censored_record_level_compact.csv.gz',index=False,compression='gzip')
    man={'raw_results_rows':raw,'eligible_waterborne_molar_records':len(d),'study_cells':len(study),'model_cells':len(model),'identified_model_cells':int(model.mle_status.str.startswith('identified').sum()),'bound_only_model_cells':int(model.mle_status.isin(['upper_bound_only','lower_bound_only']).sum()),'protective_model_cells_identified':int((model.effect_family.isin(PROTECTIVE)&model.mle_status.str.startswith('identified')).sum()),'heartbeat_records':int((d.effect_family=='heartbeat').sum()),'unknown_unmapped_lifestage_records':int((d.life_stage_information_status=='unknown_unmapped').sum()),'global_pooled_sigma_log10':gs,'method':'censored-normal maximum likelihood by EM with pooled stratum sigma; one-sided-only cells retained as bounds'};(out/'censored_toxicity_manifest.json').write_text(json.dumps(man,indent=2,ensure_ascii=False));print(json.dumps(man,indent=2,ensure_ascii=False))
if __name__=='__main__':main()
