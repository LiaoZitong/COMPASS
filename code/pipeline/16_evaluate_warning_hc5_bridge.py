#!/usr/bin/env python3
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import kendalltau, norm, spearmanr, pearsonr

PROTECTIVE=['mortality_survival','immobilization_intoxication','growth','reproduction','development_morphology']
WARNING=['heartbeat','behavior_feeding_avoidance','other_physiology']

def metrics(d, value, ref):
    x=d[value].to_numpy(float); y=d[ref].to_numpy(float); ok=np.isfinite(x)&np.isfinite(y);x=x[ok];y=y[ok]
    if len(x)<3:return {'n':len(x)}
    delta=x-y
    return {'n':len(x),'spearman_r':float(spearmanr(x,y).statistic),'pearson_r':float(pearsonr(x,y).statistic),'median_log10_ratio':float(np.median(delta)),'median_ratio':float(10**np.median(delta)),'within_factor2':float(np.mean(np.abs(delta)<=np.log10(2))),'within_factor3':float(np.mean(np.abs(delta)<=np.log10(3))),'within_factor5':float(np.mean(np.abs(delta)<=np.log10(5))),'within_factor10':float(np.mean(np.abs(delta)<=1)),'warning_at_or_below_hc5':float(np.mean(delta<=0)),'false_safe_factor2':float(np.mean(delta>np.log10(2))),'false_safe_factor5':float(np.mean(delta>np.log10(5))),'false_safe_factor10':float(np.mean(delta>1))}

def bootstrap(d,value,ref,nboot=500,seed=20260622):
    rng=np.random.default_rng(seed)
    groups=[]
    for _,g in d.groupby('dtxsid',sort=False):
        x=g[value].to_numpy(float);y=g[ref].to_numpy(float);ok=np.isfinite(x)&np.isfinite(y)
        if ok.any():groups.append((x[ok],y[ok]))
    if len(groups)<5:return pd.DataFrame()
    rows=[];ng=len(groups)
    for b in range(nboot):
        ids=rng.integers(0,ng,size=ng)
        x=np.concatenate([groups[i][0] for i in ids]);y=np.concatenate([groups[i][1] for i in ids]);delta=x-y
        rows.append({'bootstrap':b,'n':len(x),'spearman_r':float(spearmanr(x,y).statistic) if len(x)>=3 else np.nan,'pearson_r':float(pearsonr(x,y).statistic) if len(x)>=3 else np.nan,'median_log10_ratio':float(np.median(delta)),'median_ratio':float(10**np.median(delta)),'within_factor2':float(np.mean(np.abs(delta)<=np.log10(2))),'within_factor3':float(np.mean(np.abs(delta)<=np.log10(3))),'within_factor5':float(np.mean(np.abs(delta)<=np.log10(5))),'within_factor10':float(np.mean(np.abs(delta)<=1)),'warning_at_or_below_hc5':float(np.mean(delta<=0)),'false_safe_factor2':float(np.mean(delta>np.log10(2))),'false_safe_factor5':float(np.mean(delta>np.log10(5))),'false_safe_factor10':float(np.mean(delta>1))})
    return pd.DataFrame(rows)

def taxonomic_flags(g):
    ph=g.phylum_division.fillna('').str.lower();cl=g['class'].fillna('').str.lower();order=g.tax_order.fillna('').str.lower();fam=g.family.fillna('').str.lower()
    fish=ph.eq('chordata')&cl.isin(['actinopterygii','actinopteri','teleostei','osteichthyes'])
    return {'salmonid':(fish&fam.eq('salmonidae')).to_numpy(),'non_salmonid_fish':(fish&~fam.eq('salmonidae')).to_numpy(),'chordate':ph.eq('chordata').to_numpy(),'plank_crust':(cl.isin(['branchiopoda','maxillopoda','copepoda'])|order.isin(['cladocera','diplostraca','calanoida','cyclopoida'])).to_numpy(),'benthic_crust':(cl.eq('malacostraca')|order.isin(['amphipoda','isopoda','decapoda'])).to_numpy(),'insect':cl.eq('insecta').to_numpy(),'other_phylum':(~ph.isin(['arthropoda','chordata',''])).to_numpy(),'family':fam.to_numpy()}

def choose_epa_1985_slots(g):
    fl=taxonomic_flags(g)
    masks=[fl['salmonid'],fl['non_salmonid_fish'],fl['chordate'],fl['plank_crust'],fl['benthic_crust'],fl['insect'],fl['other_phylum'],np.ones(len(g),bool)]
    used=set();families=set();sel=[]
    order=sorted(range(len(masks)),key=lambda i:int(np.sum(masks[i])))
    for slot in order:
        ix=np.flatnonzero(masks[slot]);ix=[int(i) for i in ix if int(i) not in used and fl['family'][i] not in families]
        if not ix:return None
        j=max(ix,key=lambda z:g.iloc[z].latin_name)
        used.add(j);families.add(fl['family'][j]);sel.append((slot,j))
    return [j for _,j in sorted(sel)]

def epa_context_summary(p):
    rows=[]
    taxa=p.sort_values(['context_id','latin_name']).drop_duplicates(['context_id','latin_name'])
    for context_id,g in taxa.groupby('context_id',sort=False):
        gg=g.reset_index(drop=True);slots=choose_epa_1985_slots(gg)
        selected=gg.iloc[slots] if slots is not None else pd.DataFrame()
        rows.append({'context_id':context_id,'epa_1985_8_slots_met':slots is not None,'n_epa_1985_slot_species':0 if slots is None else len(slots),'epa_1985_slot_species':'' if slots is None else ' | '.join(selected.latin_name.astype(str)),'epa_1985_slot_families':'' if slots is None else ' | '.join(selected.family.fillna('').astype(str))})
    return pd.DataFrame(rows)


def shared_input_sensitivity(p, eligible_contexts, panel_rows, top5, out):
    """Refit the same EPA-eligible contexts after excluding all Top-5 species."""
    retained_ids = set(eligible_contexts.context_id.astype(str))
    non_top5 = p[
        p.context_id.astype(str).isin(retained_ids)
        & ~p.latin_name.astype(str).isin(top5)
    ].copy()
    context_rows = (
        non_top5.groupby('context_id')
        .agg(
            dtxsid=('dtxsid', 'first'),
            medium_family=('medium_family', 'first'),
            effect_family=('effect_family', 'first'),
            n_species_non_w5=('latin_name', 'nunique'),
            mean_non_w5=('mu_log10_umol_L', 'mean'),
            sd_non_w5=('mu_log10_umol_L', 'std'),
            n_model_cells_non_w5=('mu_log10_umol_L', 'size'),
        )
        .reset_index()
    )
    context_rows = context_rows[
        context_rows.n_species_non_w5.ge(5)
        & context_rows.sd_non_w5.notna()
        & context_rows.sd_non_w5.gt(.05)
    ].copy()
    context_rows['reference_hc5_non_w5_log10'] = (
        context_rows.mean_non_w5 + norm.ppf(.05) * context_rows.sd_non_w5
    )
    context_rows.to_csv(out/'hc5_excluding_top5_reference_contexts.csv', index=False)

    chemical_medium = (
        context_rows.groupby(['dtxsid', 'medium_family'])
        .agg(
            reference_hc5_non_w5_conservative_log10=('reference_hc5_non_w5_log10', 'min'),
            reference_hc5_non_w5_typical_log10=('reference_hc5_non_w5_log10', 'median'),
            n_non_w5_reference_contexts=('context_id', 'nunique'),
            n_non_w5_effect_families=('effect_family', 'nunique'),
            n_non_w5_species_min=('n_species_non_w5', 'min'),
            n_non_w5_species_median=('n_species_non_w5', 'median'),
        )
        .reset_index()
    )
    chemical_medium.to_csv(
        out/'hc5_excluding_top5_chemical_medium_reference_hc5.csv', index=False
    )
    paired = panel_rows.merge(
        chemical_medium,
        on=['dtxsid', 'medium_family'],
        how='inner',
        validate='one_to_one',
    )
    paired.to_csv(out/'hc5_excluding_top5_top5_apical_paired_rows.csv', index=False)

    def correlation_row(label, x_col, y_col, ratio=False):
        x = paired[x_col].to_numpy(float)
        y = paired[y_col].to_numpy(float)
        row = {
            'comparison': label,
            'paired_rows': len(paired),
            'spearman_rho': float(spearmanr(x, y).statistic),
            'pearson_r': float(pearsonr(x, y).statistic),
            'kendall_tau': float(kendalltau(x, y).statistic),
            'median_top5_apical_to_reference_ratio': np.nan,
            'within_factor_5': np.nan,
            'within_factor_10': np.nan,
            'reference_contexts': len(context_rows),
            'chemical_medium_references': len(chemical_medium),
        }
        if ratio:
            delta = x - y
            row.update(
                median_top5_apical_to_reference_ratio=float(10 ** np.median(delta)),
                within_factor_5=float(np.mean(np.abs(delta) <= np.log10(5))),
                within_factor_10=float(np.mean(np.abs(delta) <= 1)),
            )
        return row

    metric_rows = [
        {
            **correlation_row(
                'Top5-apical versus EPA-framework full-data HC5',
                'panel_min_apical_log10',
                'reference_hc5_typical_log10',
                ratio=True,
            ),
            'definition': (
                'Top5-apical compared with the median context-specific EPA-framework '
                'full-data HC5 for each chemical-medium row'
            ),
        },
        {
            **correlation_row(
                'Top5-apical versus HC5 excluding Top-5 species',
                'panel_min_apical_log10',
                'reference_hc5_non_w5_typical_log10',
                ratio=True,
            ),
            'definition': (
                'The same EPA-eligible contexts refitted after excluding all Top-5 '
                'species; retained with at least five remaining species and SD >0.05 '
                'log10 units'
            ),
        },
        {
            **correlation_row(
                'EPA-framework full-data HC5 versus HC5 excluding Top-5 species',
                'reference_hc5_typical_log10',
                'reference_hc5_non_w5_typical_log10',
            ),
            'definition': (
                'Shared-input sensitivity of the chemical-medium HC5 ranks in the '
                'same paired rows'
            ),
        },
    ]
    pd.DataFrame(metric_rows).to_csv(
        out/'hc5_excluding_top5_sensitivity_metrics.csv', index=False
    )
    return context_rows, chemical_medium, paired

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',required=True);ap.add_argument('--bootstrap',type=int,default=500);a=ap.parse_args();root=Path(a.root);out=root/'results/warning_hc5_bridge';out.mkdir(parents=True,exist_ok=True)
    m=pd.read_csv(root/'results/toxicity/censored_model_cells.csv.gz',low_memory=False)
    valid=m.mle_status.astype(str).str.startswith('identified')&np.isfinite(m.mu_log10_umol_L)
    m=m[valid].copy();m['dtxsid']=m.dtxsid.astype(str)
    p=m[m.effect_family.isin(PROTECTIVE)].copy();p['context_id']=p[['dtxsid','effect_family','endpoint_band_v16','duration_window_v16','medium_family']].astype(str).agg('|'.join,axis=1)
    epa=epa_context_summary(p)
    cs=p.groupby('context_id').agg(dtxsid=('dtxsid','first'),medium_family=('medium_family','first'),effect_family=('effect_family','first'),n_species=('latin_name','nunique'),mean=('mu_log10_umol_L','mean'),sd=('mu_log10_umol_L','std')).reset_index().merge(epa,on='context_id',how='left');cs=cs[cs.epa_1985_8_slots_met.fillna(False)&cs.sd.notna()&(cs.sd>.05)];cs['reference_hc5_log10']=cs['mean']+norm.ppf(.05)*cs.sd
    cs.to_csv(out/'epa_1985_reference_hc5_contexts.csv',index=False)
    cref=cs.groupby(['dtxsid','medium_family']).agg(reference_hc5_conservative_log10=('reference_hc5_log10','min'),reference_hc5_typical_log10=('reference_hc5_log10','median'),n_reference_contexts=('context_id','nunique'),n_reference_effect_families=('effect_family','nunique'),n_reference_species_min=('n_species','min'),n_reference_species_median=('n_species','median')).reset_index()
    cref.to_csv(out/'chemical_medium_reference_hc5.csv',index=False)

    w=m[m.effect_family.isin(WARNING)].copy();w['warning_time_group']=w.duration_window_v16.replace({'h00_02_ultra_early':'0-2h','h02_06_early':'2-6h','h06_24_same_day':'6-24h','h24_48':'24-48h','h48_96':'48h-7d','h96_168':'48h-7d','d04_07':'48h-7d','d07_14':'>7d','gt14d':'>7d','gt7d':'>7d','duration_unknown':'unknown'})
    wp=w.groupby(['dtxsid','latin_name','medium_family','warning_time_group']).agg(warning_threshold_log10=('mu_log10_umol_L','min'),n_warning_model_cells=('mu_log10_umol_L','size'),warning_effect_families=('effect_family','nunique')).reset_index().merge(cref,on=['dtxsid','medium_family'],how='inner')
    wp.to_csv(out/'paired_warning_threshold_vs_reference_hc5.csv.gz',index=False,compression='gzip')

    seq=pd.read_csv(root/'results/panels/national_panel_sequences.csv');top5=seq[(seq.universe=='priority')&seq.protection_target_x.eq(.95)&seq.method.eq('data_driven')].sort_values('rank').head(5).latin_name.astype(str).tolist()
    wp['scope']=np.where(wp.latin_name.isin(top5),'hc5_top5_species','all_warning_species')
    rows=[];boots=[]
    for scope,d0 in [('all_warning_species',wp),('hc5_top5_species',wp[wp.latin_name.isin(top5)])]:
        for tg,d in [('all_times',d0),*[(x,d0[d0.warning_time_group.eq(x)]) for x in ['0-2h','2-6h','6-24h','24-48h','48h-7d','>7d']]]:
            for ref in ['reference_hc5_typical_log10','reference_hc5_conservative_log10']:
                rec={'scope':scope,'warning_time_group':tg,'reference':ref,**metrics(d,'warning_threshold_log10',ref)};rows.append(rec)
                if tg=='all_times' and len(d)>=10:
                    b=bootstrap(d,'warning_threshold_log10',ref,a.bootstrap); 
                    if len(b):b['scope']=scope;b['warning_time_group']=tg;b['reference']=ref;boots.append(b)
    summary=pd.DataFrame(rows);summary.to_csv(out/'warning_hc5_concordance_summary.csv',index=False)
    if boots:pd.concat(boots,ignore_index=True).to_csv(out/'warning_hc5_concordance_bootstrap.csv.gz',index=False,compression='gzip')

    # Panel-level minimum warning threshold by chemical/medium, closest to operational alert use.
    pw=wp[wp.latin_name.isin(top5)].groupby(['dtxsid','medium_family']).agg(panel_min_warning_log10=('warning_threshold_log10','min'),n_top5_species_with_warning=('latin_name','nunique'),earliest_warning_group=('warning_time_group',lambda x:' | '.join(sorted(set(x))))).reset_index().merge(cref,on=['dtxsid','medium_family'])
    pw.to_csv(out/'top5_panel_warning_threshold_vs_reference_hc5.csv',index=False)
    panel_rows=[]
    for ref in ['reference_hc5_typical_log10','reference_hc5_conservative_log10']:
        panel_rows.append({'scope':'top5_panel_minimum_warning','reference':ref,**metrics(pw,'panel_min_warning_log10',ref)})
    pd.DataFrame(panel_rows).to_csv(out/'top5_panel_warning_hc5_metrics.csv',index=False)

    # Protective measured panel minimum as a positive control for linkage to reference HC5.
    pm=p[p.latin_name.isin(top5)].groupby(['dtxsid','medium_family']).agg(panel_min_apical_log10=('mu_log10_umol_L','min'),n_top5_species_with_apical=('latin_name','nunique')).reset_index().merge(cref,on=['dtxsid','medium_family'])
    pm.to_csv(out/'top5_panel_apical_threshold_vs_reference_hc5.csv',index=False)
    pos=[]
    for ref in ['reference_hc5_typical_log10','reference_hc5_conservative_log10']:pos.append({'scope':'top5_panel_minimum_apical','reference':ref,**metrics(pm,'panel_min_apical_log10',ref)})
    pd.DataFrame(pos).to_csv(out/'top5_panel_apical_hc5_positive_control_metrics.csv',index=False)

    non_top5_contexts, non_top5_references, non_top5_paired = shared_input_sensitivity(
        p, cs, pm, top5, out
    )

    man={'version':'v16.5_epa_reference','panel_top5':top5,'reference_definition':{'context_eligibility':'EPA_1985_8_slots feasibility from the project HC5 framework comparison: salmonid, non-salmonid fish, chordate, planktonic crustacean, benthic crustacean, insect, other phylum and one additional distinct-family slot.','typical':'median HC5 across EPA-eligible identifiable protective contexts for the same chemical and medium','conservative':'minimum HC5 across those EPA-eligible contexts','fit_species':'reference HC5 is fitted from all measured protective species within each EPA-eligible context, not from the selected slot representatives only'},'shared_input_sensitivity':{'contexts_retained':len(non_top5_contexts),'chemical_medium_rows_retained':len(non_top5_references),'paired_rows_retained':len(non_top5_paired),'definition':'The same EPA-eligible contexts were refitted after excluding all five national Top-5 species; contexts required at least five remaining protective species and SD >0.05 log10 units.'},'warning_definition':'minimum observed censored-MLE warning threshold within species x chemical x medium x time group; no external Daphnia 1 h data','interpretation_boundary':'This is a bridge validation, not a claim that heterogeneous warning endpoints are interchangeable with a regulatory HC5. Strong correlation, near-unity ratios, and low false-safe rates are empirical requirements; if not met, warning evidence remains a prioritization layer.'}
    (out/'warning_hc5_bridge_manifest.json').write_text(json.dumps(man,indent=2,ensure_ascii=False),encoding='utf-8')
    print(summary.to_string(index=False));print(pd.DataFrame(panel_rows).to_string(index=False));print(json.dumps(man,indent=2,ensure_ascii=False))
if __name__=='__main__':main()
