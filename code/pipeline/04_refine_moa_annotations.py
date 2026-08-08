#!/usr/bin/env python3
from __future__ import annotations
import argparse, io, json, zipfile
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import sparse

LEVELS = {
    'mie': {
        'chem': 'mech_v6_out/chemical_mechanism_chemical_index.csv',
        'feat': 'mech_v6_out/chemical_mechanism_feature_index.csv',
        'active': 'mech_v6_out/chemical_mechanism_active_score.npz',
        'conf': 'mech_v6_out/chemical_mechanism_confidence.npz',
    },
    'route': {
        'chem': 'mech_v6_out/chemical_route_chemical_index.csv',
        'feat': 'mech_v6_out/chemical_route_feature_index.csv',
        'active': 'mech_v6_out/chemical_route_active_score.npz',
        'conf': 'mech_v6_out/chemical_route_confidence.npz',
    },
    'chain': {
        'chem': 'mech_v6_out/chemical_chain_chemical_index.csv',
        'feat': 'mech_v6_out/chemical_chain_feature_index.csv',
        'active': 'mech_v6_out/chemical_chain_active_score.npz',
        'conf': 'mech_v6_out/chemical_chain_confidence.npz',
    },
}

def load_npz_from_zip(z: zipfile.ZipFile, name: str):
    return sparse.load_npz(io.BytesIO(z.read(name))).tocsr()

def top_features(z: zipfile.ZipFile, spec: dict, level: str, threshold: float, topn: int = 3) -> pd.DataFrame:
    chem = pd.read_csv(z.open(spec['chem'])).sort_values('row_idx')
    feat = pd.read_csv(z.open(spec['feat']))
    active = load_npz_from_zip(z, spec['active'])
    conf = load_npz_from_zip(z, spec['conf'])
    if active.shape != conf.shape:
        raise ValueError(f'{level} active/conf shape mismatch')
    feat = feat.set_index('col_idx', drop=False)
    rows=[]
    for i, dtxsid in enumerate(chem.dtxsid.astype(str)):
        a = active.getrow(i); c = conf.getrow(i)
        amap = dict(zip(a.indices.astype(int), a.data.astype(float)))
        cmap = dict(zip(c.indices.astype(int), c.data.astype(float)))
        cand=[]
        for j, av in amap.items():
            if av < threshold or j not in feat.index:
                continue
            cv=float(cmap.get(j,0.0))
            meta=feat.loc[j]
            score=float(av*max(cv,0.25))
            if level=='mie' and str(meta.get('feature_family','')).upper()!='MIE':
                continue
            if level=='route':
                # Route labels are summarized as upstream MIE -> downstream AO where available.
                up=str(meta.get('upstream_mie_names','')).strip()
                down=str(meta.get('downstream_ao_names','')).strip()
                label=f'{up} => {down}' if up and down else str(meta.get('chain_context') or meta.get('feature_name'))
            elif level=='chain':
                label=str(meta.get('feature_name','')).strip()
            else:
                label=str(meta.get('feature_name','')).strip()
            cand.append((score,float(av),cv,str(meta.get('feature_id','')),label))
        cand=sorted(cand, reverse=True)[:topn]
        row={'DTXSID':dtxsid}
        for k in range(topn):
            if k < len(cand):
                sc,av,cv,fid,label=cand[k]
                row[f'{level}_{k+1}_label']=label
                row[f'{level}_{k+1}_active']=av
                row[f'{level}_{k+1}_confidence']=cv
                row[f'{level}_{k+1}_rank_score']=sc
                row[f'{level}_{k+1}_id']=fid
            else:
                row[f'{level}_{k+1}_label']=''; row[f'{level}_{k+1}_active']=np.nan
                row[f'{level}_{k+1}_confidence']=np.nan; row[f'{level}_{k+1}_rank_score']=np.nan; row[f'{level}_{k+1}_id']=''
        rows.append(row)
    return pd.DataFrame(rows)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--aop-zip',required=True);ap.add_argument('--chemical-master',required=True);ap.add_argument('--chemical-annotations',help='Step-03 chemical annotation table supplying form and coarse-MOA fields.');ap.add_argument('--out',required=True);ap.add_argument('--threshold',type=float,default=0.1)
    a=ap.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    z=zipfile.ZipFile(a.aop_zip)
    # The raw chemical master deliberately contains only reference metadata.
    # Step 03 supplies the reproducible form/MOA annotations consumed here.
    cm=pd.read_csv(a.chemical_annotations or a.chemical_master,low_memory=False);cm['DTXSID']=cm.DTXSID.astype(str)
    required=['DTXSID','PREFERRED_NAME','chemical_form_class','primary_moa','mie_1','mie_1_score','mie_2','mie_2_score','mie_3','mie_3_score']
    missing=[column for column in required if column not in cm.columns]
    if missing:
        raise ValueError('Chemical annotations are missing required columns: '+', '.join(missing))
    merged=cm[required].copy()
    for level,spec in LEVELS.items():
        t=top_features(z,spec,level,a.threshold,3)
        merged=merged.merge(t,on='DTXSID',how='left')
    # Stable hierarchy labels. Fine levels are used only after whole-chemical validation.
    merged['moa_level_form']='FORM::'+merged.chemical_form_class.fillna('unresolved').astype(str)
    merged['moa_level_mie']=np.where(merged.mie_1_active.fillna(0)>=a.threshold,'MIE::'+merged.mie_1_label.fillna('').astype(str),merged.moa_level_form)
    def functional_family(label: object, fallback: str) -> str:
        t=str(label or '').lower()
        if not t.strip(): return fallback
        if any(k in t for k in ['deiodinase','thyroid hormone','thyroperoxidase','na+/i- symporter']): return 'MOAFAM::thyroid_axis'
        if any(k in t for k in ['voltage-gated sodium','nmda','ionotropic glutamate']): return 'MOAFAM::neurotransmission_ion_channel'
        if any(k in t for k in ['androgen receptor','ppar alpha','activation, ahr']): return 'MOAFAM::nuclear_receptor_xenobiotic'
        if any(k in t for k in ['oxidative phosphorylation','complex i','deposition of energy']): return 'MOAFAM::mitochondrial_energy'
        if any(k in t for k in ['oxidative dna','dna strand','thiol/seleno','alkylation, protein']): return 'MOAFAM::oxidative_genotoxic_reactivity'
        if 'cyp2e1' in t: return 'MOAFAM::xenobiotic_metabolism'
        if any(k in t for k in ['histone deacetylase','vegfr2','il-1r1','calcineurin','cell membrane']): return 'MOAFAM::cell_signaling_epigenetic'
        return 'MOAFAM::other_specific_mie'
    merged['moa_level_functional']=[functional_family(l,f) if float(a1 or 0)>=a.threshold else f for l,f,a1 in zip(merged.mie_1_label,merged.moa_level_form,merged.mie_1_active.fillna(0))]
    merged['moa_level_mie_signature']=np.where(
        merged.mie_1_active.fillna(0)>=a.threshold,
        'MIESIG::'+merged.apply(lambda r:' || '.join([str(r[f'mie_{k}_label']) for k in (1,2,3) if pd.notna(r[f'mie_{k}_active']) and r[f'mie_{k}_active']>=a.threshold and str(r[f'mie_{k}_label']).strip()]),axis=1),
        merged.moa_level_form,
    )
    merged['moa_level_route']=np.where(merged.route_1_active.fillna(0)>=a.threshold,'ROUTE::'+merged.route_1_label.fillna('').astype(str),merged.moa_level_mie)
    merged['moa_level_chain']=np.where(merged.chain_1_active.fillna(0)>=a.threshold,'CHAIN::'+merged.chain_1_label.fillna('').astype(str),merged.moa_level_route)
    levels=['moa_level_form','moa_level_functional','moa_level_mie','moa_level_mie_signature','moa_level_route','moa_level_chain']
    merged.to_csv(out/'chemical_moa_hierarchy.csv.gz',index=False,compression='gzip')
    rows=[]
    for lev in levels:
        vc=merged[lev].value_counts(dropna=False)
        rows.append({'level':lev,'n_groups':int(vc.size),'n_groups_ge5':int((vc>=5).sum()),'n_groups_ge10':int((vc>=10).sum()),'n_groups_ge20':int((vc>=20).sum()),'largest_group_n':int(vc.max()),'unresolved_or_form_fraction':float(merged[lev].astype(str).str.startswith('FORM::').mean())})
    pd.DataFrame(rows).to_csv(out/'moa_hierarchy_support_summary.csv',index=False)
    man={'version':'v16.5','threshold':a.threshold,'chemicals':len(merged),'levels':levels,'policy':'Fine MIE signatures, AOP routes and AOP chains are candidate probability-borrowing strata only. Whole-chemical OOF validation selects the level by effect family; no absolute toxicity value is predicted.'}
    (out/'moa_hierarchy_manifest.json').write_text(json.dumps(man,indent=2,ensure_ascii=False),encoding='utf-8')
    print(json.dumps(man,indent=2,ensure_ascii=False))
if __name__=='__main__':main()
