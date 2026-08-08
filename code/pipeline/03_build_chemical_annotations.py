#!/usr/bin/env python3
from __future__ import annotations
import argparse, io, json, re, zipfile
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import sparse

METALS={'Li','Be','Na','Mg','Al','K','Ca','Sc','Ti','V','Cr','Mn','Fe','Co','Ni','Cu','Zn','Ga','Rb','Sr','Y','Zr','Nb','Mo','Tc','Ru','Rh','Pd','Ag','Cd','In','Sn','Cs','Ba','La','Ce','Pr','Nd','Pm','Sm','Eu','Gd','Tb','Dy','Ho','Er','Tm','Yb','Lu','Hf','Ta','W','Re','Os','Ir','Pt','Au','Hg','Tl','Pb','Bi','Th','U'}

def elements(formula):
    return re.findall(r'[A-Z][a-z]?',str(formula or ''))

def classify(row):
    sm=str(row.get('SMILES') or ''); qs=str(row.get('QSAR_READY_SMILES') or ''); fm=str(row.get('MOLECULAR_FORMULA') or row.get('Formula') or '')
    els=set(elements(fm)); mets=sorted(els & METALS); carbon='C' in els or bool(re.search(r'(?<![a-z])C',sm)); multi='.' in sm; charged=bool(re.search(r'\[[^\]]*[+-][^\]]*\]',sm))
    heavy=str(row.get('Is_Heavy_Metal','')).strip().lower() in {'yes','true','1','y'}
    if not sm and not qs and not fm: cls='structure_unresolved'
    elif mets:
        if len(els)==1 and not carbon: cls='elemental_metal_or_single_metal_species'
        elif carbon and (multi or charged): cls='organometallic_or_metal_complex'
        elif carbon: cls='carbon_containing_metal_compound'
        elif multi or charged: cls='metal_salt_or_ionic_complex'
        else: cls='inorganic_metal_compound'
    elif multi or charged: cls='organic_salt_or_multicomponent'
    else: cls='discrete_nonmetal_chemical'
    parent=qs.strip() if qs.strip() else sm.strip()
    if cls in {'organic_salt_or_multicomponent'} and not qs.strip() and sm:
        comps=[x for x in sm.split('.') if 'C' in x]; parent=max(comps,key=len) if comps else sm
    if mets:
        rule='metal read-across requires same metal element and compatible oxidation/ligand/form class; QSAR-ready parent alone is insufficient'
    elif cls=='organic_salt_or_multicomponent': rule='parent-structure read-across allowed only as a probability prior, using QSAR-ready/organic parent and MOA compatibility'
    else: rule='parent-structure identity may gate MOA probability transfer; no absolute toxicity prediction'
    return pd.Series({'chemical_form_class':cls,'metal_elements':';'.join(mets),'contains_carbon':carbon,'multicomponent_smiles':multi,'explicit_charge_smiles':charged,'heavy_metal_flag':heavy,'parent_readacross_smiles':parent,'readacross_rule':rule})

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--chemical-master',required=True);ap.add_argument('--aop-data',required=True);ap.add_argument('--traits',required=True);ap.add_argument('--out',required=True);a=ap.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    cm=pd.read_csv(a.chemical_master,low_memory=False);cm['DTXSID']=cm.DTXSID.astype(str).str.strip().str.upper();form=cm.apply(classify,axis=1);cm=pd.concat([cm,form],axis=1)
    z=zipfile.ZipFile(a.aop_data)
    chem=pd.read_csv(z.open('mech_v6_out/chemical_mechanism_chemical_index.csv'));feat=pd.read_csv(z.open('mech_v6_out/chemical_mechanism_feature_index.csv'))
    active=sparse.load_npz(io.BytesIO(z.read('mech_v6_out/chemical_mechanism_active_score.npz'))).tocsr();conf=sparse.load_npz(io.BytesIO(z.read('mech_v6_out/chemical_mechanism_confidence.npz'))).tocsr()
    mie=feat[feat.feature_family.astype(str).str.upper().eq('MIE')].copy();mie_cols=set(mie.col_idx.astype(int));fmap=mie.set_index('col_idx')[['feature_id','feature_name','mechanism_source','source_confidence']].to_dict('index')
    rows=[]
    for i,r in chem.iterrows():
        s,e=active.indptr[i],active.indptr[i+1];idx=active.indices[s:e];val=active.data[s:e]
        mask=np.fromiter((int(j) in mie_cols for j in idx),dtype=bool,count=len(idx));idx=idx[mask];val=val[mask]
        if len(idx):
            scores=val
            order=np.argsort(scores)[::-1][:3]
            top=[]
            for o in order:
                j=int(idx[o]); meta=fmap[j]; top.append((meta['feature_name'],float(scores[o]),meta['feature_id']))
        else: top=[]
        rr={'DTXSID':str(r.dtxsid).upper(),'n_active_mie':int(np.sum(val>=0.1)) if len(val) else 0,'moa_evidence':'direct_active_MIE' if len(top) and top[0][1]>=0.1 else 'no_active_MIE_at_threshold'}
        for k in range(3):
            rr[f'mie_{k+1}']=top[k][0] if k<len(top) else ''
            rr[f'mie_{k+1}_score']=top[k][1] if k<len(top) else np.nan
            rr[f'mie_{k+1}_id']=top[k][2] if k<len(top) else ''
        rows.append(rr)
    moa=pd.DataFrame(rows);cm=cm.merge(moa,on='DTXSID',how='left');cm['primary_moa']=np.where(cm.mie_1_score.fillna(0)>=0.1,'MIE::'+cm.mie_1.astype(str),'MOA_UNRESOLVED')
    cm.to_csv(out/'chemical_moa_form_master.csv.gz',index=False,compression='gzip')
    cm[['DTXSID','PREFERRED_NAME','chemical_form_class','metal_elements','contains_carbon','multicomponent_smiles','explicit_charge_smiles','parent_readacross_smiles','readacross_rule','primary_moa','mie_1_score','n_active_mie','moa_evidence']].to_csv(out/'chemical_form_and_moa_audit.csv',index=False)
    tr=pd.read_csv(a.traits,low_memory=False);tr['latin_name']=tr.latin_name.astype(str).str.strip();tr.to_csv(out/'species_traits_taxonomy.csv.gz',index=False,compression='gzip')
    numeric=['trait_body_length_cm','trait_body_mass_g','trait_trophic_level','trait_longevity_y','trait_maturity_y','trait_generation_time_y','trait_habitat_breadth','trait_diet_breadth']
    cov=[]
    for c in numeric: cov.append({'field':c,'n_nonmissing':int(pd.to_numeric(tr[c],errors='coerce').notna().sum()),'fraction_nonmissing':float(pd.to_numeric(tr[c],errors='coerce').notna().mean())})
    pd.DataFrame(cov).to_csv(out/'trait_coverage.csv',index=False)
    metal_classes={'organometallic_or_metal_complex','metal_salt_or_ionic_complex','carbon_containing_metal_compound','elemental_metal_or_single_metal_species','inorganic_metal_compound'}
    man={'chemicals':len(cm),'direct_moa_at_threshold':int(cm.moa_evidence.eq('direct_active_MIE').sum()),'metal_or_metal_complex':int(cm.chemical_form_class.isin(metal_classes).sum()),'traits_species':len(tr),'policy':'MOA/taxonomy/traits transfer only sensitivity-tail probabilities, effect-family occurrence, and SSD shape class; no absolute LC50/EC50/HC5 prediction'}
    (out/'chemistry_moa_traits_manifest.json').write_text(json.dumps(man,indent=2,ensure_ascii=False));print(json.dumps(man,indent=2,ensure_ascii=False))
if __name__=='__main__':main()
