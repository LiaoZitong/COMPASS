#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, math
from pathlib import Path
import numpy as np
import pandas as pd
from numpy.polynomial.hermite import hermgauss
from scipy.stats import norm

PROTECTIVE=['mortality_survival','immobilization_intoxication','growth','reproduction','development_morphology']

def copula_union(p: np.ndarray, rho: float, nodes: int=16) -> np.ndarray:
    p=np.clip(np.asarray(p,float),1e-8,1-1e-8)
    if p.ndim==1:p=p[None,:]
    if p.shape[0]==0:return np.zeros(p.shape[1])
    if p.shape[0]==1:return p[0]
    x,w=hermgauss(nodes);z=np.sqrt(2)*x;w=w/np.sqrt(np.pi)
    thr=norm.ppf(1-p);sr=math.sqrt(max(rho,0));sd=math.sqrt(max(1-rho,1e-8));no=np.zeros(p.shape[1])
    for zz,ww in zip(z,w):no += ww*np.prod(norm.cdf((thr-sr*zz)/sd),axis=0)
    return np.clip(1-no,0,1)

def guild(row) -> str:
    king=str(row.get('kingdom','')).lower(); ph=str(row.get('phylum_division','')).lower(); cl=str(row.get('class','')).lower(); order=str(row.get('tax_order','')).lower(); fam=str(row.get('family','')).lower()
    if king in {'plantae','chromista'} or cl in {'chlorophyceae','bacillariophyceae','cyanophyceae','trebouxiophyceae','zygnematophyceae','ulvophyceae','euglenophyceae'} or 'alga' in cl or 'cyanobacter' in cl:
        return 'primary_producer_algae_plant'
    if cl in {'branchiopoda'} or order in {'cladocera','diplostraca'} or fam in {'daphniidae','moinidae','bosminidae','sididae','chydoridae'}:
        return 'cladoceran_branchiopod'
    if cl in {'malacostraca','maxillopoda','copepoda','ostracoda'} or order in {'amphipoda','isopoda','decapoda','calanoida','cyclopoida'}:
        return 'other_crustacean'
    if cl=='insecta': return 'aquatic_insect'
    if ph=='mollusca': return 'mollusk'
    if cl in {'actinopterygii','actinopteri','teleostei','osteichthyes','chondrichthyes'}: return 'fish'
    if cl=='amphibia': return 'amphibian'
    if ph=='chordata': return 'other_vertebrate'
    if king=='animalia': return 'other_invertebrate'
    return 'other_or_unresolved'

def coverage(P, idx, w, rho):
    return float(w @ copula_union(P[idx],rho)) if len(idx) else 0.0

def beam_scaffold(P, species, guilds, w, rho, k, required, beam_width=80, per_group=20):
    single=P@w
    def fast_cov(indices):
        if not indices:return 0.0
        return float(w @ (1-np.prod(1-P[list(indices)],axis=0)))
    beam=[tuple()]
    for g in required:
        cand=np.flatnonzero(guilds==g)
        cand=cand[np.argsort(single[cand])[::-1][:per_group]]
        expanded=[]
        for b in beam:
            used=set(b)
            for j in cand:
                if int(j) not in used:
                    nb=tuple(list(b)+[int(j)])
                    expanded.append((fast_cov(nb),nb))
        expanded.sort(key=lambda x:(x[0], tuple(species[i] for i in x[1])), reverse=True)
        beam=[b for _,b in expanded[:beam_width]]
    best=max(beam,key=fast_cov)
    sel=list(best)
    while len(sel)<k:
        rem=[j for j in range(len(species)) if j not in set(sel)]
        vals=[fast_cov(sel+[j]) for j in rem]
        sel.append(rem[int(np.argmax(vals))])
    return sel

def panel_summary(method,k,names,cand,P,si,w,rho,group_enrich):
    idx=[si[n] for n in names if n in si]
    sub=cand.set_index('latin_name').reindex(names).reset_index()
    g=sub.guild.fillna('other_or_unresolved')
    counts=g.value_counts()
    enrich=[group_enrich.get(x,np.nan) for x in g]
    return {
        'method':method,'k':k,'n_species':len(names),'coverage_hc5':coverage(P,idx,w,rho),
        'n_guilds':int(g.nunique()),'n_phyla':int(sub.phylum_division.nunique(dropna=True)),
        'n_classes':int(sub['class'].nunique(dropna=True)),'n_families':int(sub.family.nunique(dropna=True)),
        'n_producers':int(counts.get('primary_producer_algae_plant',0)),
        'n_cladocerans':int(counts.get('cladoceran_branchiopod',0)),
        'n_other_crustaceans':int(counts.get('other_crustacean',0)),
        'n_insects':int(counts.get('aquatic_insect',0)),
        'n_mollusks':int(counts.get('mollusk',0)),
        'n_fish':int(counts.get('fish',0)),
        'n_amphibians':int(counts.get('amphibian',0)),
        'mean_observed_guild_hc5_enrichment':float(np.nanmean(enrich)) if np.isfinite(enrich).any() else np.nan,
        'species':' | '.join(names)
    }

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',required=True);ap.add_argument('--target',type=float,default=.95);a=ap.parse_args()
    root=Path(a.root);out=root/'results/complementarity';out.mkdir(parents=True,exist_ok=True)
    xx=int(round(a.target*100)); z=np.load(root/f'results/probability/species_chemical_tail_probability_x{xx}.npz',allow_pickle=True)
    p=z['p'].astype(float);species=z['species'].astype(str);chems=z['chemicals'].astype(str);si={s:i for i,s in enumerate(species)};ci={c:i for i,c in enumerate(chems)}
    pri=pd.read_csv(root/'results/weights/national_priority_chemicals.csv');pri['DTXSID']=pri.DTXSID.astype(str);pri=pri[pri.DTXSID.isin(ci)].copy();wser=pri.set_index('DTXSID').national_weight.astype(float);wser/=wser.sum();cn=wser.index.tolist();cidx=np.array([ci[c] for c in cn]);w=wser.to_numpy(float);P=p[:,cidx]
    manifest=json.loads((root/'results/panels/panel_probability_manifest.json').read_text(encoding='utf-8'));rho=float(manifest['dependence_audit_by_x'][str(a.target)]['rho_working'])
    cand=pd.read_csv(root/'results/probability/candidate_universe_locked.csv');cand['guild']=cand.apply(guild,axis=1);cand=cand.set_index('latin_name').reindex(species).reset_index()
    guilds=cand.guild.fillna('other_or_unresolved').to_numpy(str)

    # Observed HC5 hit enrichment by ecotoxicological guild.
    model=pd.read_csv(root/'results/toxicity/censored_model_cells.csv.gz',low_memory=False)
    valid=model.mle_status.astype(str).str.startswith('identified')&np.isfinite(model.mu_log10_umol_L)
    m=model[valid&model.effect_family.isin(PROTECTIVE)].copy();m['context_id']=m[['dtxsid','effect_family','endpoint_band_v16','duration_window_v16','medium_family']].astype(str).agg('|'.join,axis=1)
    cs=m.groupby('context_id').agg(n_species=('latin_name','nunique'),mean=('mu_log10_umol_L','mean'),sd=('mu_log10_umol_L','std')).reset_index();cs=cs[(cs.n_species>=5)&cs.sd.notna()&(cs.sd>.05)];cs['hc5']=cs['mean']+norm.ppf(.05)*cs.sd
    m=m.merge(cs[['context_id','hc5']],on='context_id');m['tail_hit']=(m.mu_log10_umol_L<=m.hc5).astype(int)
    tax=cand[['latin_name','guild']].drop_duplicates();m=m.merge(tax,on='latin_name',how='left');base=float(m.tail_hit.mean())
    ge=m.groupby('guild').agg(n_observations=('tail_hit','size'),n_species=('latin_name','nunique'),n_chemicals=('dtxsid','nunique'),hc5_hit_rate=('tail_hit','mean')).reset_index();ge['hc5_hit_enrichment_vs_all']=ge.hc5_hit_rate/base;ge['sensitive_guild_rank']=ge.hc5_hit_enrichment_vs_all.rank(method='dense',ascending=False).astype(int);ge.to_csv(out/'observed_hc5_tail_enrichment_by_ecotoxicological_guild.csv',index=False)
    group_enrich=ge.set_index('guild').hc5_hit_enrichment_vs_all.to_dict()

    seq=pd.read_csv(root/'results/panels/national_panel_sequences.csv')
    panels={}
    for method in ['data_driven','taxonomy_diversity_baseline','EPA_WQC_taxonomic_requirements','Canada_Type_A_composition']:
        d=seq[(seq.universe=='priority')&seq.protection_target_x.eq(a.target)&seq.method.eq(method)].sort_values('rank')
        for k in [5,10]:
            if len(d)>=k: panels[(method,k)]=d.head(k).latin_name.astype(str).tolist()
    req=['primary_producer_algae_plant','cladoceran_branchiopod','fish','other_crustacean']
    for k in [5,10]:
        idx=beam_scaffold(P,species,guilds,w,rho,k,req)
        panels[('sensitivity_weighted_guild_scaffold',k)]=species[idx].tolist()
    rows=[]
    for (method,k),names in panels.items(): rows.append(panel_summary(method,k,names,cand,P,si,w,rho,group_enrich))
    pd.DataFrame(rows).sort_values(['k','coverage_hc5'],ascending=[True,False]).to_csv(out/'top5_top10_taxonomic_sensitivity_complementarity.csv',index=False)

    # Species and guild contribution for data-driven and scaffold panels.
    contrib=[];moa_rows=[]
    chem=pd.read_csv(root/'results/chemistry/chemical_moa_hierarchy.csv.gz',low_memory=False);chem['DTXSID']=chem.DTXSID.astype(str);mech=chem.set_index('DTXSID').moa_level_functional.to_dict();mg=np.array([mech.get(c,'FORM::unresolved') for c in cn])
    for (method,k),names in panels.items():
        if method not in {'data_driven','sensitivity_weighted_guild_scaffold'}:continue
        idx=[si[n] for n in names];PP=P[idx];q=copula_union(PP,rho);full=float(w@q)
        lead=np.argmax(PP,axis=0)
        for rank,(n,j) in enumerate(zip(names,idx),1):
            qwo=copula_union(np.delete(PP,rank-1,axis=0),rho) if k>1 else np.zeros(PP.shape[1])
            rr=cand[cand.latin_name.eq(n)].iloc[0]
            contrib.append({'method':method,'k':k,'rank':rank,'latin_name':n,'guild':rr.guild,'class':rr['class'],'family':rr.family,'single_coverage':float(w@P[j]),'leave_one_out_loss':float(w@(q-qwo)),'probability_leader_weight_share':float(w[lead==rank-1].sum()),'n_contexts':rr.n_contexts,'support_tier':rr.support_tier})
        for g in sorted(set(mg)):
            pos=np.flatnonzero(mg==g)
            if len(pos)<2:continue
            wg=w[pos];ws=float(wg.sum())
            if ws<=0:continue
            wg=wg/ws;Qg=copula_union(PP[:,pos],rho);leader=np.argmax(PP[:,pos],axis=0)
            guild_lead={}
            for ii,sp in enumerate(names):
                gu=cand[cand.latin_name.eq(sp)].guild.iloc[0];guild_lead[gu]=guild_lead.get(gu,0)+float(wg[leader==ii].sum())
            moa_rows.append({'method':method,'k':k,'moa_group':g,'n_chemicals':len(pos),'national_weight_fraction':ws,'coverage_within_moa':float(wg@Qg),'n_panel_guilds_contributing_as_probability_leader':sum(v>0 for v in guild_lead.values()),'top_contributing_guild':max(guild_lead,key=guild_lead.get),'top_guild_leader_share':max(guild_lead.values())})
    pd.DataFrame(contrib).to_csv(out/'top5_top10_species_incremental_and_taxonomic_contributions.csv',index=False)
    pd.DataFrame(moa_rows).to_csv(out/'top5_top10_moa_guild_complementarity.csv',index=False)

    man={'version':'v16.5','target':a.target,'primary_panel_sizes':[5,10],'rho':rho,'guild_definition':['primary_producer_algae_plant','cladoceran_branchiopod','other_crustacean','aquatic_insect','mollusk','fish','amphibian','other_vertebrate','other_invertebrate','other_or_unresolved'],'interpretation':'Complementarity is evaluated primarily for top-5 and secondarily top-10. It combines taxonomic/ecological guild representation, enrichment of empirically sensitive guilds, incremental measured-tail expected capture, and MOA-specific contribution. Larger k is not used as the main complementarity argument because random union probabilities become high mechanically.','scaffold_rule':'At least one producer/alga/plant, one cladoceran, one fish, and one other crustacean; remaining slots maximize calibrated measured-tail expected capture. This is a secondary sensitivity-weighted diversity panel, not a regulatory requirement.'}
    (out/'taxonomic_complementarity_manifest.json').write_text(json.dumps(man,indent=2,ensure_ascii=False),encoding='utf-8')
    print(json.dumps(man,indent=2,ensure_ascii=False))
if __name__=='__main__':main()
