#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, shutil
from pathlib import Path
import pandas as pd

TARGET_LABELS = ['x80','x90','x95']

def main():
    ap=argparse.ArgumentParser(description='Deterministically merge target-specific v16.5 outputs.')
    ap.add_argument('--x80-dir',required=True); ap.add_argument('--x90-dir',required=True); ap.add_argument('--x95-dir',required=True)
    ap.add_argument('--out-prob',required=True); ap.add_argument('--out-panels',required=True)
    a=ap.parse_args()
    runs=[Path(a.x80_dir),Path(a.x90_dir),Path(a.x95_dir)]
    outp=Path(a.out_prob); outpan=Path(a.out_panels); outp.mkdir(parents=True,exist_ok=True); outpan.mkdir(parents=True,exist_ok=True)
    # Target-specific probability files have unique x labels; common audit files are deterministic and identical.
    for rd in runs:
        for f in (rd/'prob').iterdir():
            if f.is_file(): shutil.copy2(f,outp/f.name)
    for name in ['national_panel_sequences.csv','national_k1_20_coverage_curves.csv','daphnia_magna_audit.csv']:
        frames=[pd.read_csv(rd/'panels'/name) for rd in runs if (rd/'panels'/name).exists()]
        pd.concat(frames,ignore_index=True).to_csv(outpan/name,index=False)
    for name in ['priority_weight_panel_sensitivity.csv','x90_method_comparison.csv']:
        src=runs[1]/'panels'/name
        if src.exists(): shutil.copy2(src,outpan/name)
    manifests=[json.loads((rd/'panels/panel_probability_manifest.json').read_text(encoding='utf-8')) for rd in runs]
    merged=manifests[0]
    merged['version']='v16.5'
    merged['main_protection_target']=0.95
    merged['sensitivity_targets']=[0.80,0.90,0.95]
    merged['dependence_audit_by_x']={}
    for m in manifests: merged['dependence_audit_by_x'].update(m.get('dependence_audit_by_x',{}))
    merged['execution']='x80/x90/x95 are run in isolated subprocesses and merged deterministically to bound peak memory.'
    (outpan/'panel_probability_manifest.json').write_text(json.dumps(merged,indent=2,ensure_ascii=False),encoding='utf-8')
    print(json.dumps({'merged_targets':TARGET_LABELS,'out_prob':str(outp),'out_panels':str(outpan)},indent=2))

if __name__=='__main__': main()
