#!/usr/bin/env python3
from __future__ import annotations
import argparse,json
from pathlib import Path

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--input',required=True);ap.add_argument('--output',required=True);a=ap.parse_args()
    d=json.loads(Path(a.input).read_text(encoding='utf-8'));d['version']='v16.5';d['description']='Whole-chemical calibration retained for taxonomy/direct-evidence probabilities. MOA-conditioned groups are disabled unless they pass the simplified v16.5 core-entry gate.'
    for _,v in d.get('by_protection_target',{}).items():
        v['validated_moa_groups']=[]
        v['gate_rule']='No MOA-conditioned core borrowing by default. Functional-MOA or chemical-form evidence must pass results/validation/moa_core_entry_gate.csv requirements.'
    Path(a.output).write_text(json.dumps(d,indent=2,ensure_ascii=False),encoding='utf-8')
if __name__=='__main__':main()
