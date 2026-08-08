# Methods-to-code map

| Stage | Public script | Frozen analytical role |
|---:|---|---|
| 01 | `01_prepare_censored_toxicity.py` | Build record-, study-, and model-cell censored likelihood inputs. |
| 02 | `02_audit_evidence_layers.py` | Verify protective, rapid-warning, and mechanism-support separation. |
| 03 | `03_build_chemical_annotations.py` | Build chemical form, trait, taxonomy, and coarse mechanism annotations. |
| 04 | `04_refine_moa_annotations.py` | Build MIE, route, chain, and functional-MOA candidates. |
| 05 | `05_build_national_weights.py` | Build national and state chemical-priority weights. |
| 06–07 | target estimator and merger | Estimate uncalibrated lower-20%, lower-10%, and lower-5% probabilities in isolated passes. |
| 08 | `08_validate_chemical_holdout.py` | Whole-chemical held-out calibration and ranking assessment. |
| 09 | `09_validate_moa_core_entry.py` | Apply the predeclared candidate-layer core-entry gate. |
| 10 | `10_prepare_hc5_core_calibration.py` | Lock the lower-5% core calibration parameters. |
| 11–12 | calibrated estimator and merger | Re-estimate and merge calibrated target probabilities. |
| 13 | `13_build_hc5_panels.py` | Select the fixed lower-5% national sequence and evaluate its prefixes across targets. |
| 14 | `14_build_random_baselines.py` | Generate random-panel capture baselines. |
| 15 | `15_quantify_taxonomic_complementarity.py` | Quantify taxonomic/guild complementarity and Top-5 contributions. |
| 16 | `16_evaluate_warning_hc5_bridge.py` | Compare Top5-apical and Top5-warning thresholds with measured protective reference HC5 contexts. |
| 17 | `17_assess_warning_apical_lead_potential.py` | Quantify warning timing/concentration evidence and support. |
| 18 | `18_evaluate_mechanism_evidence_and_testing.py` | Audit mechanism density and identify evidence gaps. |
| 19 | `19_compare_hc5_frameworks.py` | Compare panel subsets under context-specific framework implementations. |
| 20 | `20_build_state_panels_and_testing.py` | Build lower-5% localized state panels and national/state testing priorities. |
| 21 | `21_propagate_hc5_profile_uncertainty.py` | Propagate stated likelihood/working-SSD uncertainty. |
| 22 | `22_assess_mnar_sensitivity.py` | Evaluate unidentifiable MNAR sensitivity scenarios. |

`code/run_analysis.py` preserves this order. The static explorer reads only exported results and never calls these stages in the browser.

