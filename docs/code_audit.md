# Public-code audit

Audit date: 2026-08-08. Source analysis: frozen COMPASS/BHBT v16.5.

## Project overview

The source workspace is a Windows-native Python 3.11 analysis with 24 numbered semantic stages. Stages 01–22 generate censored toxicity cells, annotations, calibrated measured-tail probabilities, national and regional panels, framework comparisons, warning analyses, testing priorities, and uncertainty/sensitivity results. Stage 23 renders manuscript/SI assets and assembles Word outputs; stage 24 performs project- and manuscript-specific release validation.

The source workspace contains an empty `.git` directory but no valid Git repository, branch, remote, or commit history. The public freeze therefore uses file manifests, timestamps, configuration records, and SHA-256 checksums as the equivalent version identifier.

## Main analysis and source mapping

- Canonical runner: source `src/run_pipeline.py`; public copy `code/run_analysis.py`.
- Frozen national Top-20 sequence: stage 13; complete 2,144-species explorer continuation: `code/export_site_data.py` using the same deterministic lower-5% objective.
- Random/comparator baselines: stages 14 and 19 plus Figure 2 sampling code.
- Taxonomic complementarity: stage 15.
- Top5-apical and warning-to-HC5 bridge: stage 16.
- Warning timing/concentration evidence: stage 17.
- Mechanism evidence and testing gaps: stage 18.
- Regional localization and national/state testing priorities: stage 20.
- Profile-likelihood and MNAR analyses: stages 21–22.
- Main Figures 1–5: five scripts under the source `manuscript_plot/` tree.
- SI Figures S1–S7: source `src/build_si_figures.py`; excluded from this core-code package because it is tightly coupled to manuscript/SI assembly and the requested public scope is core code only.

## Dependencies and workflow constraints

The numerical stack is NumPy, pandas, SciPy, scikit-learn, and Matplotlib; Figure 5 also uses pyshp. The production environment is CPython 3.11.9. Execution order and random settings are explicit. The source pipeline uses relative project-root paths and isolated subprocesses for target-specific probability passes.

The core workflow has no spreadsheet-reading code and no manually copied numerical input. Manuscript DOCX generation uses author-controlled Word templates and layout fields; those files and scripts are outside the public package. Main figure geometry is generated with Python/Matplotlib. Figure 5 requires an external Census boundary archive.

## Content selected for the public package

- core semantic stages 01–22;
- a path-clean public runner;
- the five current main-figure scripts and their shared helpers;
- dependency and random-setting locks;
- input and output contracts;
- result and cross-package validators;
- a deterministic one-way site-data exporter;
- data-free tests and release checks.

## Content deliberately excluded

- all raw, intermediate, processed, reference, and result data;
- source acquisition/ETL utilities and notebooks;
- Word/DOCX/SI assembly, templates, submission materials, and reference libraries;
- internal review packages, prompts, discussions, and revision records;
- historical runs, archives, caches, logs, and superseded outputs;
- stage 24, which contains a machine-specific LibreOffice path and manuscript-layout checks;
- source files with machine-specific author-final document paths;
- the Census boundary archive and all other third-party archives.

## Reproducibility risks and controls

1. No data are distributed, so a fresh checkout cannot reproduce manuscript numbers without separately obtained inputs. The README states this boundary prominently.
2. The source project lacks a valid Git commit. `outputs/source_manifest_sha256.csv` and `outputs/run_metadata.json` provide the equivalent freeze record.
3. A historical build summary reports v16.4 while the current panel and validation manifests report v16.5. The public package uses the current v16.5 outputs and excludes the stale build summary.
4. Historical directories contain multiple `final`, `latest`, and revision-labelled files. The authoritative source is the documented `manuscript/current_submission/` entry point; historical and review trees are excluded.
5. Figure 5 depends on a third-party boundary file. Its URL and expected local placement are explicit.
6. Public release still requires author decisions on license, repository name/visibility, final citation, and DOI.

## Minimum organization applied

The source workspace was not changed. Selected files were copied to two sibling packages. Public-copy changes are limited to package-relative paths, code-only execution modes, validation/export utilities, and documentation. No scientific method, parameter, result, or source output was edited.

## Items requiring author confirmation

- public code license;
- GitHub repository name and whether it should remain private or become public;
- final repository URL, release tag, and Zenodo DOI;
- corresponding contact and any ORCID identifiers for citation metadata;
- whether a separate licensed data deposit will later accompany the code;
- whether tightly coupled SI-figure assembly code should be added after submission.
