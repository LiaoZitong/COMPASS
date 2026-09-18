# Figure reproduction

The package retains the five base main-figure scripts and adds the R1 replacement scripts for Figures 2, 3, and 5; SI Figures S3, S4, S7, and S8; and Table S7. Generated outputs use SVG, PDF, 600-dpi TIFF, and 300-dpi PNG formats with source-data CSVs and manifests.

| Figure | Public script | Main frozen inputs |
|---|---|---|
| 1 | `code/figures/figure_01/plot_figure_01.py` | toxicity, chemistry, probability, panel, and framework result tables |
| 2 | `code/figures/figure_02/plot_figure_02.py` | fixed cross-target curves, probability matrices, weights, comparator definitions, complementarity |
| 3 | `code/figures/figure_03/plot_figure_03.py` | Top5-apical/reference pairs and chemical follow-up ranking outputs |
| 4 | `code/figures/figure_04/plot_figure_04.py` | warning/reference and same-reference warning–apical results |
| 5 | `code/figures/figure_05/plot_figure_05.py` | localized state sequences/comparisons, testing priorities, Census boundaries |

Run `code/figures/render_all.py` after stages 01–22 and after downloading the Census archive required by Figure 5. Then run the complete R1 analysis and:

```powershell
.\.venv\Scripts\python.exe .\code\revision_r1\figures\render_revision_figures.py
```

The R1 Figure 2 renderer reuses the seeded feasible-panel membership samples created by the base Figure 2 script and rescores them with strict-LOO probabilities. SI Figure S3 additionally requires the two base SI source tables documented in its script; these are derived full-workflow inputs and are not bundled. Figures 3, 5, S4, S7, and S8 use the R1 and documented base result trees directly. The source production used Python/Matplotlib; no manual edit changes numerical content. DOCX/SI assembly remains manuscript-specific and is not part of this code release.
