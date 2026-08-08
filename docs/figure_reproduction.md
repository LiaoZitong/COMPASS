# Figure reproduction

The package includes the five current main-figure scripts and their shared Matplotlib helpers. Generated outputs are written under `outputs/figures/figure_01` through `figure_05` in SVG, PDF, 600-dpi TIFF, and 300-dpi PNG formats, with source-data CSVs and figure manifests.

| Figure | Public script | Main frozen inputs |
|---|---|---|
| 1 | `code/figures/figure_01/plot_figure_01.py` | toxicity, chemistry, probability, panel, and framework result tables |
| 2 | `code/figures/figure_02/plot_figure_02.py` | fixed cross-target curves, probability matrices, weights, comparator definitions, complementarity |
| 3 | `code/figures/figure_03/plot_figure_03.py` | Top5-apical/reference pairs and chemical follow-up ranking outputs |
| 4 | `code/figures/figure_04/plot_figure_04.py` | warning/reference and same-reference warning–apical results |
| 5 | `code/figures/figure_05/plot_figure_05.py` | localized state sequences/comparisons, testing priorities, Census boundaries |

Run `code/figures/render_all.py` after stages 01–22 and after downloading the Census archive required by Figure 5. The source production used Python/Matplotlib; no manual edit changes numerical content. Current DOCX/SI assembly and SI-figure layout remain manuscript-specific and are not part of this core release.

