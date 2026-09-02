# MAGE source data

This directory contains non-sensitive, derived source data that support figures in the MAGE
manuscript. Article PDFs, publisher-controlled figures, private uploads, provider responses,
credentials and model traces are not included.

## Figure 2

- Workbook: [`figure_2/MAGE_Figure2_Source_Data.xlsx`](figure_2/MAGE_Figure2_Source_Data.xlsx)
- Machine-readable manifest: [`MANIFEST.csv`](MANIFEST.csv)

The workbook contains eight worksheets covering all five panels:

- panel a: summary statistics for six image-text alignment methods and 120 per-output records;
- panel b: the normalized 9-model Image/Text/KG/Overall heatmap matrix;
- panels c-e: 180 per-output quality, processing-time, cost and token records plus model summaries;
- documentation: a README worksheet, data dictionary and provenance/checksum table.

Formula-derived F1, Overall, mean and sample-standard-deviation fields are retained for audit.
The panel-a alignment benchmark and the panels-b-e shared-model benchmark each contain 20 article
outputs, but they use different output-ID sets; both sets are listed explicitly in the workbook.
The values were recovered from MAGE result tables and were not digitized from the rendered figure.

## Original analysis data package

- Data directory: [`original_data/`](original_data/)
- Per-file SHA-256 manifest: [`original_data_manifest.csv`](original_data_manifest.csv)
- Deposit summary: [`original_data_summary.md`](original_data_summary.md)

This package contains 1,549 files (323,807,680 bytes): 1,539 fused MAGE-Graph JSON outputs,
one four-way morphology-evaluation JSON file and nine CSV analysis tables. The CSV files cover
morphology-model training data, out-of-fold predictions, classification metrics, SHAP summaries,
fold audits and point-level SHAP values. All JSON and CSV files were syntax-checked after copying.

For public release, local absolute CATDA paths embedded in text fields were converted to
repository-relative paths pointing to the deposited graph copies. File count, relative directory structure, row order, data values and
JSON graph structure were otherwise preserved. This repository does not include the source article
PDFs or publisher-controlled images.

A separate data licence has not yet been assigned. Repository maintainers should state the data
licence before journal submission or archival deposition.
