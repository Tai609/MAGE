# Data and code availability

## Data Availability

The derived source data supporting Figure 2 of the MAGE manuscript are publicly available in this
repository at
[`source_data/figure_2/MAGE_Figure2_Source_Data.xlsx`](source_data/figure_2/MAGE_Figure2_Source_Data.xlsx).
The workbook contains the values underlying all five panels: pair-level image-text alignment
results, model-level Image/Text/KG quality metrics, per-article processing times, estimated costs,
token usage and Overall scores. It also includes audit formulas, variable definitions and a
provenance table. A machine-readable file manifest with a SHA-256 checksum is available at
[`source_data/MANIFEST.csv`](source_data/MANIFEST.csv).

The deposited workbook contains derived benchmark measurements and does not redistribute article
PDFs, publisher-controlled figures, private uploads, API requests or responses, generated model
traces, or credentials. These materials are excluded because redistribution may be restricted by
copyright, licence, privacy or service terms. Publicly reusable third-party datasets should be
identified by their original repository, version or release date, and persistent identifier in
future dataset manifests.

No archival DOI or accession number is claimed for this GitHub deposit. If an immutable archival
record is minted later, its identifier and version should be added here before manuscript
submission.

## Code Availability

The MAGE implementation, extraction prompts, graph schema, configuration template and minimal
non-sensitive examples are available at https://github.com/Tai609/MAGE under the MIT License. API
keys, Neo4j passwords, tunnel tokens and other secrets are supplied through a private `.env` file
or process environment and are not stored in the repository.

## Repository checklist before submission

- [x] Deposit the Figure 2 source data needed to verify the plotted results.
- [x] Add a machine-readable manifest mapping the deposited workbook to the figure.
- [ ] Assign and record a data licence.
- [ ] Add an immutable release tag or archival DOI/accession, if required by the target journal.
- [ ] Deposit any additional source data needed for other manuscript figures and tables.
- [ ] Cite reused datasets with their repository, version/release date and persistent identifier.
- [ ] Provide a reviewer link or controlled-access procedure for any non-public dataset.
- [x] Run the secret scan and inspect the release diff for this Figure 2 data deposit.
