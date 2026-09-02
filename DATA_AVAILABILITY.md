# Data and code availability

This file is a manuscript-ready template. Replace bracketed fields only after the GitHub
repository and any archival dataset record have been created. Do not invent an accession number,
DOI, licence, embargo date or access committee.

## Data Availability

The MAGE source code, extraction prompts, graph schema, configuration template and minimal
non-sensitive example fixtures are available at https://github.com/Tai609/MAGE under release
[tag or commit]. The repository does not include article PDFs, publisher-controlled figures, private
uploads, API requests/responses, generated model traces, or credentials. These materials are
excluded because their redistribution may be restricted by copyright, licence, privacy or service
terms. The datasets supporting the quantitative results are deposited in [archival repository]
under [DOI/accession]. The deposit contains [raw/processed/source data, figure source data,
metadata and manifests] and is versioned together with the code release.

Publicly reusable third-party datasets are identified by their original repository, version or
release date, and persistent identifier in the accompanying dataset manifest. Where raw article
content cannot be redistributed, the repository provides the permitted derived data and metadata
needed to trace each result to its source, together with the applicable licence or access route.

## Code Availability

The MAGE implementation used for the reported pipeline is available at
https://github.com/Tai609/MAGE under the MIT License and release [tag or commit]. The environment template contains variable
names only; API keys, Neo4j passwords, tunnel tokens and other secrets are supplied by each user
through a private `.env` file or process environment and are never stored in the repository.

## Repository checklist before submission

- [ ] Replace the release tag/commit and archival repository identifier.
- [ ] Deposit figure/source data and processed graph outputs needed to verify the main claims.
- [ ] Add a machine-readable manifest mapping files to figures, tables and processing scripts.
- [ ] Record the data/code licence and any third-party restrictions.
- [ ] Cite reused datasets with repository, version/release date and persistent identifier.
- [ ] Provide a reviewer link or controlled-access procedure if a dataset cannot yet be public.
- [ ] Run the secret scan in `SECURITY.md` and verify that no `.env`, token, password or private
  absolute path appears in the release.
