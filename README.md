# MAGE: multimodal text–image extraction and graph alignment

MAGE is the public, reproducible release of a pipeline that turns a scientific article's
Markdown text and figures into a multimodal catalyst knowledge graph. The release contains
source code, prompts, schemas, a small non-sensitive example, and the configuration template
needed to reproduce the workflow without exposing provider credentials.

## Pipeline

```text
Markdown + images
        │
        ├── text LLM: synthesis, testing and characterization extraction
        ├── vision LLM: entities and evidence from each figure/image
        └── alignment: CLIP mutual-kNN candidates + LLM resolution (or LLM-only)
                         │
                         ▼
                 fused CatGraph JSON
```

The text and vision stages can use different models. The alignment model is configured
independently with `--alignment-model`; no model name or credential is hard-coded as a secret.

## Repository layout

- `extract_main.py` — single-file entry point for text, image and alignment stages.
- `run_extract_batch.py` — repeatable batch runner for `paper_<id>` directories.
- `tools/cat_graph/` — graph extraction, image extraction, aligners and evaluation utilities.
- `models/` — provider adapters and embedding helpers; credentials are read from environment
  variables only.
- `prompts/` — text/vision extraction and feature prompts.
- `PDF_TO_MD/` — optional PDF-to-Markdown preprocessing helpers (credentials are placeholders).
- `tools/neo4j/` — optional import utilities for a local Neo4j instance.
- `examples/` — minimal graph fixtures that contain no API responses or credentials.
- `docs/` — CatGraph schema and data-format documentation.
- `.env.example` — safe configuration template; copy it to `.env` locally.

Large article collections, uploaded files, model responses, generated graphs, logs and private
benchmark material are intentionally excluded from this public release. See
[`DATA_AVAILABILITY.md`](DATA_AVAILABILITY.md) for the manuscript-ready wording and the fields
that must be filled after the GitHub repository and any archival dataset deposit are created.

## Installation (Windows PowerShell)

```powershell
cd MAGE
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
Copy-Item .env.example .env
# Edit .env locally. Do not commit it.
```

The dependency file is intentionally pinned to the validated research environment. If a lighter
installation is needed, install only the packages used by the selected provider and stage, then
record the resulting versions in the repository release notes.

### Provider routing and credential safety

Without a custom base URL, each credential is sent only to its official provider endpoint:

- `OPENAI_API_KEY` → `api.openai.com`;
- `GOOGLE_API_KEY` → `generativelanguage.googleapis.com`;
- `DEEPSEEK_API_KEY` → `api.deepseek.com`.

To use APIYi or another OpenAI-compatible gateway, configure its base URL explicitly. APIYi
requires both `APIYI_BASE_URL` and `APIYI_API_KEY`; MAGE does not fall back to an official-provider
key for an APIYi endpoint. Provider-specific custom endpoints can be set with
`OPENAI_API_BASE`, `GOOGLE_API_BASE`, or `DEEPSEEK_API_BASE`.

### Optional PDF preprocessing

The main MAGE pipeline starts from Markdown and images. To enable the optional public Fitz
PDF-to-Markdown helper, install its additional dependency:

```powershell
pip install -r PDF_TO_MD\requirements.txt
```

Camelot table extraction is skipped with a warning when Camelot is unavailable. The legacy Paddle
adapter depends on a project-specific `image_parser.py` that is not part of this public release and
therefore fails with an explicit message instead of an unresolved import.

## Input convention

For one article, keep the Markdown and its images together:

```text
inputs/
└── paper_1/
    ├── images/
    │   ├── Fig1.png
    │   └── Fig2.jpg
    └── txt/
        └── full.md
```

When the input is a directory, MAGE looks for `full.md` by default. To process every Markdown
file below a directory, pass `--target-filename ""`. Image files are read from the output
directory's `images/` folder when present, otherwise from the article directory's `images/`
folder.

## Reproduce one article

Run from the MAGE directory. Replace the model identifiers with models available to you; the
corresponding environment variable must be set in your private `.env`.

```powershell
python extract_main.py ..\inputs\paper_1 `
  --output-dir ..\outputs\paper_1 `
  --mode extract `
  --processes 1 `
  --text-model openai_gpt-4o-mini `
  --vision-model openai_gpt-4o-mini `
  --alignment-model openai_gpt-4o-mini
```

To generate the optional ML dataset in the same run, use `--mode both` and provide a feature
specification (the HER example is included):

```powershell
python extract_main.py ..\inputs\paper_1 `
  --output-dir ..\outputs\paper_1 `
  --mode both `
  --feature-file prompts\features_to_extract_HER.txt
```

Expected graph artifacts are written below `<output-dir>/graph/`:

- `<paper>_text_raw.json` — text-only graph before image fusion;
- `<paper>_image_raw.json` — image-derived graph with source-image provenance;
- `<paper>_output.json` — fused multimodal graph;
- `<paper>_result.json` and usage metadata — run-level provenance under `metadata/`.

## Batch processing

```powershell
python run_extract_batch.py `
  --project-root . `
  --data-root ..\inputs `
  --output-root ..\outputs `
  --paper-ids 1 2 3 `
  --mode extract `
  --text-model openai_gpt-4o-mini `
  --vision-model openai_gpt-4o-mini `
  --alignment-model openai_gpt-4o-mini
```

Use `--stop-on-error` for strict CI-style runs. Keep generated output outside the repository (or
under an ignored directory) so that model responses and article-derived content are not committed
by accident.

## Alignment choices

The default alignment strategy is the hybrid mutual-kNN + LLM resolver in
`tools/cat_graph/aligner/knn/entity_aligner_knn.py`. The LLM-only resolver is available in
`tools/cat_graph/aligner/entity_aligner_llm.py`. Both preserve image provenance and emit an
alignment summary in the run metadata.

## Optional Neo4j import

Start a local Neo4j instance, set `NEO4J_URI`, `NEO4J_USER` and `NEO4J_PASSWORD` in the private
environment, then import a fused graph:

```powershell
python import_catgraph.py ..\outputs\paper_1\graph\full_output.json `
  --neo4j_uri $env:NEO4J_URI `
  --neo4j_user $env:NEO4J_USER `
  --neo4j_password $env:NEO4J_PASSWORD
```

Do not place Neo4j passwords in command history shared with collaborators or in issue reports.

## Tests

The minimal regression suite covers nested image JSON parsing, node/edge ID consistency, MIME
selection and provider credential routing:

```powershell
python -m unittest discover -s tests -v
python -m ruff check --select E9,F63,F7,F82 .
```

The same checks run automatically in GitHub Actions.

## Security and provenance

- Credentials are loaded only from environment variables or a private `.env` file.
- `.gitignore` blocks `.env`, local inputs/uploads, outputs, logs, caches and key material.
- Before publishing a release, run the secret scan described in
  [`SECURITY.md`](SECURITY.md) and inspect the Git diff for absolute local paths.
- Article PDFs, figures and model responses may be subject to publisher, licence or privacy
  restrictions. Deposit only data for which you have redistribution rights; otherwise publish
  metadata, derived statistics or a documented access route.

## Citation

Please cite the MAGE release together with the associated manuscript. Replace the placeholder
release tag and archival identifier in `DATA_AVAILABILITY.md` after the archival release is
minted. Repository: https://github.com/Tai609/MAGE.
