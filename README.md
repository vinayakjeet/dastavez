# Dastavez

Agentic RAG over messy Indian government scheme documents, with a page-anchored
citation on every answer and an open eval set whose own label error rate is
published beside every accuracy number.

## Problem

Scheme eligibility rules live in PDFs that are scanned, bilingual, full of
tables, and revised by circular. Being approximately right is being wrong: the
exception clause is the whole answer, and a citation that points at the wrong
page is worse than none, because it looks like proof.

This project answers questions over that corpus and refuses when retrieval
cannot support an answer. Its contribution is not the retriever. It is the eval
set: 150 questions with gold pages, a measured label error rate, and a
replication of the Portuguese admin-docs pipeline study (naive loader floor,
hand-curated ceiling, converter against chunking) on Indic documents, which
nobody has published.

## Architecture

Two trees, instrumented from the first commit with Spanlight spans whose
attributes are declared in `app/spans.py` and specified in `bench/stages.md`
(hashed, so boundaries cannot move after a result is seen).

Ingestion is offline: `corpus/fetch.py` pulls the corpus with provenance,
`dastavez/ingest/` converts, cleans, splits and enriches under a hashed
`PipelineConfig` (`dastavez/pipeline.py`), and chunks land in SQLite through
`dastavez/store.py`. The served API reads Postgres with pgvector behind the
same `ChunkStore` interface; ablation runs stay on SQLite because they rewrite
the whole corpus repeatedly and should not compete with the API for quota.

Four converters sit behind one interface: pypdf (the floor), Docling (OCR on,
off, and routed), Marker and MinerU. Marker and MinerU convert from their own
environments into a shared block cache keyed by content hash and converter
version (`dastavez/offline_convert.py`), because their torch pins conflict with
Docling's and the served service should never carry a model stack. Block kinds
are normalised onto one vocabulary so the section-respecting splitter means the
same thing for every arm.

Retrieval is hybrid: dense multilingual-e5 embeddings (`dastavez/retrieval.py`)
and BM25 that keeps hyphens and slashes (`dastavez/lexical.py`), fused in
`dastavez/hybrid.py`. Answers quote their pages or refuse; the refusal says
what was searched. Every model call goes through Tollgate, which holds the only
provider credentials in the portfolio.

## Benchmarks

Regenerate with `uv run python bench/status.py` (summary) and
`uv run python bench/parsing_quality.py --converter <name>` (per page).

Corpus: 30 documents, 353 pages, four schemes, 12 image-only. `corpus/fetch.py
--verify` re-downloads and checks every hash.

Gold ceiling: 24 hand-curated pages (`corpus/gold/`), 8 with tables, 6 with
Hindi, each hashed in `corpus/gold/index.jsonl`.

Parsing quality against gold, mean over covered pages (definitions and their
sources in `dastavez/parsing.py`; these locate content and structure loss and
do not predict retrieval or answer quality):

| converter | pages | text similarity | table TEDS |
|---|---|---|---|
| pypdf | 24 | 0.588 | 0.000 |
| docling-noocr | 24 | 0.537 | 0.433 |
| docling (OCR, partial coverage) | 6 | 0.786 | 0.878 |
| marker (3 pages) | 3 | 0.885 | n/a |
| mineru (3 pages) | 3 | 0.923 | n/a |

The shape to read: pypdf keeps prose and has no tables at all; docling-noocr
loses every scanned page and reads the PM-KMY 23-row contribution chart at
TEDS 1.000. The KYC form page defeats both (0.037 and 0.245). Marker, mineru
and the OCR arm have full-gold numbers pending their M6 conversion passes.

Retrieval: on the reference question, "What is the amount of income support
paid to each farmer family per year?", the answer quotes page 2 of
pmkisan-og-revised-en, which reads "an amount of Rs.6000/- per year is released
by the Central Government". Eight of nine probe queries disagree between BM25
and dense on the top result (`bench/retriever_comparison.py`), which is the
case for hybrid rather than a result about accuracy.

## Technical Decisions

See [DECISIONS.md](DECISIONS.md). The short version: block kinds are a shared
vocabulary, not converter-native labels; heavy converters run from their own
environments against a shared cache; SQLite for ablation and Postgres for
serving, behind one interface; conversion output is cached because Docling
measured 31 seconds per page warm and the ablation matrix multiplies that.

## What Broke

See [LEARNING.md](LEARNING.md) for the running log. Highlights:

- Every table in the corpus was silently invisible for the project's first two
  milestones: Docling's `TableItem` carries no `.text`, so the converter loop
  skipped all of them without an error. Found while wiring the splitting axis,
  when the table-protecting splitter turned out to have never seen a table.
- The corpus's own text layers lie on the hard pages: "Frequentl v Asked
  Questions", "SI\4F", "Par culars" for "Particulars". Gold pages are
  transcribed from images, not layers.
- Marker 2.0 autodetects its inference backend from the GPU's existence rather
  than its usability, then fails three layers down asking for Docker. On a
  CUDA-less torch install: `SURYA_INFERENCE_BACKEND=llamacpp` plus a stock
  `llama-server` binary.

## Run It

Requires [uv](https://docs.astral.sh/uv/). No Docker needed for local use.

```
uv sync --extra ingest
uv run python corpus/fetch.py --verify
uv run pytest
uv run ruff check .
```

Ask a question:

```
uv run uvicorn app.main:app --reload
curl -X POST localhost:8000/demo/chat -H "Content-Type: application/json" \
  -d "{\"prompt\": \"What is the amount of income support paid to each farmer family per year?\"}"
```

Ingest under a named configuration:

```
uv run python -m dastavez.ingest.run --converter docling-routed --splitter section --metadata enriched
```

Parsing quality against gold, per converter per page:

```
uv run python bench/parsing_quality.py --converter docling-noocr
```

Using a real model goes through Tollgate: set `TOLLGATE_URL` and
`DASTAVEZ_MODEL` in `.env` (see `.env.example`, and [QUOTAS.md](QUOTAS.md) for
free-tier limits). The deployed URL is pending (BACKLOG M0.6); the service
definition in `render.yaml` is the reproducible path to it.
