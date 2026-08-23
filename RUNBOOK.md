# Runbook

How to run every part of this project from a clean checkout, what each command
actually costs, and what to do when something is wrong.

Written for the person picking this up cold, which after a week is the author.

## Setup

```bash
uv sync                  # serving dependencies only
uv sync --extra ingest   # adds docling, sentence-transformers, pypdf
```

The split is deliberate. Ingestion pulls torch, torchvision and transformers, several
gigabytes and minutes of build time, and the deployed service never converts a PDF:
it answers questions against chunks that were written to disk on a laptop. A Render
free-tier build should not install a machine learning toolchain to serve a search
endpoint.

Copy `.env.example` to `.env`. **This service holds no provider credential.** Every
model call goes through Tollgate, so the only thing to set here is where Tollgate is:

```
TOLLGATE_URL=http://127.0.0.1:8077/v1
DASTAVEZ_MODEL=groq/openai/gpt-oss-120b
```

## The corpus

```bash
uv run python corpus/discover.py --out corpus/candidates.tsv   # find candidate PDFs
uv run python corpus/fetch.py                                  # download and hash
uv run python corpus/fetch.py --verify                         # re-hash, report drift
```

`discover.py` crawls the scheme sites and confirms each link really returns a PDF by
checking for `%PDF` magic bytes. A 200 that returns HTML is the failure that quietly
poisons a corpus, because it writes a login page to disk under a `.pdf` name and the
damage surfaces much later as a converter producing nothing.

It produces **candidates, not corpus**. Choosing which documents belong is judgement:
"is this the operational guideline or an awards ceremony flyer" is not something a
link extractor answers. Curate into `corpus/sources.yaml` by hand.

`fetch.py --verify` exits non-zero on drift and is the reproduction check. Anyone
should be able to rebuild the corpus and get the same hashes.

**Expect links to rot.** A hand-assembled list resolved one time in five. That rate is
a property of this document class and it is why the manifest carries a retrieval date
and a content hash per document.

## Ingestion

```bash
uv run python -m dastavez.ingest.run --list-matrix
uv run python -m dastavez.ingest.run --document pmkisan-og-revised-en
uv run python -m dastavez.ingest.run --converter docling-routed --splitter section
```

Four axes: `--converter`, `--cleaning`, `--splitter`, `--metadata`. 32 configurations,
4 conversion passes, because only the converter axis re-opens the PDF.

### What it costs, measured on this machine

| | |
|---|---|
| Docling, warm, OCR on | 31.0 s per page |
| Docling, warm, `do_ocr=False` | 2.4 s per page |
| pypdf | milliseconds |
| Corpus | 353 pages, 22 documents with a text layer, 8 image-only |
| One routed conversion pass | about 0.39 h |
| M6, four converter arms | about 1.55 h |

Conversion output is cached under `corpus/.converted/`, keyed by document content hash
plus converter identity and version. **Do not delete it casually.** Without the cache
the M6 matrix is 32 configurations times three runs times a conversion pass, which
does not happen in a day. Use `--refresh` to force one document through again.

## Embedding and asking

```bash
uv run python -m dastavez.index_build --run <run_id>
```

`<run_id>` comes from the ingest output, for example
`pypdf.none.fixed.none.a5037433bfc4`.

The embedding model is multilingual (`intfloat/multilingual-e5-small`) and that is not
a preference: an English-only model would embed Devanagari into noise, the English
slice would score well, and the Hindi slice would be quietly worthless behind an
acceptable-looking aggregate.

**Call `Embedder.warm()` before timing anything.** The model load is 51 seconds against
a warm retrieval median of 127 milliseconds. Without warming, the first query reports
the load as retrieval latency, which is how a sibling project once published 1184 ms
against a provider's own 559 ms.

To answer a question, Tollgate must be running:

```bash
cd ../tollgate && uv run python -m uvicorn app.main:app --host 127.0.0.1 --port 8077
```

## Tests and gates

```bash
uv run pytest
uv run ruff check .
uv run python scripts/check_conventions.py
```

The conventions check is a CI step, not advice. It blocks em dashes, emoji, AI
attribution and live credential shapes, and prints "solves / blocks / prevents" for a
human to look at rather than guessing whether a sentence is a claim.

`tests/conftest.py` pins `TOLLGATE_URL` at a dead port before anything imports the
app, so no test can reach a live gateway by accident.

## When something is wrong

**A converter produces zero chunks.** Expected for `pypdf` against an image-only
document, and it is the correct outcome rather than a bug: those 8 documents are the
population the OCR-cascade study is measured over. Check `has_text_layer` in the
manifest before investigating.

**Retrieval takes a minute.** The embedder was not warmed. See above.

**`stages.sha256` mismatch.** `bench/stages.md` changed. That file defines what every
published latency covers, so the mismatch is asking whether every published number was
re-run. Re-hash only after answering that.

**A span raises `UndeclaredAttribute`.** Only in tests, where the contract is strict.
Production drops the attribute and logs it, because instrumentation that throws turns
a typo into an outage. Add the attribute to `CONTRACT` in `app/spans.py`, and if it
could carry document text, do not.

**Tollgate returns 502 saying a key is unset while `.env` sets it.** Fixed, but worth
knowing: `Settings` reads `.env`, while provider keys are resolved with
`os.environ.get(api_key_env)` because the variable name is configuration. Tollgate now
loads `.env` into the environment at startup.

**Groq returns 404 for a model.** The catalogue moves. Check the live `/v1/models`
listing rather than a docs page, and update `last_verified` in `quotas.yaml` when you
do. `llama-3.3-70b-versatile` was removed and every fork inherited it.

## The numbers this project will publish, and what they exclude

Retrieval latency covers `dastavez.retrieve` including the query encode, excluding
the model load and excluding rerank. Answer latency is reported separately and never
summed with retrieval: retrieval is local and CPU-bound, answering is a network call
to a free tier whose tail this project does not control, and adding them would let
provider variance swallow every retrieval improvement.

Every accuracy number is published beside the eval set's own estimated label error
rate. A headline accuracy computed against unaudited labels measures the system and
its labels together with no way to tell which moved.
