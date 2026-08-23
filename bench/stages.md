# Stage boundaries

What every span in Dastavez starts and ends at, and what each published latency
number covers.

This file is hashed into `stages.sha256` and the hash is checked by the bench
harness. The reason is not ceremony. A sibling project published a session latency of
1184ms against a provider's own 559ms, because its span wrapped a concurrency
semaphore and half the figure was queueing reported as work. Retrieval latency is a
published number here, and a measurement whose boundaries can move after a result is
seen is not a measurement.

Changing this file is allowed. Changing it without re-running every published number
is not, which is what the hash check enforces.

## Two trees, not one

Ingestion and question answering are separate jobs with separate lifetimes, and
forcing them under one root would produce a trace nobody can read.

```
dastavez.ingest                     one per document, per configuration, offline
  dastavez.ingest.convert
  dastavez.ingest.chunk
  dastavez.ingest.embed

dastavez.question                   one per question asked, online
  dastavez.decompose
  dastavez.retrieve
    dastavez.retrieve.dense
    dastavez.retrieve.lexical
    dastavez.retrieve.fuse
  dastavez.rerank
  dastavez.answer
```

## Boundaries

| Span | Starts at | Ends at |
|---|---|---|
| `dastavez.ingest` | Before the document is opened | After its chunks and embeddings are committed |
| `dastavez.ingest.convert` | Before the converter is handed the path, or before the cache is read | After blocks are returned, from either source |
| `dastavez.ingest.chunk` | Before the first block is divided | After the last chunk is constructed |
| `dastavez.ingest.embed` | Before the first text is encoded | After the vectors are committed |
| `dastavez.question` | The first line of the handler, after the question is parsed | Immediately before the answer or refusal is returned |
| `dastavez.decompose` | Before the question is examined for multi-hop structure | After sub-questions are produced, or after it is decided there are none |
| `dastavez.retrieve` | Before the first retriever is called | After fused, ordered candidates exist |
| `dastavez.retrieve.dense` | Before the query is encoded | After the nearest-neighbour scan returns |
| `dastavez.retrieve.lexical` | Before the BM25 query is issued | After it returns |
| `dastavez.retrieve.fuse` | Before the two candidate lists are combined | After the fused ordering exists |
| `dastavez.rerank` | Before the cross-encoder is given its pairs | After the reordered list exists |
| `dastavez.answer` | Before the prompt is assembled | After the model's response is fully read, or immediately after a refusal is constructed |

### The query encode is inside `retrieve.dense`, the model load is not

Encoding the question is unavoidable cost of dense retrieval and a caller waits for
it on every question, so excluding it would report a retrieval latency no user
experiences.

Loading the embedding model is different and is excluded. It happens once per
process, and measured here it is 51 seconds against a warm retrieval median of 127
milliseconds. Counting it would report the first question as four hundred times
slower than every later one and would put a startup cost inside a per-question
percentile.

So `Embedder.warm()` is called at startup, the load duration is reported as its own
number the way a connection setup is, and any run where retrieval latency is
published must state that the embedder was warm. A cold first measurement is not
wrong, it is a different measurement, and the two are never averaged.

### A refusal still opens `dastavez.answer`

A refusal is an answer, not an error path, and it has a duration. Closing the tree
early on refusal would make refusals invisible in a latency distribution and would
make the mean look better than it is, since refusals are the fast case.

## What each published number covers

| Number | Covers |
|---|---|
| retrieval latency | `dastavez.retrieve`, including the query encode, excluding rerank |
| rerank latency | `dastavez.rerank` alone |
| answer latency | `dastavez.answer`, which is dominated by the model and is reported separately for that reason |
| end to end | `dastavez.question`, which is the only number a caller feels |

Retrieval and answer are never summed into one figure. Retrieval is local and
CPU-bound; answering is a network call to a free-tier provider whose tail is not this
project's to control. Adding them produces a number whose variance is dominated by
something the pipeline cannot influence, and improvements to retrieval would vanish
inside it.

## What never goes on a span

No question text, no answer text, no chunk text, no document text.

This service ingests documents from the public internet and puts their contents into
a prompt, which makes it the largest untrusted-input surface in the portfolio. Span
attributes are indexed by the backend and are the documented anti-pattern for prompt
content: size limits, plus text sitting somewhere queryable by anyone with dashboard
access.

Counts, scores, page numbers, document ids and language tags say what an operator
needs and cannot be read back into the content they describe.

## Ingestion is timed but not compared to serving

`dastavez.ingest.convert` will record durations in the thousands of seconds, because
Docling with OCR measured 31 seconds per page. That is offline batch cost and it
belongs in the ablation's cost column, never in a latency percentile beside
`dastavez.question`.
