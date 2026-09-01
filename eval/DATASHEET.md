# Dastavez eval set, v1

150 question-answer pairs over Indian central scheme documents, with page-anchored
citations. Published as a versioned artifact: the content hash in
`eval/questions.sha256` is what a published number is measured against, and it
changes when a question, an answer or a citation changes.

**Content hash:** `88cfeb9aeaacf140befaa0f3a0b69a9326b4b341b2a0050e4150e5443eb084fa`
**Built:** 2026-08-31, by `scripts/build_eval_set.py`
**Corpus snapshot:** `corpus/manifest.jsonl`, 30 documents, 353 pages, retrieved
2026-08-22, chunk run `pypdf.strip.fixed.none.d5c4ff3b99cd`

## Who wrote it

Drafted by `claude-opus-5` on 2026-08-31, recorded as `annotator_ids` on every
row. **These are drafts, not adjudicated labels.** No human has verified them yet.

That distinction is the whole reason this file says so in its second paragraph.
The two text-to-SQL benchmarks this project's brief cites measured annotation
error rates of 52.8% and 62.8%, and re-evaluating leaderboard agents against
corrected labels moved them by up to nine ranks. A model-authored gold answer is
exactly the kind of label that looks right and is not, and a set that hides its
provenance invites the reader to trust it more than it deserves.

M4.3 and M4.4 are the audit that measures how wrong this is: a second annotator of
a different model family on a 30-question subset, hand adjudication of the
disagreements, then Cohen's kappa with its Landis-Koch band and an estimated label
error rate for the full 150. **Until that runs, no accuracy number measured on
this set means anything**, because the error bar on the labels is unknown and
could be larger than any difference between two pipeline configurations.

## What is in it

| Type | Count | What it tests |
|---|---|---|
| `factual` | 30 | A single fact on a single page |
| `exclusion` | 25 | Why a person does not qualify |
| `eligibility_multihop` | 25 | A criterion and an exclusion from different documents |
| `procedural` | 25 | How to apply, what documents are needed |
| `table_lookup` | 18 | The answer is a cell |
| `unanswerable` | 27 | The corpus genuinely does not say |

| Language | Count |
|---|---|
| `en` | 96 |
| `hinglish` | 32 |
| `hi` | 22 |

Difficulty is tagged `easy`, `medium` or `hard`, and is the drafter's judgement
rather than a measured property. It is worth re-deriving from measured accuracy
once the set has been run, and worth discarding if the two disagree.

### The unanswerable slice

Twenty-seven questions the corpus cannot answer, each carrying a `note` recording
what was searched for and not found. The note is the evidence that the absence was
checked rather than assumed, and it is what a reviewer disputes if the drafter
missed a page that does answer the question.

Refusal correctness cannot be measured without these. A system that never refuses
scores perfectly on a set that contains none, which is how a RAG evaluation
reports a number that has nothing to do with whether the system is safe to deploy.

## Scheme and language coverage

Four schemes, unevenly weighted, because the corpus is unevenly weighted:

- **PM-KISAN**, 10 documents. The richest source: eligibility, the 01.02.2019
  cut-off, six exclusion categories, succession, and the 2019 revision that
  removed the two-hectare ceiling.
- **PM-KMY**, 3 documents. The contribution table by entry age, which is where
  most `table_lookup` questions come from.
- **Ujjwala**, 5 documents. The thirteen-criterion deprivation checklist.
- **PMAY-U**, 6 documents. Under-represented: it contributes the largest share of
  corpus pages and almost none of the questions.

**The question's language is not the document's language.** Every source document
that extracted cleanly is in English, so a Hindi or Hinglish question here is a
question a real user would ask against an English page. That is the harder
retrieval case and it is deliberate.

## What this does not cover

- **PMAY-U is under-sampled.** It is 164 of the 286 pages with extractable text
  and fewer than ten questions. A number on this set is not a number on PMAY-U.
- **No Devanagari source pages are cited.** The Hindi pages in the Ujjwala corpus
  extracted as almost nothing under pypdf, which is the OCR-cascade failure this
  project studies elsewhere. Questions whose answers live only on those pages
  cannot be scored until the OCR path lands, so none were written.
- **One drafter, one sitting.** The set reflects one view of what matters in these
  documents. Questions a beneficiary would actually ask, phrased the way they
  would phrase them, are not represented; these are phrased the way someone who
  has read the guidelines would phrase them.
- **No adversarial or ambiguous questions.** Nothing here is deliberately
  misleading, and real queries frequently are.
- **The corpus is a snapshot.** Scheme rules change. Every answer is correct
  against documents retrieved on 2026-08-22 and may be wrong about the scheme as
  it stands today. The PM-KISAN two-hectare ceiling is already an example: the
  original FAQ and the revised FAQ disagree, and both are in the corpus.

## Reproducing

```
uv run python scripts/build_eval_set.py --check   # validate, write nothing
uv run python scripts/build_eval_set.py           # rebuild and rehash
```

The build refuses to emit a set that breaks a structural rule or cites a page that
is not in the corpus, so a question invented rather than read fails at build time
rather than at judging time.
