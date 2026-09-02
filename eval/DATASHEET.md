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
row, and audited on 2026-09-02 by a second annotator of a different model family.
**No human has read all 150.** The audit found no label error in the 30 it
checked, and the adjudication of its six disagreements was performed by the same
party that drafted the set, which is a real weakening: an independent adjudicator
might resolve some of them the other way.

That distinction is the whole reason this file says so in its second paragraph.
The two text-to-SQL benchmarks this project's brief cites measured annotation
error rates of 52.8% and 62.8%, and re-evaluating leaderboard agents against
corrected labels moved them by up to nine ranks. A model-authored gold answer is
exactly the kind of label that looks right and is not, and a set that hides its
provenance invites the reader to trust it more than it deserves.

M4.3 and M4.4 are that audit, and they have now run. The results are in the next
section. The short version is that the audit found no label error and instead
found something else, which is the usual way of these things.

What the audit does not license is treating the labels as verified. Thirty of 150
were checked by a second annotator and none of the remaining 120 were, so the
error bar on this set is "no errors seen in a fifth of it" rather than "no
errors". A pipeline difference smaller than that uncertainty is not a difference.

## The label audit, 2026-09-02

**Second annotator:** `gemini-3.6-flash`, a different model family from the
drafter, on a seeded stratified subset of 30 of the 150.

| | |
|---|---|
| Raw agreement | 80.0% |
| Chance agreement | 58.9% |
| **Cohen's kappa** | **0.514**, moderate on the Landis-Koch bands |
| Disagreements | 6 of 30 |

**The label is answerability, not the gold answer.** Showing the second annotator
the gold answer and asking whether the page supports it produces no usable kappa:
the drafter's side of the table has no variance, so chance agreement is total and
the statistic collapses. Answerability is assigned with real variance by both
annotators, and it is the label that matters most, since an unanswerable question
that is actually answerable poisons the refusal slice.

### Every one of the six disagreements was a retrieval miss, not a label error

Adjudicated by opening each cited page and checking it contains the claim. **All
six citations hold.** The second annotator said "not answerable" because BM25 had
handed it the wrong passages, not because the gold answer was wrong.

So the audited label error rate is **0 of 30**, and the 20% disagreement rate
measures the retrieval layer rather than the labels. Reporting it as a label error
rate, which is what the number looks like at first glance, would have been wrong
in the direction that flatters nobody: it would understate the eval set and
mislead about where the problem is.

### The retrieval miss has a language

| Question language | Retrieval misses on the audited 30 |
|---|---|
| `hi` | 3 of 5 (60%) |
| `en` | 3 of 20 (15%) |
| `hinglish` | 0 of 5 (0%) |

BM25 misses Devanagari questions four times more often than English ones. Five
Hindi questions is a small sample and the rate is not a precise number, but the
direction matches what this corpus already shows elsewhere: pypdf recovers almost
nothing from Devanagari pages, so the lexical index has little to match against.
This is the OCR-cascade problem arriving in the retrieval layer.

**Reproduce:** `uv run python scripts/second_annotator.py --n 30 --report`

## Citation support, checked mechanically across all 150

`scripts/check_citations.py` checks whether the numbers a gold answer asserts
appear on the page it cites. Of the 123 answerable questions, 46 carry a checkable
numeric claim and **45 of those hold**.

The one failure is `dz-064`, whose answer states a difference of Rs. 145 between
entry ages 40 and 18. The page states Rs. 200 and Rs. 55; the difference is
arithmetic the reader performs. It is left flagged rather than special-cased,
because a citation check that silently accepts derived numbers accepts a great
deal else besides.

The first version of this check reported eight failures. Five were the date
`1.6.2019`, which `pmkisan-faq-revised` page 1 carries as "1 .6.2019" with a space
that pypdf inserted, and two were numbers the question supplied rather than claims
the page had to support. Both are now handled, and both are the reason a checker
built over damaged extractions has to be told what damage looks like.

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
