# Gold pages

The hand-curated ceiling. Every automated configuration in the M6 ablation is
measured against these transcriptions, so they are transcribed as the documents
should have been read, not as any converter happens to output them.

`index.jsonl` carries one row per page: document id, page number, scheme,
language, whether the page carries a table, the file path, and the sha256 of the
transcription. `corpus/fetch.py --verify` re-verifies the corpus; a future
`verify` pass over this directory re-verifies the gold set the same way.

## What is in the set

24 pages across all four schemes with documents in the corpus: pm-kisan (7),
pm-kmy (4), pmay-u (5), ujjwala (8). Six pages carry Hindi content: the four
image-only Ujjwala scans plus the CLAP manual and reform booklet covers, whose
content is Devanagari. Eight pages carry tables, including the PM-KMY
contribution chart (23 data rows), the AHP table of contents, the Tata AIG
policy schedule, and the Ujjwala deprivation checklist.

Ayushman Bharat is absent because it is absent from the corpus: no document was
ever fetched for it (see BACKLOG M0.1). A ceiling cannot span documents that do
not exist.

## Transcription conventions

- Content only. Running headers, footers, page numbers, logos and stamps are
  furniture, not content, and are omitted. The cleaning axis strips them; these
  pages define what "stripped" should mean.
- Reading order is visual order, top to bottom, left to right within a line.
  Form pages (the Ujjwala KYC form) are transcribed in the order a reader
  meets the fields, which is the order the eval set's questions are asked in.
- Headings become markdown headings. Body text flows as paragraphs; line breaks
  in the source are layout, not meaning. List markers are kept verbatim.
- Tables are GitHub markdown tables. Multi-line cells separate their lines with
  `<br>`, because a table cell with a paragraph break in it is not a table cell
  any parser agrees on.
- Boxed Q&A layouts (the PM-KISAN FAQ) are transcribed as headings and
  paragraphs. The boxes are presentation; the reading order is linear.
- Source wording is preserved, including its own typos and its en dashes. The
  convention checker's dash rule does not apply under `corpus/gold/` for this
  reason; the credential rule still does.

## Provenance and status

Transcribed on 2026-08-24 against page images rendered from the fetched PDFs,
cross-checked against each page's text layer where one exists (the four Hindi
scans have none). The pass took about three hours of agent-assisted work:
rendering, transcription, and a second look at every page against its image.
The text layers on this corpus are not trustworthy as a source: the PM-KISAN
revised FAQ page reads "Frequentl v Asked Questions" and "SI\4F" out of its own
text layer, and pmayu-ahp-sop page 9 reads "Par culars" where the page prints
"Particulars". The images, not the layers, are the authority here.

These pages are the ceiling baseline. Before any M6 number is published against
them, they need a human pass: the transcription is careful but it was produced
in one session, and a ceiling nobody has re-read is a ceiling nobody can defend.
Until that pass happens, treat gold-page metric numbers as provisional.
