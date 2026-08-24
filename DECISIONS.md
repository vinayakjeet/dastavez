# Decisions

Every nontrivial choice gets an entry here at the time it's made - not
reconstructed later from memory. Newest entries at the top.

## Format
```
## YYYY-MM-DD: <short title>
**Context:** what problem/question forced a decision.
**Decision:** what was chosen.
**Alternatives considered:** what else was on the table, and why it lost.
**Consequences:** what this makes easier/harder later.
```

## 2026-08-24: Block kinds are a shared vocabulary, not converter-native labels

**Context:** M1.4's section-respecting splitter keys block kinds (headings may not be
cut, tables are atomic). Docling names headings `SectionHeaderItem` and tables
`TableItem`; Marker says `SectionHeader` and `Table`; MinerU's content list says
`text_level` and `table`. Keeping each converter's native labels would have made the
splitter structure-aware for one arm and blind for the others, and the M6 table would
have reported that as a property of the converters.

**Decision:** every converter maps its labels onto one vocabulary at conversion time,
with Docling's names as the reference set because they arrived first and carry a level.
The mapping lives beside each adapter (`MarkerConverter.KINDS`,
`content_list_to_blocks`) rather than in the splitter.

**Alternatives considered:** per-converter splitter configuration (doubles the axis
for no scientific gain); raw labels with a translation layer at split time (same
coupling, further from where it is testable).

**Consequences:** the kind column in any results table means the same thing for every
row. A new converter must publish its mapping or its section-respecting numbers are
not comparable, which the adapter's KINDS table makes hard to skip.

## 2026-08-24: Marker and MinerU convert from their own environments into the shared cache

**Context:** marker-pdf and mineru both pin torch-family dependencies against Docling's,
and resolving them into one environment produces a stack none of them was measured on.
The served API never converts a PDF, so the project env has no reason to carry any of
the three.

**Decision:** heavy converters run from a dedicated environment driven by
`python -m dastavez.offline_convert`, writing blocks into `corpus/.converted` keyed by
content hash plus converter identity and version. The project venv imports neither
library; ingest reads cached blocks like any other arm and fails with an install hint
if asked to convert live without it.

**Alternatives considered:** optional-dependencies extras in one env (dependency
conflict is exactly what this avoids); a microservice per converter (network, auth and
deployment for an offline batch job); dropping the two converters from the study
(the brief names them as the SOTA arms).

**Consequences:** reproducing a conversion needs two commands instead of one, both
documented in the module docstring. The cache, not the environment, is the interchange
format, which also keeps converter versions pinned per cache entry rather than per
machine state.

## 2026-08-24: Gold transcriptions are data, so the dash rule does not apply to them

**Context:** the corpus's own documents use en dashes ("Pradhan Mantri Awas Yojana
– Urban 2.0"), and the gold transcription must reproduce them or it stops being a
ceiling and becomes a paraphrase. The convention checker blocks en dashes in every
tracked file, which would have forced exactly that paraphrase.

**Decision:** the checker exempts `corpus/gold/` from the dash rule only. Every other
rule, credentials above all, still applies to gold pages, because a fixture is where
a real credential gets pasted "just for a minute".

**Alternatives considered:** normalising dashes in the transcription (corrupts the
measurement baseline); exempting all of `corpus/` (the manifest is prose-adjacent and
loses nothing by being checked).

**Consequences:** a test pins the asymmetry: a gold page may carry a source en dash
and must still fail on a planted credential. New data directories need a deliberate
decision like this one, not a quiet copy of the exemption.

<!-- Add entries above this line. -->
