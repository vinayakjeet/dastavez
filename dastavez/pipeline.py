"""One pipeline configuration, and the hash that makes two of them comparable.

The four axes are converter, cleaning, splitting and metadata enrichment. They exist
as configuration from the first milestone rather than as choices hardcoded now and
generalised later, because hardcoding one converter and adding three at M6 is how an
ablation becomes a rewrite.

The hash is the point. A results table is only meaningful if a row can be traced back
to exactly the settings that produced it, and "exactly" has to survive someone
changing a default six days later. Hashing the settings rather than naming the run
means two runs that claim to be the same configuration either are, or the hash says
they are not.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from itertools import product

CONVERTERS = ("pypdf", "docling", "docling-noocr", "docling-routed", "marker", "mineru")
CLEANINGS = ("none", "strip")
SPLITTERS = ("fixed", "section")
METADATA = ("none", "enriched")


@dataclass(frozen=True)
class PipelineConfig:
    converter: str = "pypdf"
    cleaning: str = "none"
    splitter: str = "fixed"
    metadata: str = "none"

    def __post_init__(self) -> None:
        for value, allowed, axis in (
            (self.converter, CONVERTERS, "converter"),
            (self.cleaning, CLEANINGS, "cleaning"),
            (self.splitter, SPLITTERS, "splitter"),
            (self.metadata, METADATA, "metadata"),
        ):
            if value not in allowed:
                raise ValueError(f"unknown {axis} {value!r}, expected one of {allowed}")

    @property
    def digest(self) -> str:
        canonical = json.dumps(asdict(self), sort_keys=True)
        return hashlib.sha256(canonical.encode()).hexdigest()[:12]

    @property
    def run_id(self) -> str:
        return f"{self.converter}.{self.cleaning}.{self.splitter}.{self.metadata}.{self.digest}"

    def as_row(self) -> dict[str, str]:
        return {**asdict(self), "config_hash": self.digest}


def matrix() -> list[PipelineConfig]:
    """Every configuration the ablation covers.

    Six converters times two cleanings times two splitters times two metadata
    settings is ninety-six. Only the converter axis costs a conversion pass, because
    the other three transform blocks a converter already produced, so the whole
    matrix costs six conversions.
    """
    return [
        PipelineConfig(converter=c, cleaning=cl, splitter=s, metadata=m)
        for c, cl, s, m in product(CONVERTERS, CLEANINGS, SPLITTERS, METADATA)
    ]


def one_axis_at_a_time(base: PipelineConfig | None = None) -> list[PipelineConfig]:
    """The configurations needed to attribute a change to a single axis.

    The full matrix answers which configuration is best. It does not answer which axis
    the improvement came from, because in a full matrix every row differs from every
    other in several places at once. The replicated study's headline claim is that
    chunking and metadata mattered more than the converter, and that claim needs each
    axis moved on its own against a fixed baseline.
    """
    base = base or PipelineConfig()
    varied = [base]
    for axis, options in (
        ("converter", CONVERTERS),
        ("cleaning", CLEANINGS),
        ("splitter", SPLITTERS),
        ("metadata", METADATA),
    ):
        for option in options:
            if getattr(base, axis) == option:
                continue
            varied.append(PipelineConfig(**{**asdict(base), axis: option}))
    return varied
