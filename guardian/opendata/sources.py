"""
sources.py
==========

OPEN DATASET REGISTRY — provenance is part of the result.

Guardian's driver-state engine was tuned on six drivers. Six. Every learned
model we tried memorised *who* rather than *what* (see docs/WRITEUP.md), and the
physiological thresholds we shipped instead were never checked against anyone
outside that handful of people. This module exists to fix that: it points the
same eye/jaw signals at public datasets with many more faces.

Two rules govern everything here, and both are deliberate:

1.  **We never vendor the images.** Datasets are fetched at run time from the
    Hugging Face Hub into the user's own cache. This repository ships code, not
    other people's data.
2.  **Licence and provenance travel with the numbers.** Every source below
    records where it came from and what we are allowed to do with it, and the
    evaluator copies that into its report. A result whose data you cannot trace
    is not evidence.

Honest caveat that applies to all of them: these are *session-level* labels. A
driver filmed during a drowsy session is labelled drowsy for the whole clip,
so an individual frame marked `drowsy` may well show wide-open eyes. That makes
them sound for comparing DISTRIBUTIONS across many people, and unsound for
scoring any single frame. The evaluator reports it that way.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class OpenDataset:
    """One public dataset, with the paperwork attached."""

    key: str
    repo_id: str
    #: Hugging Face split to read.
    split: str
    #: Column holding the image, and the column holding the label.
    image_column: str
    label_column: str
    #: Maps the dataset's own label values onto Guardian's vocabulary.
    #: Guardian speaks `drowsy` / `alert`; everything else is the source's dialect.
    #: `hash=False`: a frozen dataclass hashes every field, and a dict is
    #: unhashable - without this the registry values cannot go in a set.
    label_map: dict[object, str] = field(hash=False)
    licence: str
    provenance: str
    notes: str
    n_examples: int | None = None
    aliases: tuple[str, ...] = field(default_factory=tuple)

    def normalise(self, raw_label: object) -> str | None:
        """Translate a source label into Guardian's vocabulary, or None if unknown."""
        if raw_label in self.label_map:
            return self.label_map[raw_label]
        # Datasets sometimes hand back the class name instead of the index.
        if isinstance(raw_label, str):
            lowered = raw_label.strip().lower()
            for key, value in self.label_map.items():
                if isinstance(key, str) and key.strip().lower() == lowered:
                    return value
        return None


#: The default. Best provenance of everything we surveyed: it cites a published
#: paper, ships as clean parquet with a train/test split, and the images are
#: colour face crops from real driver footage rather than re-processed
#: derivatives. Its one weakness is that the uploader set no licence tag, which
#: is exactly why we fetch it rather than redistribute it.
DDD = OpenDataset(
    key="ddd",
    repo_id="akahana/Driver-Drowsiness-Dataset",
    split="test",
    image_column="image",
    label_column="label",
    label_map={0: "drowsy", 1: "alert", "Drowsy": "drowsy", "Non Drowsy": "alert"},
    licence="NOT DECLARED by the uploader - treat as research-use only",
    provenance=(
        "Hugging Face: akahana/Driver-Drowsiness-Dataset. The dataset card cites "
        "https://doi.org/10.1007/978-981-33-6893-4_6 . 41,793 colour face crops "
        "at 227x227, split 33,434 train / 8,359 test."
    ),
    notes=(
        "Session-level labels: frames inherit the label of the driving session "
        "they came from, so a single 'drowsy' frame may show open eyes."
    ),
    n_examples=8359,
    aliases=("akahana", "default"),
)

#: Larger and carrying an Apache-2.0 tag, but the dataset card is empty - no
#: provenance at all. A permissive tag on a re-upload does not make the
#: underlying footage permissive, so we keep it selectable and say so plainly
#: rather than letting the licence field flatter us.
MANITH = OpenDataset(
    key="manith",
    repo_id="Manith/driver_drowsiness_detection_dataset",
    split="train",
    image_column="image",
    label_column="label",
    label_map={0: "drowsy", 1: "alert", "drowsy": "drowsy", "notdrowsy": "alert"},
    licence="apache-2.0 (declared) - provenance undocumented; verify before relying on it",
    provenance=(
        "Hugging Face: Manith/driver_drowsiness_detection_dataset. Dataset card "
        "contains a licence tag and nothing else. ~6 GB across four zips."
    ),
    notes="Unverified origin. Included for breadth, not as the primary evidence.",
    aliases=(),
)

#: A third opinion, useful mainly as a consistency check: if a threshold holds
#: on two independent uploads it is less likely to be an artefact of one.
N7 = OpenDataset(
    key="n7",
    repo_id="n7i5x9/driver-drowsiness-dataset",
    split="test",
    image_column="image",
    label_column="label",
    label_map={0: "drowsy", 1: "alert", "drowsy": "drowsy", "not_drowsy": "alert"},
    licence="NOT DECLARED by the uploader - treat as research-use only",
    provenance=(
        "Hugging Face: n7i5x9/driver-drowsiness-dataset. 23,116 images at "
        "640x640, split train/validation/test."
    ),
    notes="Session-level labels, same caveat as `ddd`.",
    n_examples=2313,
    aliases=(),
)

REGISTRY: dict[str, OpenDataset] = {}
for _source in (DDD, MANITH, N7):
    REGISTRY[_source.key] = _source
    for _alias in _source.aliases:
        REGISTRY[_alias] = _source

DEFAULT_KEY = "ddd"


def get(key: str = DEFAULT_KEY) -> OpenDataset:
    """Look a dataset up by key or alias, failing loudly with the valid options."""
    try:
        return REGISTRY[key]
    except KeyError:
        canonical = sorted({s.key for s in REGISTRY.values()})
        raise KeyError(f"unknown dataset {key!r}; choose one of {canonical}") from None
