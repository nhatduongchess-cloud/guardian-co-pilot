# Guardian Co-Pilot

**An explainable driver-safety layer that reads both sides of the windscreen —
the road ahead and the driver behind the wheel.**

*Observe → Understand → Predict → Protect*

A personal engineering project by [Quang Nhat Duong](https://github.com/nhatduongchess-cloud).

---

## The problem

Over 11,000 people die on Vietnamese roads each year, and fatigue and
distraction lead the causes. Today's driver-assistance has three gaps:

- **It is reactive.** Emergency braking fires roughly 1–2 s before impact — too
  late at urban speeds.
- **It uses one threshold for everyone.** A novice and a 20-year veteran get
  identical warnings, so drivers learn to ignore them.
- **It never explains itself.** A car that brakes or beeps without saying why
  loses the driver's trust, and the feature gets switched off.

Guardian's answer is to watch the road *and* the driver, and to make every
warning explainable.

---

## What is actually built

| Layer | What it does | Runs standalone? |
|---|---|:---:|
| **Perception** | YOLOv8 + SGBM stereo depth → calibrated distance, collision-cone gating, tracking → time-to-collision | needs private footage |
| **Driver state** | MediaPipe FaceLandmarker blendshapes → PERCLOS, closed-run, jaw → explainable rule engine | needs private footage |
| **World model** | Fuses scene, driver, vehicle and context into one state per frame | needs private footage |
| **Decision** | Planning engine proposes; a deterministic safety kernel verifies before anything reaches the vehicle | needs private footage |
| **Open-data check** | Re-tests the shipped driver-state thresholds on public datasets | ✅ **yes** |

> **On reproducibility.** This repository ships code, not footage. The
> perception and driver-state pipelines were developed against a private trip
> dataset that is not mine to publish, so those modules need you to bring your
> own data (`GUARDIAN_DATA_ROOT`). The **open-data evaluation below runs for
> anyone**, from a clean checkout, with no private data at all — which is why it
> is the part I point people at first.

---

## Do the thresholds survive contact with strangers?

The driver-state engine is four physiological thresholds. They beat every
learned model I tried — a fine-tuned ResNet18 scored **18.5**, the rule engine
**84.5** — but that comparison hides a real weakness: the dataset had only **six
drivers**, so the CNN simply memorised *who* each person was, and the thresholds
themselves were never tested on anybody else.

`guardian/opendata/` fixes that. It points the same eye and jaw signals at
public datasets containing hundreds of unseen faces, and reports what it finds.

```bash
python -m guardian.opendata.evaluate --dataset ddd --limit 1200
```

Reports land in [`docs/opendata/`](docs/opendata/). Two datasets, two very
different answers:

| dataset | faces found | eye-closure AUC | alert closed-rate | drowsy closed-rate | threshold separates? |
|---|---:|---:|---:|---:|:---:|
| `ddd` (41k imgs, cites a paper) | 100% | **0.78** | 0.020 | 0.159 | ✅ yes |
| `n7` (23k imgs, unlabelled origin) | 58% | 0.55 | 0.169 | 0.188 | ❌ no |

**What that means.** PERCLOS is by definition the fraction of frames whose eyes
are closed, so the class-wise closed-frame rate is the same quantity the shipped
`0.08` threshold is expressed in. On `ddd` that threshold lands cleanly *between*
the two classes — it transfers to faces it was never tuned on. On `n7` it does
not: the alert class also sits at 0.169, so a rule set there would fire on
wide-awake drivers. The second result is reported as loudly as the first,
because a threshold that only works on half the data you tried is a threshold
with a known limit, not a finished one.

Two honest caveats, both surfaced in every report:

- These are **session-level labels** — a frame from a drowsy session may show
  wide-open eyes — so this compares *distributions*, and is not a score for the
  rule engine.
- **Jaw opening does not separate the classes at all** (AUC 0.49 on `ddd`).
  Yawning is rare and session labels do not mark the moment it happens.

### Provenance is part of the result

`guardian/opendata/sources.py` records each dataset's id, licence and origin,
and the evaluator copies all three into the report. Nothing is redistributed
here: datasets stream from the Hugging Face Hub into your own cache at run time.
Worth stating plainly — the best-provenanced source (`ddd`, which cites
[a published paper](https://doi.org/10.1007/978-981-33-6893-4_6)) carries **no
licence tag**, while the one tagged Apache-2.0 has an entirely empty dataset
card. Treat both as research-use and verify before building anything commercial
on them.

---

## Quickstart

```bash
pip install -r requirements.txt

# Runs standalone — no private data needed.
python -m guardian.opendata.evaluate --dataset ddd --limit 1200
python -m pytest guardian/opendata/tests/ -q
```

To run the perception / decision pipelines you need your own trip footage:

```bash
export GUARDIAN_DATA_ROOT=/path/to/your/data     # PowerShell: $env:GUARDIAN_DATA_ROOT=...
python guardian/pipeline.py --trips T01 T02
```

---

## What I learned building it

The findings I'd defend in a review, each measured rather than asserted:

1. **Physiology beat deep learning, because of the data shape.** Six drivers is
   six independent samples. The CNN hit a training loss of 0.0002 in one epoch
   and then predicted a single class for every frame of an unseen person. It had
   learned *who*, not *what*.
2. **I leaked identity into my own "identity-invariant" features.** The first
   feature set included head pose, brow and squint — all camera- and face-shape
   dependent. Removing them nearly doubled the score (23.8 → 43.7).
3. **Higher resolution is not free.** Upscaling 640×360 source to 1280 destroyed
   small-object recall (bike detections 13/19 → 0/19); the interpolated blur
   pushed the detector into nonsense classes. Model capacity was never the
   bottleneck — thresholding was.
4. **An honest protocol costs you points and is worth it.** Leave-one-trip-out
   tuning turned an in-sample 90.7 into a truthful 84.5. The smaller number is
   the one reported.
5. **A threshold has to be tested on people it has never seen** — which is what
   `guardian/opendata/` is for, and how the `n7` limitation above surfaced.

---

## Repository layout

```
guardian/
├── opendata/         ← runs standalone: thresholds vs public datasets
│   ├── sources.py      dataset registry (licence + provenance)
│   ├── blendshapes.py  MediaPipe eye/jaw signals, no dataset coupling
│   └── evaluate.py     distributions, ROC AUC, threshold verdict
├── challenge1/       perception: detection, stereo depth, TTC
├── challenge2/       driver state: blendshapes, rule engine, ML ablations
├── world_model/      per-frame fused state
├── decision/         planning engine + deterministic safety kernel
├── pipeline.py       end-to-end chain vs a fixed-threshold baseline
└── demo/             HUD renderer + Streamlit dashboard
docs/
├── opendata/         generated reports (committed as evidence)
└── WRITEUP.md        approach, ablations, known issues
```

---

## Licences

| Component | Licence | Use |
|---|---|---|
| YOLOv8 (Ultralytics) | **AGPL-3.0** | obstacle + phone detection |
| MediaPipe FaceLandmarker | Apache-2.0 | face blendshapes |
| scikit-learn, OpenCV, NumPy, pandas | BSD / Apache-2.0 | general |

All pretrained weights are the public COCO / MediaPipe releases, downloaded on
first use. **On AGPL-3.0:** YOLOv8 is copyleft, so a distributed product built on
it must publish its source. For a commercial deployment I would swap in a
permissively licensed detector — the driver-state rule engine has no AGPL
dependency at all.

---

*"AI should assist people, never replace safe decisions."*
