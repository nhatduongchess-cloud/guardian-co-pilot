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
| **Drowsiness demo** | Measures and draws the driver-state engine against labelled footage | result committed; rerun needs footage |
| **Explanation** | Turns one verified command into one sentence a driver can act on, in Vietnamese or English | ✅ **yes** |
| **Learning loop** | Moves one driver's drowsiness threshold within hard safety bounds, from their own trip outcomes | ✅ **yes** |
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

## Can it actually see a sleepy driver?

That is the claim the whole system rests on, so it is measured rather than
asserted. `guardian/drowsiness/` runs the shipped rule engine over **3,600
labelled frames** — six drives at 20 fps, each carrying a per-frame ground-truth
driver state — and reports the two numbers that matter together:

| | |
|---|---|
| **Impairment caught** | **99.2%** — 2,083 of 2,100 frames where the driver really was drowsy, yawning or micro-sleeping |
| **False alarms** | **9.8%** — 147 of 1,500 clear frames flagged anyway (**6.2%** once the 10 s PERCLOS window has filled) |
| **Lag after a real state change** | median **1.05 s**, worst **2.10 s** |

```bash
python -m guardian.drowsiness.demo       # console report + docs/drowsiness/
```

That command needs the labelled footage, which is not mine to publish — so the
**output is committed instead**, and you can read the result without it. Only the
module's own 15 tests run from a clean checkout; they test the measurement
(a recall that counts the wrong frames is worse than no recall), not the detector.

It writes one SVG per drive — ground truth, what Guardian said, and the measured
eyelid trace on a shared axis — plus `report.json` and a standalone
`index.html`. **[Open the rendered demo →](docs/drowsiness/index.html)**

Three things that page shows which a single accuracy number would hide:

- **Sustained closure is the easy case, and it is the dangerous one.** The
  micro-sleep and yawning drives are called correctly on *every frame*. Eyes
  shut past 1.2 s is micro-sleep whether the driver is twenty or sixty — there
  is nothing to train and nothing to overfit.
- **The lag is bought, not accidental.** Output is a majority vote over ~2 s
  because raw per-frame decisions flicker. That is exactly why the
  drowsy → distracted change on `T06-Sample` takes 2.10 s to appear. Steadier
  warnings cost response time, and the trade should be visible.
- **`T04-Sample` is where it breaks.** On one driver whose eyes never close, the
  jaw threshold fires anyway and produces stretches of phantom *yawning* — 81
  false alarms in 600 clear frames. A resting mouth posture that reads as an
  open jaw is a real failure mode, and a fixed global threshold has no answer to
  it. That is the case the learning loop below exists for.

**Six drivers is six independent samples.** These numbers show the thresholds
fire on real physiology; they are not a population claim, which is what
`guardian/opendata/` is for. Driver footage is licensed academic data
(NTHU-DDD) and is never reproduced here — only the signals derived from it, and
a test fails if an image ever appears in a rendered timeline.

---

## Saying why, in the driver's language

A warning nobody understands is a warning people switch off. Every intervention
carries one sentence explaining itself, built from the same verified command the
vehicle acted on — so the explanation cannot drift from the decision.

```
COMMAND  brake 15.0%  assist_brake  [critical]  source=kernel
  4. Comfort & Drivability   dampened   [20% -> 15%]  rate limited to 15%/frame

  SAYS: Phanh hỗ trợ 15%: người đi bộ phía trước, cách 26,9 m, TTC 3,0 giây.
        Guardian can thiệp sớm hơn bình thường vì tài xế buồn ngủ.
        - Tài xế buồn ngủ — mắt nhắm trung bình 0,26, PERCLOS 14%.
        - Cần khoảng 2,5 giây để tài xế phản ứng.
        - Safety Kernel giảm mức can thiệp ở kiểm tra độ êm khi lái.
  (explanation in 0.05 ms)
```

Three decisions in that output are load-bearing:

- **It is templated, not generated.** The budget is 2 s; templates answer in
  ~0.05 ms. A generative narrator can be attached (`narrator=`), is never on the
  safety path, and if it throws, the templates answer anyway.
- **Every user-visible string lives in `vocabulary.py`.** Adding a language is a
  translation job, not a code change.
- **Nothing English leaks into the Vietnamese.** The driver-state monitor writes
  an English sentence for the audit log; interpolating it produced
  *"Tài xế buồn ngủ — drowsy: average eye closure 0.26…"*. The evidence is now
  rebuilt from the monitor's **numbers** in the target language, and a test
  fails if an English fragment reappears.

Run it yourself: `python -m guardian.pipeline --trace T01d --lang en`.

## Learning one driver without unlearning safety

Drivers are not identical, and a threshold tuned on a population annoys half of
them. `guardian/learning/` moves one driver's PERCLOS threshold from their own
trip outcomes — but the adjustment is deliberately **asymmetric and bounded**:

| Rule | Effect on the threshold | Why |
|---|---|---|
| Loosen — driver dismissed ≥30% of alerts | **+0.005** / trip (more tolerant) | Slack is earned slowly |
| Tighten — a real drowsy event was missed | **−0.02** / trip (more sensitive) | Taken back four times faster |
| Hard floor / ceiling | clamped to **0.04 – 0.15** | The driver can never leave this band |

A missed hazard outranks everything else in the trip: even if the same drive was
full of nuisance alerts, the system was not sensitive enough where it counted,
and comfort does not get a vote. The constructor **refuses to start** if
tighten ≤ loosen.

So a driver cannot train the car into silence. Fifty consecutive trips where
every single alert is dismissed:

```
[t49] held at 0.150: false-alarm rate 100% but the safety ceiling is reached
```

and one genuine miss after that takes back four trips' worth of slack at once:

```
tightened 0.150 -> 0.130: 1 hazard(s) went unwarned
```

Every adjustment returns the sentence that justifies it — a personalisation
nobody can explain is a personalisation nobody will sign off.

---

## Quickstart

```bash
pip install -r requirements.txt

# Runs standalone — no private data needed.
python -m guardian.opendata.evaluate --dataset ddd --limit 1200
python -m pytest guardian/ -q          # 83 tests, no dataset, no simulator
```

To run the perception / decision pipelines you need your own trip footage:

```bash
export GUARDIAN_DATA_ROOT=/path/to/your/data     # PowerShell: $env:GUARDIAN_DATA_ROOT=...
python -m guardian.pipeline --trips T01 T02
python -m guardian.pipeline --trace T01 --lang en    # what the driver hears
```

---

## What is deliberately not built

The design this grew out of describes a full in-vehicle product. What is here is
the part that can be built and *tested* on a laptop; the rest needs a car.
Listing it rather than implying it exists:

| Not built | What it would need |
|---|---|
| AAOS / VHAL head-unit HMI | An Android Automotive target and vehicle HAL |
| CAN-bus telemetry (real speed, steering, brake) | A vehicle or a CAN bench — speed is currently read from the trip data |
| Battery state-of-health monitoring | Pack-level BMS access |
| Spoken (TTS) delivery of the explanation | A voice engine; the text and its timing budget are done |
| ONNX / Jetson latency numbers | The target board — current latencies are laptop CPU |
| Fleet-wide learning across drivers | A fleet, and a privacy review before any of it leaves the car |

The learning loop is also **single-driver and in-memory**: it demonstrates the
bounded-adjustment policy, not a production profile store.

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
6. **NaN is not a value, it is a silent veto.** Writing the degraded-operation
   tests turned up a real defect: the safety kernel validated speed, driver
   state and staleness, but never the scene's time-to-collision. `NaN` loses
   every comparison it takes part in, so a frame where perception had failed
   looked *exactly* like clear road — and the car told the driver it "needs 2.7s
   to stop but has nans". The kernel now rejects non-finite and negative TTC and
   geometry outright. Invalid input has to be caught where it enters, because
   downstream it is indistinguishable from safe.
7. **A safety message in two languages at once is a safety message people stop
   trusting.** The Vietnamese explanation was interpolating the monitor's
   English audit sentence. Fixed by rebuilding the evidence from the numbers, in
   whichever language is being spoken — and pinned by a test that fails on any
   English fragment.

---

## Repository layout

```
guardian/
├── opendata/         ← runs standalone: thresholds vs public datasets
│   ├── sources.py      dataset registry (licence + provenance)
│   ├── blendshapes.py  MediaPipe eye/jaw signals, no dataset coupling
│   └── evaluate.py     distributions, ROC AUC, threshold verdict
├── drowsiness/       ← does the engine actually see a sleepy driver? (evidence committed)
│   ├── detect.py       recall, false alarms, response lag - measured
│   ├── timeline.py     hand-written SVG: truth vs output vs eyelid trace
│   └── demo.py         writes docs/drowsiness/
├── challenge1/       perception: detection, stereo depth, TTC
├── challenge2/       driver state: blendshapes, rule engine, ML ablations
│   └── labels.py       the five-state label contract, importable on its own
├── world_model/      per-frame fused state
├── decision/         planning engine + deterministic safety kernel
├── explain/          ← runs standalone: verified command → one spoken sentence
│   ├── vocabulary.py   every user-visible string, VI + EN
│   └── explainer.py    templated Slow Path, optional generative narrator
├── learning/         ← runs standalone: bounded per-driver threshold tuning
├── pipeline.py       end-to-end chain vs a fixed-threshold baseline
└── demo/             HUD renderer + Streamlit dashboard
docs/
├── drowsiness/       generated demo: SVG timelines, report.json, index.html
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
