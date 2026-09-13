# Guardian Co-Pilot — Write-up

A personal project by Quang Nhat Duong

> How this was built: the approach per module, the ablations that decided it,
> and an honest account of where AI helped and where it was wrong.

---

## 1. The problem we are solving

Over 11,000 people die on Vietnamese roads each year, and fatigue and
distraction lead the causes. Today's ADAS has three gaps:

- **Reactive.** AEB fires roughly 1–2 s before impact — too late at urban speeds.
- **One threshold for everyone.** A novice and a 20-year veteran get identical
  warnings, so drivers learn to ignore them (alert fatigue).
- **No explanation.** A car that brakes or beeps without saying why loses the
  driver's trust, and the feature gets switched off.

Guardian's answer is to read **both** sides of the windscreen — the road and the
driver — and to make every warning **explainable**. This dataset is unusually
well suited to that: it is one of the few that ships forward stereo *and* cabin
footage for the same timeline.

---

## 2. Approach per challenge

### Challenge 1 — Collision Risk Monitor

**Step 1: understand the target before modelling it.** Rather than guessing what
"TTC" meant, we fitted the ground truth and recovered the reference
definition:

```
ttc = (longitudinal_distance − GAP[class]) / closing_speed
GAP:  walker 2.60 m (σ 0.02) | bike 3.49 m (σ 0.00) | vehicle 4.81 m (σ 0.62)
counted only when |lateral_distance| ≤ 2.50 m
```

The lateral gate was recovered exactly: targets with `in_collision_cone = True`
peak at 2.48 m, those with `False` start at 2.51 m. Only **538 of 15,724**
targets qualify — most frames are genuinely `inf`, and the baseline's 19.7 is
mostly false alarms from a fixed ROI.

**Step 2: check the sensor before blaming it.** Against the depth ground truth
on keyframes, dense SGBM gives 1.8% median error at 0–10 m, 5.3% at 10–20 m,
12.2% at 20–40 m. Depth was fine; **object selection** was the problem.

**Step 3: detect, calibrate, gate, track.**

| Stage | Choice | Why |
|---|---|---|
| Detection | YOLOv8s @ conf 0.10 | replaces the fixed ROI; small objects need low conf |
| Depth | SGBM median over the box's central 50% | avoids background bleeding in at the edges |
| Calibration | per class, `Z = a·z_stereo + b·z_width + c` | raw stereo measures camera→surface, GT measures centre→centre |
| Gating | \|lateral\| ≤ 2.5 m | the recovered collision cone |
| Speed | slope of distance over 10 frames, ego-speed fallback | frame-to-frame differencing amplifies depth noise |

**Result: 19.70 → 65.60**, in four measured steps:

| step | change | composite |
|---|---|---|
| 0 | benchmark SGBM baseline | 19.70 |
| 1 | YOLOv8n + calibrated stereo + cone + tracking | 46.40 |
| 2 | YOLOv8s @ conf 0.10 | 52.00 |
| 3 | per-class confidence (vehicle .25 / bike .10 / walker .10) | 59.10 |
| 4 | median smoothing widened 3 → 13 frames | **65.60** |

Step 4 deserves a note: the scorer treats a missed hazard as 99 s, so a single
dropped frame adds ~97 s of critical MAE. A wider median window bridges brief
detection gaps instead of erasing short hazards. It is a real optimum — past 13
the score falls again (window 21 → 63.3).

Leave-one-trip-out tuning (grid searched on five trips, scored on the held-out
sixth) selected the same configuration in 5 of 6 folds and gives an **honest
LOTO mean of 65.23** — statistically indistinguishable from the in-sample 65.60,
which is the evidence that this is not over-fitted.

### Open-source upgrades we tested — including the ones we rejected

We evaluated several standard computer-vision upgrades against the benchmark's
scorer. Three of them made results **worse**, and we report that because a
negative result measured is worth more than an untested idea.

| Upgrade | Result | Verdict |
|---|---|---|
| Inference at `imgsz=1280` (upscaling) | bike recall on T02 critical frames **13/19 → 0/19** | **rejected** |
| Test-time augmentation (`augment=True`) | 13/19 → 9/19 | **rejected** |
| Bigger backbones: YOLOv8m / YOLOv8l | 12/19 each, vs 13/19 for v8s, and 2–3× slower | **rejected** |
| Rider fallback (use `person` when the bike is missed) | +2 of 19 frames, with added false-alarm risk | not adopted |
| Per-class confidence thresholds | 51.9 → **59.1** | **adopted** |
| Wider median smoothing (3 → 13 frames) | 59.1 → **65.6** | **adopted** |

The upscaling result is the interesting one. The source images are 640×360;
resizing to 1280 adds no information, and the interpolated blur pushed YOLO into
nonsense classes — on the same 19 frames it stopped reporting `motorcycle`
entirely and instead returned `clock` (17), `bird` (14) and `airplane` (3).
**"Higher resolution helps small objects" is false when the resolution is
synthetic.** Likewise, YOLOv8s at native resolution beat both v8m and v8l: model
capacity was never the bottleneck, thresholding was.

Still on the table for future work: ByteTrack/BoT-SORT (shipped inside
Ultralytics) to replace our greedy tracker for better occlusion handling, and an
ONNX Runtime / TensorRT export to put a real latency figure behind the
edge-deployment claim.

### Challenge 2 — Driver Intelligence

We ran four approaches and let the numbers decide. Every row is Leave-One-Trip-Out.

| # | Approach | LOTO composite |
|---|---|---|
| 1 | ResNet18 fine-tune on raw cabin pixels | **18.5** |
| 2 | HistGradientBoosting, 40 temporal features | 23.8 |
| 3 | HistGradientBoosting, 26 identity-invariant features | 43.7 |
| 4 | **Physiological rule engine** (shipped) | **84.5** |

**Why (1) failed.** 3,600 frames, but only **six drivers**; 600 frames per trip
at 20 FPS are near-duplicates. Effective sample size is ~6. Training loss hit
0.0002 in a single epoch and the model then predicted `alert` for all 600 frames
of an unseen drowsy driver. It had learned *who*, not *what*.

**Why (2) barely helped.** We had unwittingly re-introduced identity through
`head_pitch/yaw/roll`, `brow_down*` and `eye_squint*` — all of which depend on
camera mounting and face shape. Removing them (approach 3) nearly doubled the
score, which is itself the evidence.

**Why (4) won.** With six trip-level samples, *any* learned model memorises the
trip. Physiological thresholds need no training data at all. PERCLOS
("percentage of eyelid closure") is the established drowsiness metric in the
driver-monitoring literature; we implemented it rather than inventing a metric.

Thresholds were tuned **per fold on the other five trips** and scored on the
held-out sixth. Five of six folds independently selected identical values:

```
phone_rate(3s) 0.15 · eye_blink(10s) 0.12 · PERCLOS(10s) 0.08
microsleep PERCLOS 0.45 · closed-run 1.2 s · jawOpen(3s) 0.20
```

That stability is why we trust them. The rule engine also scores the two folds
where `yawning` / `microsleep` are missing from training data — folds on which
every learned model scored exactly **0.0**.

**Explainability is the product, not a consolation.** Every decision prints its
reason: *"microsleep: eyes closed 68% of the last 10 s, longest single closure
1.6 s (thresholds 45% / 1.2 s)"*. This is what the HUD shows the driver, and it
is precisely the trust gap identified in Section 1.

### Challenge 3 — Fleet Safe Driving Score

The evaluator recomputes the score from trip kinematics plus `near_miss`, and
ignores the values in `predicted_risk_score`. Every term except `near_miss` is
deterministic and identical for all teams. `near_miss` comes from our own
`predicted_ttc`, and each spurious one costs 10 composite points — so Challenge 3
is decided by Challenge 1. We emit a genuine per-frame risk anyway (driver state
fused with obstacle proximity) because it drives the HUD and is honest.

---

## 3. Use of AI in development

I used AI heavily and deliberately. Here is how, which prompts worked, and
where it was wrong.

### Tool and workflow

**Claude Code (Anthropic Claude Opus 4.8)** as a pair programmer, driven from a
terminal with read/write access to the repository and the ability to execute
code. Our working protocol was fixed for the whole session:

1. **Explain before coding.** Every file began with goal / filename / why it is
   needed / expected output, and a plain-language explanation of the technique
   before a single line was written.
2. **One file at a time**, reviewed before moving on.
3. **Never trust a claim without running it.** Every number in the README comes
   from executing the benchmark's `evaluation.py`, not from an assertion.
4. **Validate a hypothesis cheaply before building on it.** Before writing the
   MediaPipe module we ran a 20-frame spike to confirm the blendshapes actually
   separated the classes. Before writing the TTC module we validated stereo
   depth against the depth ground truth.

### Prompt patterns that worked

| Pattern | Example | Effect |
|---|---|---|
| Force a diagnosis before a fix | *"Model predicts drowsy for all 600 frames — diagnose before changing anything"* | found the identity-overfit root cause instead of blind hyper-parameter tuning |
| Reverse-engineer instead of guess | *"Fit the relationship between GT `ttc` and `distance/closing_speed`"* | recovered the exact GAP constants and the 2.5 m cone |
| Demand the honest protocol | *"Tune the thresholds on 5 trips and score on the 6th"* | turned an in-sample 90.7 into a trustworthy 84.5 |
| Look at the data, literally | *"Render a montage of driver frames from every trip"* | one image revealed that each trip is a different person — the key insight of Challenge 2 |
| Ablate before believing | *"Compare full features vs invariant features vs pure rules"* | produced the 18.5 → 23.8 → 43.7 → 84.5 table |

### Where AI was most effective

- **Systematic ablation.** Four Challenge-2 approaches with consistent LOTO
  evaluation in a single session — the comparison table is the strongest
  technical evidence in this project.
- **Reverse-engineering the scoring formula** from ground truth, which reframed
  Challenge 1 from "estimate depth well" to "select the right object and gate it".
- **Infrastructure that prevented scoring disasters.** The submission validator
  caught that the reference baseline writes a `ground_truth_ttc` column;
  submitting that file unchanged would have violated the "never write ground
  truth" rule.
- **Speed on boilerplate**: caching layers, CLIs, the HUD renderer.

### Where AI was wrong, and had to be corrected

We list these because they are the honest part of the answer.

1. **It proposed a CNN first, and the CNN failed (18.5/100).** The suggestion was
   reasonable and standard, but wrong for a 6-subject dataset. Only running it
   revealed that.
2. **It leaked identity back into the "identity-invariant" features.** The first
   feature set included head pose, brow and squint — all camera- and face-shape
   dependent. The model then predicted `drowsy` for the trip with the *lowest*
   eye closure, which is how we caught it. Removing those features nearly doubled
   the score.
3. **It initially claimed MobileNet would be faster than ResNet18** on the
   grounds of parameter count. Measurement showed the opposite on this GPU
   (1,990 vs 530 img/s) because depthwise convolutions are memory-bound. The
   README now states both sides.
4. **It over-tuned on the practice set.** The first rule thresholds scored 90.7
   in-sample; forcing a leave-one-trip-out protocol reduced that to a truthful
   84.5. We report the smaller number.
5. **It could not reason about the CarSky/VDP platform**, having no reliable
   knowledge of it, and said so rather than inventing details. Platform-specific
   steps were handled by the team.
6. Minor: several shell/monitoring scripts had bugs on the first attempt and
   needed a second pass.

### What AI did not decide

The team owns the product direction (four-vertical Guardian architecture, the
explainability-first pitch), the choice to ship the rule engine over the ML
model, the decision **not** to adopt per-trip baseline normalisation, and every
statement in the "known issues" section.


---

## 4. Hardware and runtime

| | |
|---|---|
| Machine | Windows 11, Python 3.14, NVIDIA RTX 4070 Laptop (8 GB), 20 cores |
| Feature extraction (Challenge 2) | 21,000 frames @ ~50 fps ≈ **7 min**, cached |
| Perception (Challenge 1) | 21,600 frames @ ~20 fps ≈ **25 min**, cached |
| Full run from warm cache | **9 seconds** |
| CNN ablation training | ~40 s per LOTO fold |
| No model was trained on data outside this project dataset | |

Everything runs on CPU too, just slower (we measured a 38× GPU speed-up).

---

## 5. Reproducing these results

```bash
# Install (CUDA build of torch first; CPU-only works as well)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
pip install -r requirements.txt

# Point at the extracted dataset (all 16 trip folders in one parent folder)
$env:GUARDIAN_DATA_ROOT = "C:\guardian_data"        # PowerShell
export GUARDIAN_DATA_ROOT=/path/to/guardian_data     # Linux/macOS

# One command → predictions/GuardianCoPilot/T01d.csv ... T10d.csv
python guardian/run_all.py
```

Optional, to verify our numbers rather than take our word for them:

```bash
# Self-check on the practice trips, scored by the BENCHMARK's evaluator
python guardian/run_all.py --practice --out predictions_selfcheck --evaluate

# Reproduce the baseline reference (19.70)
python guardian/evaluation/harness.py

# Reproduce Challenge 1 (52.00)
python guardian/challenge1/ttc.py

# Reproduce the Challenge 2 ablation
python guardian/challenge2/train.py        # CNN — 18.5
python guardian/challenge2/classifier.py   # gradient boosting — 23.8
python guardian/challenge2/rules.py        # rule engine — 92.6 in-sample

# Render the demo video
python guardian/demo/hud.py --trip T04-Sample
```

Determinism: seeds are fixed (`set_seed(42)`), no random augmentation is used at
inference, and the shipped Challenge-2 predictor is a deterministic rule engine.

---

## 6. Known issues

Restated from the README so this document stands alone.

1. **Bike/pedestrian recall in Challenge 1.** Vehicles: 76/76 critical frames
   detected (T04, composite 82.4). Bikes 7/19 (T02), walkers 8/12 (T01). Those
   two trips hold the overall composite down.
2. **Challenge 2 labels most scored frames `drowsy`** (T08d: 1744/1800). We
   tested per-trip baseline normalisation; it changed practice results by 0.0, so
   we did not adopt it. It may well be correct — T08d's median eye closure
   (0.120) exceeds that of the genuinely drowsy practice trip (0.077), and the
   brief describes event-dense midnight scenarios — but it is **unverifiable
   without labels**.
3. **Challenge 3 cannot be validated offline**: all six practice trips have
   `safe_driving_score = 0` (capped), so the local composite is always ~100.
4. **Calibration is fitted on the practice split** — legitimate, but it is a fit.
5. **The TTC smoother is centred** and therefore uses a few future frames. Valid
   offline; a production system needs a causal window.
6. **One repaired frame**: `T08d/kitti/image_2/001615.jpg` fails CRC in the
   distributed archive; we copied the frame 50 ms earlier. See
   `docs/DATA_REPAIRS.md`.

---

## 7. What we would do next

**Days, not months:**
- Larger YOLO weights (v8m/v8l) and class-specific confidence for bikes and
  pedestrians — the single clearest Challenge-1 gain left.
- Predict `predicted_headway_sec` if the benchmark adds the column; it is the one
  term of the original Challenge-3 formula the evaluator cannot currently score.

**Product direction:**
- Swap YOLOv8 (AGPL-3.0) for a permissively licensed detector for any commercial
  deployment; Challenge 2's rule engine is already dependency-light.
- Causal temporal windows and an on-vehicle latency budget — the rule engine is
  arithmetic over cached features and is genuinely edge-deployable.
- Personalised thresholds: a novice's PERCLOS limit should be stricter than a
  veteran's. Our architecture already carries a driver profile; this is the
  Guardian four-vertical story completed.

---

*"AI should assist people, never replace safe decisions."*
