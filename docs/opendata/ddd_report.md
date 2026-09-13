# Guardian driver-state signals on an open dataset

_Generated 2026-09-13T23:00:38+00:00._

## Dataset

- **Source**: `akahana/Driver-Drowsiness-Dataset` (split `test`)
- **Licence**: NOT DECLARED by the uploader - treat as research-use only
- **Provenance**: Hugging Face: akahana/Driver-Drowsiness-Dataset. The dataset card cites https://doi.org/10.1007/978-981-33-6893-4_6 . 41,793 colour face crops at 227x227, split 33,434 train / 8,359 test.
- **Caveat**: Session-level labels: frames inherit the label of the driving session they came from, so a single 'drowsy' frame may show open eyes.

## What this does and does not show

Guardian's rules are window statistics (PERCLOS over 10 s, jaw mean over
3 s). These are loose stills, so the windows cannot be rebuilt and this is
**not** a score for the rule engine. What is comparable is PERCLOS's own
numerator: the fraction of frames whose eyes are closed.

## Results

- Frames evaluated: **1200** (face detected in 1200, 1.0 detection rate)

| class | frames | eye_blink median | closed-frame rate | jaw_open median |
|---|---:|---:|---:|---:|
| alert | 553 | 0.1056 | 0.0199 | 0.0007 |
| drowsy | 647 | 0.2351 | 0.1592 | 0.0006 |

- Eye-closure ROC AUC (drowsy vs alert): **0.7798**
- Jaw-opening ROC AUC (drowsy vs alert): **0.4893**

## Verdict

The shipped PERCLOS threshold of **0.08** lands **between the two classes** on this dataset: drowsy closes +0.0792 above it while alert stays -0.0601 relative to it. On these faces - none of them among the six it was tuned on - the threshold transfers.
