# Guardian driver-state signals on an open dataset

_Generated 2026-09-13T23:01:03+00:00._

## Dataset

- **Source**: `n7i5x9/driver-drowsiness-dataset` (split `test`)
- **Licence**: NOT DECLARED by the uploader - treat as research-use only
- **Provenance**: Hugging Face: n7i5x9/driver-drowsiness-dataset. 23,116 images at 640x640, split train/validation/test.
- **Caveat**: Session-level labels, same caveat as `ddd`.

## What this does and does not show

Guardian's rules are window statistics (PERCLOS over 10 s, jaw mean over
3 s). These are loose stills, so the windows cannot be rebuilt and this is
**not** a score for the rule engine. What is comparable is PERCLOS's own
numerator: the fraction of frames whose eyes are closed.

## Results

- Frames evaluated: **1200** (face detected in 694, 0.5783 detection rate)

| class | frames | eye_blink median | closed-frame rate | jaw_open median |
|---|---:|---:|---:|---:|
| alert | 326 | 0.107 | 0.1687 | 0.3146 |
| drowsy | 368 | 0.1276 | 0.1875 | 0.5319 |

- Eye-closure ROC AUC (drowsy vs alert): **0.5519**
- Jaw-opening ROC AUC (drowsy vs alert): **0.5995**

## Verdict

**The threshold does not discriminate here.** Drowsy clears 0.08 (+0.1075), but so does alert (+0.0887) - a rule set at this value would fire on wide-awake drivers. The classes are only +0.0188 apart, so the signal, not the cut point, is what failed on this data.
