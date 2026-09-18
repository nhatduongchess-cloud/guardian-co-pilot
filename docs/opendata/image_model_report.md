# A face model on public driver images, and what it is worth

_Generated 2026-09-18T14:47:26+00:00._

> **Read this before the numbers.** The public image datasets carry no
> participant id, so the split below is a split of **frames, not
> drivers** - consecutive frames of the same face sit on both sides of
> it. The in-dataset score is therefore not a cross-driver score and is
> **not comparable** to anything in `rldd_report.md`. It is reported so
> that the second number has something to fall from.

## What was trained

- **Dataset**: `akahana/Driver-Drowsiness-Dataset`
- **Licence**: NOT DECLARED by the uploader - treat as research-use only
- **Protocol**: random_frame_split - the dataset ships no participant id, so frames of the same face appear in both train and test. NOT a cross-driver protocol and not comparable to rldd_report.md.
- **Epochs**: 3

## Result

| split | n | accuracy | balanced acc. | drowsy recall | alert recall | AUC |
|---|---:|---:|---:|---:|---:|---:|
| ddd test (random-frame split) | 3000 | 0.998 | 0.9979 | 1.0 | 0.9957 | 0.9999 |
| n7 (never trained on) | 2313 | 0.5024 | 0.49 | 0.5862 | 0.3938 | 0.4974 |

## The gap

Balanced accuracy falls **-0.5079** when the same weights meet a different upload of the same task (0.9979 to 0.49), against a chance floor of 0.5.

The weights do not transfer. Most of the in-dataset score was the model recognising sessions it had already seen, which is exactly the failure this project documented on six drivers - reproduced here with thousands of faces, because more faces without a driver-aware split does not fix it.
