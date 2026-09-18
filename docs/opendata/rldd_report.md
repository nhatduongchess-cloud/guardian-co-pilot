# Guardian's rules against sixty drivers

_Generated 2026-09-18T14:44:36+00:00._

## Dataset

- **Source**: UTA-RLDD blink features (Ghoddoosian et al., CVPRW 2019)
- **Repository**: https://github.com/rezaghoddoosian/Early-Drowsiness-Detection.git
- **Licence**: MIT (the authors' repository). The underlying UTA-RLDD video is released separately under its own agreement and is NOT used here - only the published per-blink features are read.
- **Citation**: R. Ghoddoosian, M. Galib, V. Athitsos, 'A Realistic Dataset and Baseline Temporal Model for Early Drowsiness Detection', IEEE/CVF CVPR Workshops, 2019.
- **Participants**: 60
- **Normalisation**: Per-subject: each feature is standardised against the last third of that participant's own alert session.

## Split integrity

The files carry no participant id, so subject-disjointness cannot be
confirmed directly. Its necessary consequence can be, and was:

- Windows appearing in both the train and test side of a fold: **0**
- Windows shared between two folds' test sets: **0**
- Verdict: **leak-free**

## Result

Fit on four folds, scored on the fifth, five times. `held out` is the
number that counts; `in sample` is the same predictor scored on the
rows it was fitted on, and the gap between them is what this table is
for.

| predictor | held-out accuracy | held-out macro-F1 | in-sample macro-F1 | gap |
|---|---:|---:|---:|---:|
| rule | 0.5004 | 0.5095 | 0.5192 | +0.0097 |
| boosted_on_stats | 0.4968 | 0.5089 | 0.9997 | +0.4908 |
| logistic_on_window | 0.4792 | 0.4729 | 0.5392 | +0.0663 |
| majority | 0.3669 | 0.2685 | 0.3018 | +0.0333 |
| gru_on_sequence | 0.523 | 0.5325 | 0.7287 | +0.1962 |

## Per session, which is what a co-pilot actually emits

Guardian does not publish a fresh opinion every few minutes and
leave the driver to average them - it smooths window verdicts into
one trip state. Scored the same way here, by majority vote over each
driving session (174
held-out sessions in total, every driver unseen):

| predictor | session accuracy | session macro-F1 |
|---|---:|---:|
| rule | 0.5115 | 0.5125 |
| boosted_on_stats | 0.4483 | 0.4505 |
| logistic_on_window | 0.5057 | 0.5139 |
| majority | 0.3333 | 0.2639 |
| gru_on_sequence | 0.5402 | 0.5292 |

## Collapsed to the two states Guardian actually ships

The cabin engine either raises a drowsiness alarm or stays quiet.
Restricted to the held-out sessions whose truth is unambiguous
(`alert` or `drowsy`, 116 of them - the middle class
dropped), the two ways of being wrong cost different things: a miss
is a safety failure now, a false alarm is how a driver learns to
ignore the system.

| predictor | drowsy caught | recall on drowsy | false alarms on alert | false-alarm rate |
|---|---:|---:|---:|---:|
| rule | 21/58 | 0.3621 | 1 | 0.0172 |
| boosted_on_stats | 22/58 | 0.3793 | 9 | 0.1552 |
| logistic_on_window | 32/58 | 0.5517 | 3 | 0.0517 |
| majority | 23/58 | 0.3966 | 23 | 0.3966 |
| gru_on_sequence | 25/58 | 0.431 | 3 | 0.0517 |

Read honestly, this table does not favour the rule. It sits at the
quietest, least sensitive operating point of anything here - almost
never wrong about an alert driver, and silent through most of the
drowsy ones - while both the recurrent model and logistic regression
catch more drowsy sessions for the same handful of false alarms, and
dominate it on this trade-off. Part of that is an objective mismatch
worth naming: the cut points were chosen to maximise three-class
macro-F1, which is not the alarm objective, so this table scores an
operating point the search never aimed at. Part of it is not an
excuse - a rule tuned for macro-F1 is the rule this repository would
actually ship, and on sixty strangers it would stay quiet through two
drowsy drives in three.

## Which signal carries it

An AUC below 0.5 is not a weak signal - it is a strong one pointing the
other way. `separation` is the distance from chance in whichever
direction the statistic actually runs.

| statistic | AUC drowsy vs alert | direction | separation | alert median | drowsy median |
|---|---:|---|---:|---:|---:|
| duration_mean | 0.7676 | higher when drowsy | 0.7676 | 0.0369 | 0.6629 |
| duration_p90 | 0.81 | higher when drowsy | 0.81 | 1.0494 | 2.1213 |
| frequency_mean | 0.7362 | higher when drowsy | 0.7362 | -2.0253 | 20.8966 |
| amplitude_mean | 0.3406 | lower when drowsy | 0.6594 | 0.1261 | -0.3502 |
| velocity_mean | 0.2919 | lower when drowsy | 0.7081 | 0.1787 | -0.4447 |

## Thresholds chosen per fold

Chosen by coordinate ascent on the training folds only. Stability
across folds is itself evidence: a cut point that moves every time is
fitting the drivers, not the state.

| fold | duration_mean_drowsy | duration_p90_micro | duration_mean_low | velocity_mean_low |
|---|---:|---:|---:|---:|
| 1 | 1.2 | 3.0 | 0.4 | -0.2 |
| 2 | 1.2 | 3.0 | 0.4 | -0.2 |
| 3 | 0.8 | 2.0 | 0.4 | -0.2 |
| 4 | 1.2 | 3.0 | 0.4 | -0.2 |
| 5 | 1.2 | 3.0 | 0.4 | -0.2 |

## What this table is not

It is not a bid for state of the art. The authors' own temporal model
was designed and trained for exactly this task on exactly this data and
reports better accuracy than anything here; a four-threshold rule is not
competing with it. The comparison being made is narrower and, for this
project, more useful: given identical inputs and an identical
subject-disjoint protocol, does fitting a model to other drivers beat
cutting a per-driver statistic at a fixed point? Read the `gap` column
before the score column.

## Caveats

- Labels are session-level. A window taken from a drowsy session is labelled drowsy even if the participant happened to be alert in that minute, so per-window scores are a floor, not a ceiling.
- The front-end is dlib eye-aspect-ratio blink detection, not the MediaPipe blendshapes Guardian runs in the cabin. Conclusions here are about the METHOD - per-driver baselining plus threshold rules - and not about the specific numeric cut points shipped in challenge2/rules.py, which are defined on a different signal.
- Windows overlap: consecutive windows share blinks, so the effective number of independent observations is smaller than the row count.
