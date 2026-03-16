# Evaluation Reporting

This document describes the separate greedy evaluation pipeline for trained PPO checkpoints.

Primary source files:

- [analysis/evaluate_policy.py](../analysis/evaluate_policy.py)
- [scripts/evaluate.py](../scripts/evaluate.py)

## 1. Purpose

The evaluation pipeline is separate from training so that:

- training behavior is unchanged
- training runtime is unchanged
- evaluation can be run on saved checkpoints at any time

The evaluator runs greedy fresh-start episodes against the final target zone and writes a dedicated report plus dashboard.

## 2. What It Measures

Per checkpoint, the evaluator reports:

- `target-zone hit rate`
- `stable target-zone stay rate`
- `MD failure rate`
- `long-run success rate`
- `mean max CV1`
- `mean final CV1`

Definitions:

- `target-zone hit rate`: fraction of evaluation episodes where CV1 entered the target zone at least once
- `stable target-zone stay rate`: fraction of evaluation episodes where the trajectory remained in-zone for at least `STABILITY_STEPS`
- `long-run success rate`: in this evaluator, this is reported as the stable target-zone stay rate under greedy fresh-start evaluation
- `MD failure rate`: fraction of evaluation episodes terminated by MD failure / NaN handling

## 3. Evaluation Mode

The evaluator uses:

- greedy policy actions
- fresh-start episodes
- final target zone by default
- no PPO updates

That makes it a clean checkpoint evaluation path rather than another training run.

## 4. Main Command

Evaluate all checkpoints:

```powershell
python scripts\evaluate.py --config-module combined_2d
```

Evaluate one checkpoint only:

```powershell
python scripts\evaluate.py --config-module combined_2d --episode 50
```

Use more evaluation episodes for a stronger long-run estimate:

```powershell
python scripts\evaluate.py --config-module combined_2d --episodes 20
```

## 5. CLI Options

Defined in [analysis/evaluate_policy.py](../analysis/evaluate_policy.py):

| Option | Meaning |
|---|---|
| `--config-module` | Config module to import |
| `--checkpoint-dir` | Checkpoint directory override |
| `--checkpoint-glob` | Optional checkpoint glob override |
| `--episode` | Evaluate only one checkpoint episode |
| `--episodes` | Evaluation episodes per checkpoint |
| `--target-center` | Override evaluation target center |
| `--target-half-width` | Override evaluation target half-width |
| `--run` | Output evaluation run directory |

## 6. Outputs

The evaluator writes to:

```text
results_PPO/evaluation_runs/eval_<timestamp>/
```

Files:

- `checkpoint_eval_summary.csv`
- `checkpoint_eval_episodes.csv`
- `checkpoint_eval_summary.json`
- `checkpoint_eval_dashboard.png`
- `checkpoint_eval_report.md`

## 7. Dashboard

The dashboard summarizes checkpoint-level performance with:

- target-zone hit rate by checkpoint
- stable target-zone stay rate by checkpoint
- MD failure rate by checkpoint
- a compact per-checkpoint performance table

## 8. Recommended Use

Recommended order:

1. train the model
2. evaluate checkpoints with `scripts/evaluate.py`
3. inspect the dashboard and report
4. only then decide whether the agent is trained enough for downstream use
