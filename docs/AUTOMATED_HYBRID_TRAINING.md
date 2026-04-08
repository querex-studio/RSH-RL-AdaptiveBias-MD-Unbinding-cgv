# Automated Hybrid PPO Training

This document explains the closed-loop hybrid controller added in version `7.3.0`.

Primary source files:

- [analysis/hybrid_auto_controller.py](../analysis/hybrid_auto_controller.py)
- [scripts/hybrid_auto.py](../scripts/hybrid_auto.py)
- [main_2d.py](../main_2d.py)
- [combined_2d.py](../combined_2d.py)
- [analysis/hybrid_restart_selection.py](../analysis/hybrid_restart_selection.py)

Related documents:

- [HYBRID_RESTART_SELECTION.md](./HYBRID_RESTART_SELECTION.md)
- [PCA_ANALYSIS.md](./PCA_ANALYSIS.md)
- [TICA_ANALYSIS.md](./TICA_ANALYSIS.md)
- [CLI_WORKFLOW.md](./CLI_WORKFLOW.md)

## Purpose

The automated hybrid controller makes PCA/TICA useful for PPO learning without pretending that PCA/TICA are part of each PPO update.

The loop is:

```text
1. Train PPO from the original bound structure.
2. Run PCA and/or TICA on the saved DCD trajectories.
3. Rank restart frames using CV progress, cluster rarity, and analysis-space displacement.
4. Export selected frames as restart PDBs.
5. Train the next PPO round with a mixture of:
   - fresh-start episodes from the original bound structure
   - restart episodes from PCA/TICA-selected PDBs
6. Repeat.
```

This lets PCA/TICA affect the **between-round training distribution**, not the individual PPO step update.

## Why This Helps PPO

Fresh-start PPO episodes are still essential because the final task is to learn unbinding from the original bound state.

Restart episodes help because they expose PPO to difficult intermediate and late-stage pathway states more often. That can improve:

- sample efficiency
- target-zone hit rate
- stable target-zone stay rate
- recovery from intermediate states
- pathway diversity
- late-stage Mg-phosphate separation behavior

The default controller uses restart states only after round 1.

## Important Scientific Caveat

This is not the original full Adaptive CVgen algorithm.

Accurate description:

```text
PPO-biased MD with PCA/TICA-guided automatic restart curriculum.
```

Avoid claiming:

```text
Full Adaptive CVgen
```

unless online CV-weight learning and round-by-round Adaptive CVgen reward variables are added.

## Main Command

Recommended 2-round test:

```powershell
python scripts\hybrid_auto.py --rounds 2 --initial-episodes 10 --episodes-per-round 10 --restart-fraction 0.30 --analysis-space both --top-restarts 10
```

Lower-cost test:

```powershell
python scripts\hybrid_auto.py --rounds 2 --initial-episodes 10 --episodes-per-round 5 --restart-fraction 0.30 --analysis-space tica --top-restarts 5 --max-traj 20 --stride 5 --contact-stride 5 --max-residues 40
```

Longer experiment:

```powershell
python scripts\hybrid_auto.py --rounds 4 --initial-episodes 20 --episodes-per-round 20 --restart-fraction 0.30 --analysis-space both --top-restarts 20
```

## Restart Fraction

The recommended starting point is:

```text
restart_fraction = 0.30
```

That means:

```text
70% fresh-start episodes
30% PCA/TICA-selected restart episodes
```

This balance is deliberate. Too many restart episodes can teach the agent to solve only late-stage states rather than the full original unbinding task.

## Curriculum Handling

Restart episodes use the final target zone by default because they usually start from intermediate or late-stage pathway states.

Restart episodes do **not** promote the fresh-start curriculum. Only fresh-start episodes update the curriculum success window. This prevents a near-target restart from falsely making the bound-state curriculum look solved.

## Outputs

The controller writes:

```text
results_PPO/hybrid_auto/<timestamp>/
```

Key files:

- `hybrid_auto_summary.json`
- `hybrid_auto_report.md`

Each round also writes analysis runs:

```text
results_PPO/analysis_runs/hybrid_auto_round_01_pca/
results_PPO/analysis_runs/hybrid_auto_round_01_tica/
results_PPO/analysis_runs/hybrid_auto_round_02_pca/
results_PPO/analysis_runs/hybrid_auto_round_02_tica/
```

Each PCA/TICA run then contains its hybrid restart outputs:

```text
hybrid_restart/
pca_hybrid_restart/
```

## When To Use It

Use automated hybrid training after the basic 10-episode workflow is working:

```powershell
python main_2d.py
python scripts\pca.py --config-module combined_2d --max-traj 0
python scripts\tica.py --config-module combined_2d --max-traj 0 --lag 5
```

Then switch to:

```powershell
python scripts\hybrid_auto.py --rounds 2 --initial-episodes 10 --episodes-per-round 10 --restart-fraction 0.30 --analysis-space both --top-restarts 10
```

## How To Judge Success

After a hybrid run, compare:

- target-zone hit rate
- stable target-zone stay rate
- MD failure rate
- mean max CV1
- mean final CV1
- PCA/TICA pathway diversity
- chemical plausibility of exported restart PDBs

The key scientific question is whether restart-guided training improves fresh-start evaluation, not only whether restart episodes succeed.
