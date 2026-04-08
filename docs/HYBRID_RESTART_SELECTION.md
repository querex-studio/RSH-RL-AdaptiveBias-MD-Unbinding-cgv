# Hybrid PPO + Adaptive-CVgen-Style Restart Selection

This document describes the optional hybrid workflow built around the existing PPO-biased MD model.

Primary source files:

- [analysis/hybrid_restart_selection.py](../analysis/hybrid_restart_selection.py)
- [scripts/hybrid_restart.py](../scripts/hybrid_restart.py)
- [scripts/pca_hybrid_restart.py](../scripts/pca_hybrid_restart.py)
- [analysis/tica_phosphate_pathway.py](../analysis/tica_phosphate_pathway.py)
- [analysis/pca_phosphate_pathway.py](../analysis/pca_phosphate_pathway.py)

Related documents:

- [AUTOMATED_HYBRID_TRAINING.md](./AUTOMATED_HYBRID_TRAINING.md): closed-loop PPO training with PCA/TICA-selected restart states
- [TICA_ANALYSIS.md](./TICA_ANALYSIS.md): TICA feature construction and seed-candidate scoring
- [CV_SELECTION.md](./CV_SELECTION.md): CV2 definition and automatic CV2 candidate ranking
- [EVALUATION_REPORTING.md](./EVALUATION_REPORTING.md): checkpoint evaluation and success metrics
- [CLI_WORKFLOW.md](./CLI_WORKFLOW.md): command order for training and analysis

## Purpose

The hybrid workflow implements the practical "Option B" design:

```text
PPO-biased MD explores Mg-phosphate unbinding pathways.
TICA projects the saved trajectories into slow-coordinate space.
Adaptive-CVgen-style scoring ranks restart frames after the PPO run.
Selected frames are exported as restart PDBs for follow-up validation or restart experiments.
```

This does not replace PPO training. PPO remains the online RL model and still controls the Gaussian bias actions during OpenMM propagation.

This is also not the full original Adaptive CVgen method. The project does not implement online `W`, `WP`, `delta`, `alpha`, or round-by-round CV-weight learning. Instead, it uses an Adaptive-CVgen-style downstream controller for restart selection.

## Scientific Role

The workflow is designed to keep the roles clean:

- PPO-biased MD is used for accelerated exploration over difficult unbinding barriers.
- TICA is used to identify slow trajectory modes and metastable regions.
- Cluster rarity and slow-mode displacement are used to avoid selecting only redundant high-CV1 frames.
- Exported restart structures can be tested separately with unbiased MD, weakly biased MD, or PPO evaluation/restart experiments.

This separation is important because TICA populations from biased trajectories should not be interpreted as unbiased kinetics without additional correction.

## Required Inputs

Run PPO training first:

```powershell
python main_2d.py
```

Then run TICA:

```powershell
python scripts\tica.py --config-module combined_2d --max-traj 0 --lag 5
```

The hybrid selector consumes these TICA outputs:

- `data/phosphate_pathway_tica_scores_all.csv`
- `data/tica_adaptive_seed_candidates.csv`

The TICA candidate score is:

```text
score =
  progress_weight * CV1_progress
+ cv2_weight * CV2_progress
+ novelty_weight * cluster_rarity
+ slow_mode_weight * TICA_distance_from_initial
```

For the active model, the CVs are:

- `CV1`: Mg-P distance, atoms `7799-7840`
- `CV2`: Mg-phosphate O2 distance, atoms `7799-7842`

`CV2_PROGRESS_DIRECTION = "increase"` is respected by TICA candidate scoring, so increasing Mg-O2 separation is treated as forward progress.

## Main Command

Use the latest TICA run automatically and restrict the first test to episode 10:

```powershell
python scripts\hybrid_restart.py --config-module combined_2d --max-episode 10 --top-restarts 10
```

Use a specific TICA run:

```powershell
python scripts\hybrid_restart.py --config-module combined_2d --tica-run results_PPO\analysis_runs\<tica_run> --max-episode 10 --top-restarts 10
```

Create plots and CSVs without exporting PDB files:

```powershell
python scripts\hybrid_restart.py --config-module combined_2d --tica-run results_PPO\analysis_runs\<tica_run> --max-episode 10 --top-restarts 10 --no-export-pdb
```

Disable the episode limit after the test run:

```powershell
python scripts\hybrid_restart.py --config-module combined_2d --max-episode 0 --top-restarts 20
```

## PCA-Space Hybrid Command

The same restart exporter can also consume PCA outputs after `scripts\pca.py` has written `pca_adaptive_seed_candidates.csv`:

```powershell
python scripts\pca_hybrid_restart.py --config-module combined_2d --max-episode 10 --top-restarts 10
```

Equivalent explicit command:

```powershell
python scripts\hybrid_restart.py --config-module combined_2d --space pca --max-episode 10 --top-restarts 10
```

PCA-space outputs are written under:

```text
results_PPO/analysis_runs/<pca_run>/pca_hybrid_restart/
```

Use TICA-space restart selection for slow-coordinate diversity and PCA-space restart selection for large-amplitude structural diversity.

## Outputs

By default, outputs are written under:

```text
results_PPO/analysis_runs/<tica_run>/hybrid_restart/
```

For PCA-space selection, outputs are written under:

```text
results_PPO/analysis_runs/<pca_run>/pca_hybrid_restart/
```

Key outputs:

- `data/hybrid_restart_candidates.csv`
- `data/hybrid_transition_pathways.csv`
- `data/hybrid_restart_summary.json`
- `data/hybrid_restart_report.md`
- `figs/analysis/hybrid_two_transition_pathways_f_over_kT.png`
- `figs/analysis/hybrid_two_transition_pathways_unbiased_f_over_kT.png`
- `figs/analysis/hybrid_two_transition_pathways_biased_unbiased_side_by_side.png`
- `figs/analysis/pca_hybrid_two_transition_pathways_f_over_kT.png` for PCA-space selection
- `figs/analysis/pca_hybrid_two_transition_pathways_unbiased_f_over_kT.png` for PCA-space selection
- `figs/analysis/pca_hybrid_two_transition_pathways_biased_unbiased_side_by_side.png` for PCA-space selection
- `restart_candidates/seed_rank_*.pdb`

The restart candidate CSV records:

- hybrid rank
- original TICA rank
- PPO episode, if recoverable from the trajectory label
- trajectory label and resolved DCD path
- frame index
- `CV1`, `CV2`, `TIC1`, and `TIC2`
- TICA cluster
- adaptive score terms
- exported PDB path and export status

## Two-Pathway F/kT Plot

The TICA-space figure:

```text
figs/analysis/hybrid_two_transition_pathways_f_over_kT.png
```

The PCA-space figure:

```text
figs/analysis/pca_hybrid_two_transition_pathways_f_over_kT.png
```

Bias-reweighted approximate unbiased estimates:

```text
figs/analysis/hybrid_two_transition_pathways_unbiased_f_over_kT.png
figs/analysis/pca_hybrid_two_transition_pathways_unbiased_f_over_kT.png
```

Side-by-side sampled-biased vs bias-reweighted estimates:

```text
figs/analysis/hybrid_two_transition_pathways_biased_unbiased_side_by_side.png
figs/analysis/pca_hybrid_two_transition_pathways_biased_unbiased_side_by_side.png
```

uses a TICA-space histogram:

```text
F/kT = -ln(P)
```

with the minimum finite value shifted to zero. The plot is generated with `imshow`, and two selected transition pathways are overlaid in TICA Dim 0 vs TICA Dim 1 space.

The files with `unbiased` in the name are bias-reweighted estimates using saved episode bias metadata:

```text
weight = exp(beta * bias_energy)
```

They are not independent unbiased MD trajectories. Treat them as an approximate correction for the biased sampling landscape.

Pathway selection is intentionally simple and transparent:

1. Select the highest-ranked restart candidate as pathway 1 endpoint.
2. Prefer a second endpoint from a different TICA cluster.
3. If no different cluster is available, select the candidate with the largest TICA-space separation from pathway 1.
4. Draw each pathway from the same trajectory's early frames up to the selected endpoint.

The two overlaid pathways are therefore candidate transition traces, not independently proven mechanistic channels. They should be interpreted together with the exported PDBs, CV traces, and follow-up simulations.

For PCA, the same logic is applied in PC1/PC2 space. PCA-space pathways should be described as structurally diverse pathway candidates, not slow kinetic modes.

## Recommended 10-Episode Test

The user's current test preference is 10 episodes. A practical test sequence is:

```powershell
python main_2d.py
python scripts\tica.py --config-module combined_2d --max-traj 0 --lag 5
python scripts\hybrid_restart.py --config-module combined_2d --max-episode 10 --top-restarts 10
```

PCA-space companion test:

```powershell
python scripts\pca.py --config-module combined_2d --max-traj 0
python scripts\pca_hybrid_restart.py --config-module combined_2d --max-episode 10 --top-restarts 10
```

Then inspect:

```powershell
Get-ChildItem results_PPO\analysis_runs -Directory | Sort-Object LastWriteTime -Descending | Select-Object -First 5
```

Open the hybrid report from the TICA run:

```powershell
Get-Content results_PPO\analysis_runs\<tica_run>\hybrid_restart\data\hybrid_restart_report.md
```

If the 10-episode run exports reasonable, nonredundant restart PDBs and the two-pathway F/kT plot shows separated slow-coordinate traces, increase the episode count and rerun TICA/hybrid selection on the larger dataset.

## Follow-Up Use

The exported PDB candidates can be used for:

- unbiased MD validation
- weakly biased follow-up MD
- PPO evaluation from selected restart states
- manual structural inspection of pathway intermediates

For publication or reporting, describe this as:

```text
PPO-biased MD with TICA-based Adaptive-CVgen-style restart selection.
```

Do not describe it as a full Adaptive CVgen implementation unless the online CV-weight learning and round controller are added.
