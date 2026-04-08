# PCA Analysis

This document explains the dedicated phosphate-pathway PCA pipeline, its inputs, parameters, outputs, and how `PC1` and `PC2` are defined and interpreted.

For slow-coordinate analysis, use the companion TICA workflow in [TICA_ANALYSIS.md](./TICA_ANALYSIS.md). For automatic `CV2` candidate ranking, use [CV_SELECTION.md](./CV_SELECTION.md). For TICA/PCA-selected restart PDBs and two-pathway F/kT plotting, use [HYBRID_RESTART_SELECTION.md](./HYBRID_RESTART_SELECTION.md). For closed-loop PPO rounds using PCA/TICA-selected restart states, use [AUTOMATED_HYBRID_TRAINING.md](./AUTOMATED_HYBRID_TRAINING.md).

Primary source files:

- [analysis/pca_phosphate_pathway.py](../analysis/pca_phosphate_pathway.py)
- [scripts/pca.py](../scripts/pca.py)
- [scripts/pca_hybrid_restart.py](../scripts/pca_hybrid_restart.py)

## 1. Purpose

The current PCA workflow is a phosphate-pathway PCA built on residue-pair distance features.

The pipeline is designed to answer:

- which residues participate in the phosphate pathway
- how those residues rearrange across trajectories
- what the dominant collective modes are
- how those modes relate to `CV1` and `CV2`

PCA is still useful after adding TICA because it provides a variance-based structural decomposition. TICA answers a different question: which coordinates preserve time-lagged dynamical memory.

## 2. PCA Logic

Implemented structure:

1. Collect DCD trajectories.
2. Find protein residues that come within a cutoff of the phosphate selection across all sampled frames.
3. Collect residues that come within the input cutoff of the phosphate.
4. If too many residues are selected, tighten the cutoff until the selected set is at or below `max-residues`.
5. Build all residue-pair distance features among that selected residue set.
6. Compute those features for all sampled frames from all selected trajectories.
7. Fit `IncrementalPCA` on the pooled frame-by-feature matrix.
8. Save pooled PCA outputs and per-trajectory projections.
9. Define `PC1` and `PC2` explicitly using loadings and correlations to `CV1` and `CV2`.
10. Rank PCA-space restart candidates using CV progress, PCA cluster rarity, and PC1/PC2 displacement.
11. Plot two candidate transition pathways on a PC1/PC2 `free energy / kT` imshow map.

## 3. Feature Definition

### 3.1 Residue selection

Residues are found using:

```text
(<residue_sel>) and around <cutoff> (<phosphate_sel>)
```

Current defaults:

- `phosphate_sel = "segid HETC and not name H*"`
- `residue_sel = "protein"`
- `cutoff = 6.0`

### 3.2 Residue representation

Each selected residue is represented by:

- `com`: heavy-atom center of mass
- `ca`: CA atom if available

### 3.3 Distance features

If `N` residues are selected, the number of pair-distance features is:

```text
N_features = N * (N - 1) / 2
```

## 4. Inputs

The PCA pipeline consumes:

- topology from the configured PSF file
- DCD trajectories, typically from `results_PPO/dcd_trajs/*.dcd`
- CV atom indices from the config module for `CV1` and `CV2` correlation outputs

## 5. Main Entrypoint

Wrapper:

```powershell
python scripts\pca.py --config-module combined_2d
```

## 6. Recommended Command Order

### 6.1 First full run

```powershell
python scripts\pca.py --config-module combined_2d --max-traj 0
```

### 6.2 Practical full run with explicit settings

```powershell
python scripts\pca.py --config-module combined_2d --max-traj 0 --cutoff 6.0 --max-residues 60 --residue-repr com
```

### 6.3 Lower-cost run

```powershell
python scripts\pca.py --config-module combined_2d --max-traj 0 --contact-stride 5 --stride 5
```

### 6.4 Random subset test

```powershell
python scripts\pca.py --config-module combined_2d --max-traj 20 --sample random --seed 42
```

## 7. CLI Options

Defined in [analysis/pca_phosphate_pathway.py](../analysis/pca_phosphate_pathway.py):

| Option | Meaning |
|---|---|
| `--config-module` | Config module to import |
| `--top` | Topology file override |
| `--traj-glob` | DCD glob override |
| `--max-traj` | Maximum number of trajectories; `<= 0` means all |
| `--sample` | `first` or `random` selection mode |
| `--seed` | Random seed for trajectory sampling |
| `--align-sel` | Alignment selection before feature extraction |
| `--phosphate-sel` | Phosphate atom selection |
| `--residue-sel` | Candidate residue pool selection |
| `--cutoff` | Residue-phosphate cutoff in Angstrom |
| `--contact-stride` | Stride during residue discovery |
| `--stride` | Stride during PCA feature generation |
| `--max-residues` | Max residues retained after cutoff tightening |
| `--residue-repr` | `com` or `ca` |
| `--n-components` | Number of PCA components |
| `--batch-size` | `IncrementalPCA` batch size |
| `--bins` | Bins for pooled `PC1/PC2` FES |
| `--clusters` | PCA-space clusters for restart-candidate analysis |
| `--top-candidates` | Number of PCA restart-candidate rows to write |
| `--per-cluster-candidates` | Maximum candidates retained per PCA cluster |
| `--top-loadings` | Number of strongest loadings to plot |
| `--progress-weight` | Weight for CV1 progress in PCA candidate scoring |
| `--cv2-weight` | Weight for CV2 progress in PCA candidate scoring |
| `--novelty-weight` | Weight for PCA cluster rarity in candidate scoring |
| `--structural-mode-weight` | Weight for PC1/PC2 displacement from the initial PCA point |
| `--run` | Output run directory |
| `--runs-root` | Root directory for PCA runs |
| `--atom1`, `--atom2`, `--atom3`, `--atom4` | Optional CV atom overrides |

## 8. How `PC1` and `PC2` Are Defined

The pipeline defines `PC1` and `PC2` explicitly using:

1. full signed residue-pair loading tables
2. top-loading bar plots
3. correlations to `CV1`
4. correlations to `CV2`
5. a text report summarizing dominant residue-pair motions

This ties the first two components to:

- strongest residue-pair distance changes
- sign of those changes
- relation to the project CVs

## 9. Main Outputs

Outputs go under:

```text
results_PPO/analysis_runs/<timestamp>/
```

Key data outputs:

- `data/phosphate_pathway_summary.json`
- `data/phosphate_pathway_residues.csv`
- `data/phosphate_pathway_pairs.csv`
- `data/phosphate_pathway_scores_all.csv`
- `data/<traj_name>_pc_scores.csv`
- `data/pc1_pair_loadings.csv`
- `data/pc2_pair_loadings.csv`
- `data/pca_adaptive_seed_candidates.csv`
- `data/phosphate_pathway_pca_cluster_centers.npy`
- `data/phosphate_pathway_pca_cluster_labels.npy`
- `data/phosphate_pathway_pca_adaptive_score.npy`
- `data/phosphate_pathway_report.md`

Key figure outputs:

- `figs/analysis/phosphate_pathway_pca_scree.png`
- `figs/analysis/phosphate_pathway_pca_fes.png`
- `figs/analysis/phosphate_pathway_pc1_pc2_all_trajectories.png`
- `figs/analysis/pc1_top_pair_loadings.png`
- `figs/analysis/pc2_top_pair_loadings.png`
- `figs/analysis/pc1_vs_cv1.png`
- `figs/analysis/pc1_vs_cv2.png`
- `figs/analysis/pc2_vs_cv1.png`
- `figs/analysis/pc2_vs_cv2.png`
- `figs/analysis/pca_clusters_seed_candidates.png`
- `figs/analysis/pca_adaptive_score_map.png`
- `figs/analysis/pca_two_transition_pathways_f_over_kT.png`
- `figs/per_trajectory/<traj_name>_pc1_pc2_time.png`
- `figs/per_trajectory/<traj_name>_pc_timeseries.png`

## 10. PCA-Space Restart Candidate Scoring

The PCA workflow now mirrors the TICA candidate diagnostic, but the interpretation is variance-based rather than slow-mode-based:

```text
score =
  progress_weight * CV1_progress
+ cv2_weight * CV2_progress
+ novelty_weight * PCA_cluster_rarity
+ structural_mode_weight * PC1_PC2_displacement_from_initial
```

For the active Mg-O2 `CV2`, `CV2_PROGRESS_DIRECTION = "increase"` is respected, so increasing Mg-O2 separation is treated as forward progress.

Use the candidate table to find frames that combine Mg-phosphate progress with structural diversity in PCA space:

```text
results_PPO/analysis_runs/<pca_run>/data/pca_adaptive_seed_candidates.csv
```

The PCA two-pathway plot is:

```text
results_PPO/analysis_runs/<pca_run>/figs/analysis/pca_two_transition_pathways_f_over_kT.png
```

It uses:

```text
F/kT = -ln(P)
```

on a PC1/PC2 histogram and overlays two candidate pathway traces selected from different PCA clusters when possible.

## 11. PCA-Space Hybrid Restart Export

To export restart PDBs from PCA candidates, run after the PCA script:

```powershell
python scripts\pca_hybrid_restart.py --config-module combined_2d --max-episode 10 --top-restarts 10
```

Equivalent explicit command:

```powershell
python scripts\hybrid_restart.py --config-module combined_2d --space pca --max-episode 10 --top-restarts 10
```

Outputs go under:

```text
results_PPO/analysis_runs/<pca_run>/pca_hybrid_restart/
```

Key outputs:

- `data/hybrid_restart_candidates.csv`
- `data/hybrid_transition_pathways.csv`
- `data/hybrid_restart_report.md`
- `figs/analysis/pca_hybrid_two_transition_pathways_f_over_kT.png`
- `restart_candidates/seed_rank_*.pdb`

## 12. Interpretation Guidance

Use:

- scree plot for explained variance
- pooled `PC1/PC2` plots for global structure
- per-trajectory projections for episode evolution
- loading bar plots and CSVs for physical interpretation
- CV-correlation plots to determine whether a component is unbinding-like, CV2-like, or orthogonal
- PCA candidate plots to identify diverse high-progress structural frames
- PCA hybrid restart exports for follow-up structural inspection or validation

## 13. Practical Guidance

If too many residues are selected, the script now lowers the effective cutoff automatically until the residue count fits `--max-residues`.

If the run is too slow, increase:

- `--contact-stride`
- `--stride`

Or reduce:

- `--max-traj`
- `--max-residues`

## 14. Relationship to TICA

Use PCA and TICA together rather than treating one as a drop-in replacement:

- Use PCA to identify large-amplitude phosphate-pathway residue-pair rearrangements.
- Use TICA to identify slow metastable or barrier-crossing coordinates.
- Use PCA restart candidates for structural diversity.
- Use TICA restart candidates for slow-coordinate diversity.
- If a PCA component correlates with `CV1` but leading TICs do not, the hand-picked CV may explain large displacement but not slow kinetics.
- If a leading TIC has weak `CV1` or `CV2` correlation, inspect its feature loadings; it may indicate a missing pathway CV.
- Use [docs/TICA_ANALYSIS.md](./TICA_ANALYSIS.md) for the TICA command, outputs, and Adaptive-CVgen-inspired seed-candidate diagnostics.
- Use [docs/CV_SELECTION.md](./CV_SELECTION.md) when you want a trajectory-ranked `CV2` candidate before changing the training config.
