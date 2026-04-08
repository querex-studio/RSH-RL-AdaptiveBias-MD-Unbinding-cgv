# TICA Analysis

This document explains the phosphate-pathway TICA pipeline for Mg-phosphate unbinding analysis.

Primary source files:

- [analysis/tica_phosphate_pathway.py](../analysis/tica_phosphate_pathway.py)
- [scripts/tica.py](../scripts/tica.py)

Related documents:

- [PCA_ANALYSIS.md](./PCA_ANALYSIS.md): variance-based phosphate-pathway structural modes
- [CV_SELECTION.md](./CV_SELECTION.md): automatic CV2 candidate ranking before TICA validation
- [HYBRID_RESTART_SELECTION.md](./HYBRID_RESTART_SELECTION.md): restart PDB export and two-pathway TICA F/kT plotting after TICA
- [AUTOMATED_HYBRID_TRAINING.md](./AUTOMATED_HYBRID_TRAINING.md): closed-loop PPO rounds using PCA/TICA-selected restart structures
- [POST_PROCESSING_ANALYSIS.md](./POST_PROCESSING_ANALYSIS.md): CV-space and episode-level post-processing before dimensionality reduction
- [EVALUATION_REPORTING.md](./EVALUATION_REPORTING.md): checkpoint success metrics before using TICA candidates for restart/evaluation planning

## Purpose

TICA is the slow-coordinate analysis layer. It complements the existing PCA workflow:

- PCA finds large-variance phosphate-pathway motions.
- TICA finds time-lagged slow modes that are more appropriate for barrier crossing, metastable states, and unbinding pathways.

The PCA workflow now has matching candidate and two-pathway F/kT plots. Use PCA-space candidates for structural diversity and TICA-space candidates for slow-coordinate diversity.

The shared `Adaptive_CVgen-main` folder does not include the full simulator or original reward optimizer. This project therefore adds a reproducible TICA pipeline and an Adaptive-CVgen-inspired seed-candidate diagnostic without claiming to reproduce the paper's full `W * Theta` reward model.

## Feature Definition

The default feature set is `combined`:

1. residue-pair distances among protein residues that approach the phosphate selection
2. distances from the phosphate center to each selected residue

This is more suitable for Mg-phosphate unbinding than the protein-folding notebook's generic C-alpha contact features because it keeps the ligand/pathway geometry explicit.

Supported `--feature-set` values:

| Value | Meaning |
|---|---|
| `combined` | residue-pair distances plus phosphate-residue distances |
| `residue-pair` | only selected residue-pair distances |
| `phosphate-distance` | only phosphate-to-selected-residue distances |
| `cv-only` | only configured `CV1` and `CV2`; useful for quick tests |
| `combined-with-cv` | combined feature set plus configured `CV1` and `CV2` |

## Main Commands

Full TICA on all trajectories:

```powershell
python scripts\tica.py --config-module combined_2d --max-traj 0 --lag 5
```

Lower-cost run:

```powershell
python scripts\tica.py --config-module combined_2d --max-traj 20 --contact-stride 5 --stride 5 --lag 5 --max-residues 40
```

Ligand-focused feature run:

```powershell
python scripts\tica.py --config-module combined_2d --max-traj 0 --feature-set phosphate-distance --lag 5
```

Recommended order after training:

1. `scripts\post_process.py` for CV-space and episode-level diagnostics
2. `scripts\cv2_autoselect.py` for automatic CV2 candidate ranking
3. `scripts\pca.py` for variance-based phosphate-pathway structural modes
4. `scripts\tica.py` for slow-mode analysis and seed-candidate ranking
5. `scripts\hybrid_restart.py` for Optional-B restart PDB export and two-pathway F/kT visualization

## Main Outputs

Outputs go under:

```text
results_PPO/analysis_runs/<timestamp>/
```

Key data outputs:

- `data/phosphate_pathway_tica_summary.json`
- `data/phosphate_pathway_tica_report.md`
- `data/phosphate_pathway_tica_scores_all.csv`
- `data/phosphate_pathway_tica_features.csv`
- `data/phosphate_pathway_tica_residues.csv`
- `data/phosphate_pathway_tica_eigenvalues.npy`
- `data/phosphate_pathway_tica_timescales_ps.npy`
- `data/tic1_feature_loadings.csv`
- `data/tic2_feature_loadings.csv`
- `data/tica_adaptive_seed_candidates.csv`
- `hybrid_restart/data/hybrid_restart_candidates.csv` after running `scripts\hybrid_restart.py`
- `hybrid_restart/data/hybrid_transition_pathways.csv` after running `scripts\hybrid_restart.py`

Key figures:

- `figs/analysis/phosphate_pathway_tica_fes.png`
- `figs/analysis/phosphate_pathway_tic1_tic2_all_trajectories.png`
- `figs/analysis/tic1_top_feature_loadings.png`
- `figs/analysis/tic2_top_feature_loadings.png`
- `figs/analysis/tic1_vs_cv1.png`
- `figs/analysis/tic1_vs_cv2.png`
- `figs/analysis/tic2_vs_cv1.png`
- `figs/analysis/tic2_vs_cv2.png`
- `figs/analysis/tica_clusters_seed_candidates.png`
- `figs/analysis/tica_adaptive_score_map.png`
- `hybrid_restart/figs/analysis/hybrid_two_transition_pathways_f_over_kT.png` after running `scripts\hybrid_restart.py`

## Adaptive Seed-Candidate Diagnostic

The candidate score is:

```text
score =
  progress_weight * CV1_progress
+ cv2_weight * CV2_progress
+ novelty_weight * cluster_rarity
+ slow_mode_weight * TICA_distance_from_initial
```

This is useful for ranking frames that may be good restart structures or for diagnosing whether PPO is exploring useful slow-coordinate regions. It is not a replacement for the PPO reward and not the original Adaptive CVgen `W * Theta` reward.

Default weights:

| Option | Default |
|---|---:|
| `--progress-weight` | `1.0` |
| `--cv2-weight` | `0.35` |
| `--novelty-weight` | `0.25` |
| `--slow-mode-weight` | `0.25` |

## Interpretation

Use TICA together with the existing CV and FES plots:

- If TIC1 or TIC2 correlates strongly with `CV1`, the slow mode likely tracks Mg-phosphate unbinding.
- If a TIC correlates strongly with `CV2`, the auxiliary corridor is dynamically meaningful.
- If leading TICs have weak CV correlations but strong pathway-residue loadings, the current hand-picked CVs may be missing a slow rearrangement.
- If top seed candidates come from rare clusters with high CV1 progress, they are good candidates for restart/evaluation.
- If candidates have high novelty but poor CV1 progress, those frames may represent off-pathway exploration.

## Hybrid Restart Follow-Up

After TICA, export the 10-episode test restart set:

```powershell
python scripts\hybrid_restart.py --config-module combined_2d --max-episode 10 --top-restarts 10
```

This creates restart PDBs and overlays two distinct candidate transition pathways on a TICA-space `free energy / kT` imshow plot. See [HYBRID_RESTART_SELECTION.md](./HYBRID_RESTART_SELECTION.md) for the full workflow and interpretation limits.
