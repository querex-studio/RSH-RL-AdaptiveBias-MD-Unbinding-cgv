# Version History

This file tracks project changes using semantic-style version numbering:

- `MAJOR`: incompatible workflow or modeling changes
- `MINOR`: new features or analysis capabilities
- `PATCH`: debugging, cleanup, and compatibility fixes

Within each version, changes are grouped as:

- `Major updates`
- `Modifications`
- `Minor updates`
- `Debugging`

## 7.3.0 - 2026-04-08

### Major updates

- Added closed-loop hybrid PPO training in [analysis/hybrid_auto_controller.py](./analysis/hybrid_auto_controller.py) with [scripts/hybrid_auto.py](./scripts/hybrid_auto.py).
- Added [docs/AUTOMATED_HYBRID_TRAINING.md](./docs/AUTOMATED_HYBRID_TRAINING.md) to document round-level PPO training with PCA/TICA-selected restart states.
- Extended [main_2d.py](./main_2d.py) and [combined_2d.py](./combined_2d.py) so selected restart PDBs can be mixed into later PPO rounds without changing the per-step PPO update.

### Modifications

- Hybrid training now supports the workflow: PPO round -> PCA/TICA analysis -> restart PDB export -> next PPO round with a configurable restart fraction.
- Restart episodes use the final target zone by default and do not promote the fresh-start curriculum, preventing late-stage restarts from falsely marking the original bound-state curriculum as solved.
- [analysis/hybrid_restart_selection.py](./analysis/hybrid_restart_selection.py) now writes bias-reweighted approximate unbiased F/kT pathway plots in addition to sampled biased F/kT plots.
- Added side-by-side sampled-biased vs bias-reweighted pathway plots for TICA and PCA hybrid restart outputs.
- Added `--temperature` to hybrid restart plotting for bias reweighting.

### Minor updates

- Updated [README.md](./README.md), [docs/CLI_WORKFLOW.md](./docs/CLI_WORKFLOW.md), [docs/HYBRID_RESTART_SELECTION.md](./docs/HYBRID_RESTART_SELECTION.md), [docs/TRAINING_MODEL_AND_EPISODE_PLOTTING.md](./docs/TRAINING_MODEL_AND_EPISODE_PLOTTING.md), and [docs/EVALUATION_REPORTING.md](./docs/EVALUATION_REPORTING.md) with the automated hybrid controller and bias-reweighted pathway outputs.
- Clarified that `unbiased` hybrid F/kT plots are bias-reweighted estimates from saved bias metadata, not independently generated unbiased MD trajectories.

### Debugging

- Fixed `--no-export-pdb` in [analysis/hybrid_restart_selection.py](./analysis/hybrid_restart_selection.py) so it still keeps candidate rows for pathway plotting while skipping PDB writes.

## 7.2.0 - 2026-04-07

### Major updates

- Added a phosphate-pathway TICA pipeline in [analysis/tica_phosphate_pathway.py](./analysis/tica_phosphate_pathway.py).
- Added [scripts/tica.py](./scripts/tica.py) as the CLI wrapper for slow-coordinate analysis.
- Added [docs/TICA_ANALYSIS.md](./docs/TICA_ANALYSIS.md) to document feature choices, commands, outputs, and interpretation.
- Added automatic CV2 candidate ranking in [analysis/cv2_autoselect.py](./analysis/cv2_autoselect.py) with [scripts/cv2_autoselect.py](./scripts/cv2_autoselect.py).
- Added [docs/CV_SELECTION.md](./docs/CV_SELECTION.md) to document Mg-phosphate oxygen and pathway CV2 selection.
- Added hybrid PPO/TICA restart selection in [analysis/hybrid_restart_selection.py](./analysis/hybrid_restart_selection.py) with [scripts/hybrid_restart.py](./scripts/hybrid_restart.py).
- Added [docs/HYBRID_RESTART_SELECTION.md](./docs/HYBRID_RESTART_SELECTION.md) to document Option-B restart selection, exported PDBs, and two-pathway TICA F/kT visualization.
- Added PCA-space restart-candidate diagnostics and two-pathway F/kT plotting directly to [analysis/pca_phosphate_pathway.py](./analysis/pca_phosphate_pathway.py).
- Added [scripts/pca_hybrid_restart.py](./scripts/pca_hybrid_restart.py) for PCA-space hybrid restart PDB export.

### Modifications

- TICA uses phosphate-aware features by default: selected residue-pair distances plus phosphate-to-residue distances.
- TICA outputs include eigenvalues, implied timescales, TIC score CSVs, top feature loadings, TIC-vs-CV correlations, FES plots, trajectory projections, and per-trajectory time series.
- Added an Adaptive-CVgen-inspired seed-candidate diagnostic that ranks frames using CV1 progress, optional CV2 progress, TICA cluster rarity, and slow-mode displacement.
- Added `CV2_PROGRESS_DIRECTION` in [combined_2d.py](./combined_2d.py) so distance-based `CV2` can represent either closing contacts or increasing unbinding-like distances.
- CV2 auto-selection prioritizes Mg to phosphate O2, Mg to each phosphate oxygen, smooth Mg-phosphate oxygen coordination, and nearby pathway atom distances.
- Updated the active `CV2` definition to Mg-phosphate O2 distance, atom `7799-7842`, with increase-mode progress and explicit axis labels.
- Episode XY diagnostics now distinguish the true final target from the curriculum stage target, preventing early-stage curriculum targets around `3.8-4.0 A` from being mislabeled as the final `7.5 A` target.
- CV2 auto-selection time-series plots now mark single-frame smoke-test data and annotate that DCD trajectories are required for a true time series.
- TICA candidate scoring now respects `CV2_PROGRESS_DIRECTION`, so the active Mg-O2 `CV2` is scored as an increasing-progress coordinate.
- Hybrid restart selection consumes `tica_adaptive_seed_candidates.csv` and `phosphate_pathway_tica_scores_all.csv`, ranks restart frames, exports top restart PDBs, and writes a `free energy / kT` imshow plot with two distinct transition pathway traces in TICA Dim 0 vs TICA Dim 1.
- PCA now writes `pca_adaptive_seed_candidates.csv`, `pca_clusters_seed_candidates.png`, `pca_adaptive_score_map.png`, and `pca_two_transition_pathways_f_over_kT.png`.
- PCA candidate scoring mirrors the TICA diagnostic while using PCA cluster rarity and PC1/PC2 displacement instead of TICA slow-mode displacement.
- [analysis/hybrid_restart_selection.py](./analysis/hybrid_restart_selection.py) now supports `--space pca`, so the same restart exporter can consume either TICA or PCA candidate tables.

### Minor updates

- Updated [README.md](./README.md) and [docs/CLI_WORKFLOW.md](./docs/CLI_WORKFLOW.md) with TICA commands and outputs.
- Updated [docs/PCA_ANALYSIS.md](./docs/PCA_ANALYSIS.md) to describe how PCA and TICA should be used together.
- Updated [docs/POST_PROCESSING_ANALYSIS.md](./docs/POST_PROCESSING_ANALYSIS.md) to place TICA in the downstream analysis order.
- Updated [docs/TRAINING_MODEL_AND_EPISODE_PLOTTING.md](./docs/TRAINING_MODEL_AND_EPISODE_PLOTTING.md) to clarify that TICA is downstream analysis, not online PPO training.
- Updated [docs/EVALUATION_REPORTING.md](./docs/EVALUATION_REPORTING.md) to connect checkpoint evaluation with optional TICA restart-candidate analysis.
- Updated [docs/TICA_ANALYSIS.md](./docs/TICA_ANALYSIS.md) with links to the related PCA, post-processing, and evaluation docs.
- Updated [docs/CLI_WORKFLOW.md](./docs/CLI_WORKFLOW.md), [README.md](./README.md), [docs/POST_PROCESSING_ANALYSIS.md](./docs/POST_PROCESSING_ANALYSIS.md), and [docs/TRAINING_MODEL_AND_EPISODE_PLOTTING.md](./docs/TRAINING_MODEL_AND_EPISODE_PLOTTING.md) with the CV2 auto-selection workflow.
- Updated [README.md](./README.md), [docs/CLI_WORKFLOW.md](./docs/CLI_WORKFLOW.md), [docs/TICA_ANALYSIS.md](./docs/TICA_ANALYSIS.md), [docs/PCA_ANALYSIS.md](./docs/PCA_ANALYSIS.md), [docs/POST_PROCESSING_ANALYSIS.md](./docs/POST_PROCESSING_ANALYSIS.md), [docs/CV_SELECTION.md](./docs/CV_SELECTION.md), [docs/TRAINING_MODEL_AND_EPISODE_PLOTTING.md](./docs/TRAINING_MODEL_AND_EPISODE_PLOTTING.md), [docs/HYBRID_RESTART_SELECTION.md](./docs/HYBRID_RESTART_SELECTION.md), and [docs/EVALUATION_REPORTING.md](./docs/EVALUATION_REPORTING.md) with the hybrid restart-selection workflow.

## 7.1.0 - 2026-03-26

### Major updates

- Added selectable Gaussian bias dimensionality through `BIAS_MODE` in [combined_2d.py](./combined_2d.py).
- Set the default bias mode to `1d`, while keeping the existing `2d` bias workflow available.
- Reworked episode diagnostics in [analysis/plot_episode_bias_fes.py](./analysis/plot_episode_bias_fes.py) into a four-panel XY diagnostic layout aligned with the visualization reference.

### Modifications

- Bias metadata export now records the active bias mode plus both CV1 and CV2 target zones.
- Episode bias surfaces now support both 1D and 2D bias logs; 1D bias is visualized as a CV1 surface extruded across CV2.
- PCA residue selection in [analysis/pca_phosphate_pathway.py](./analysis/pca_phosphate_pathway.py) no longer ranks residues by contact frequency.
- PCA now tightens the effective phosphate cutoff when needed to satisfy `--max-residues`, and reports the effective cutoff in the outputs.
- PCA plots and reports now show variance captured by `PC1`, `PC2`, and `PC1+PC2`.
- Episode diagnostics now write quantitative transition-state analysis outputs:
  - `episode_XXXX_transition_state_summary.json`
  - `episode_XXXX_transition_crossings.csv`

### Minor updates

- Updated [README.md](./README.md) to describe selectable 1D/2D biasing.
- Updated [docs/PCA_ANALYSIS.md](./docs/PCA_ANALYSIS.md) to document cutoff-adjusted residue selection.
- Updated [docs/TRAINING_MODEL_AND_EPISODE_PLOTTING.md](./docs/TRAINING_MODEL_AND_EPISODE_PLOTTING.md) to note the new bias-mode switch.

### Debugging

- Generalized post-processing utilities to read 1D and 2D bias logs without failing.
- Generalized stacked slice visualization code to use the shared bias-grid builder instead of assuming 2D-only bias entries.
- Preserved downstream bias-log compatibility by continuing to export the 7-column bias-log schema, with `None` placeholders for 1D-only fields.

## 7.0.0 - Baseline

### Major updates

- Established the PPO-driven 2D CV training environment and episode pipeline present before the current change set.
