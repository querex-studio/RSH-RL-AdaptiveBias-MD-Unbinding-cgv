# PCA Analysis

This document explains the dedicated phosphate-pathway PCA pipeline, its inputs, parameters, outputs, and how `PC1` and `PC2` are defined and interpreted.

Primary source files:

- [analysis/pca_phosphate_pathway.py](../analysis/pca_phosphate_pathway.py)
- [scripts/pca.py](../scripts/pca.py)

## 1. Purpose

The current PCA workflow is a phosphate-pathway PCA built on residue-pair distance features.

The pipeline is designed to answer:

- which residues participate in the phosphate pathway
- how those residues rearrange across trajectories
- what the dominant collective modes are
- how those modes relate to `CV1` and `CV2`

## 2. PCA Logic

Implemented structure:

1. Collect DCD trajectories.
2. Find protein residues that come within a cutoff of the phosphate selection across all sampled frames.
3. Rank residues by contact frequency.
4. Keep the top `max-residues` residues if needed.
5. Build all residue-pair distance features among that selected residue set.
6. Compute those features for all sampled frames from all selected trajectories.
7. Fit `IncrementalPCA` on the pooled frame-by-feature matrix.
8. Save pooled PCA outputs and per-trajectory projections.
9. Define `PC1` and `PC2` explicitly using loadings and correlations to `CV1` and `CV2`.

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
| `--max-residues` | Max residues retained after ranking |
| `--residue-repr` | `com` or `ca` |
| `--n-components` | Number of PCA components |
| `--batch-size` | `IncrementalPCA` batch size |
| `--bins` | Bins for pooled `PC1/PC2` FES |
| `--top-loadings` | Number of strongest loadings to plot |
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
- `figs/per_trajectory/<traj_name>_pc1_pc2_time.png`
- `figs/per_trajectory/<traj_name>_pc_timeseries.png`

## 10. Interpretation Guidance

Use:

- scree plot for explained variance
- pooled `PC1/PC2` plots for global structure
- per-trajectory projections for episode evolution
- loading bar plots and CSVs for physical interpretation
- CV-correlation plots to determine whether a component is unbinding-like, CV2-like, or orthogonal

## 11. Practical Guidance

If too many residues are selected, lower:

- `--cutoff`
- `--max-residues`

If the run is too slow, increase:

- `--contact-stride`
- `--stride`

Or reduce:

- `--max-traj`
- `--max-residues`
