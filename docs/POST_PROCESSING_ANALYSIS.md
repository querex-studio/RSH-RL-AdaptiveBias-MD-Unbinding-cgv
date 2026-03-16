# Post-Processing Analysis

This document covers post-processing analysis after training: trajectory aggregation, episode-surface plotting, and run-level output organization.

Primary source files:

- [analysis/post_process.py](../analysis/post_process.py)
- [scripts/post_process.py](../scripts/post_process.py)

## 1. Purpose

Post-processing is the step after training where saved trajectories and episode metadata are re-read to produce analysis runs under `results_PPO/analysis_runs/`.

It is separate from training because:

- training writes raw or episode-level outputs
- post-processing aggregates or re-renders them in a new analysis run

## 2. Inputs

The post-processing pipeline mainly consumes:

- DCD files from `results_PPO/dcd_trajs/`
- topology from the configured PSF/PDB
- episode CV CSVs from `results_PPO/full_trajectories/`
- episode metadata JSON files from `results_PPO/episode_meta/`

## 3. Main Entrypoint

Wrapper:

```powershell
python scripts\post_process.py --config-module combined_2d
```

The wrapper calls [analysis/post_process.py](../analysis/post_process.py).

## 4. Recommended Use Order

### 4.1 Basic run-level post-processing

```powershell
python scripts\post_process.py --config-module combined_2d
```

### 4.2 Add per-episode surface plots

```powershell
python scripts\post_process.py --config-module combined_2d --episode-surfaces
```

### 4.3 Add pooled all-trajectory plot

```powershell
python scripts\post_process.py --config-module combined_2d --episode-surfaces --all-trajectories
```

## 5. CLI Options

Defined in [analysis/post_process.py](../analysis/post_process.py):

| Option | Meaning |
|---|---|
| `--config-module` | Config module to import, usually `combined_2d` |
| `--run` | Existing analysis run directory to reuse |
| `--runs-root` | Root directory containing analysis runs |
| `--traj-glob` | Override DCD file pattern |
| `--top` | Override topology file |
| `--max-traj-plots` | Maximum number of individual trajectory plots |
| `--stride` | Frame stride while reading trajectories |
| `--episode-surfaces` | Generate per-episode FES+bias surface plots |
| `--max-episode-plots` | Maximum number of episode-surface plots |
| `--episode-plot-stride` | Stride for episode CV samples in episode-surface plots |
| `--all-trajectories` | Create a combined plot of all trajectories |
| `--all-traj-stride` | Stride for the combined all-trajectories plot |

## 6. What the Script Produces

A post-processing run writes to:

```text
results_PPO/analysis_runs/<timestamp>/
```

Typical contents include:

- run metadata JSON
- copied or derived CV arrays
- per-trajectory plots
- combined trajectories plots
- optionally per-episode surface plots

## 7. Relationship to Episode Plotting Scripts

Use direct episode plotting scripts when you need one specific episode:

- [analysis/plot_episode_bias_fes.py](../analysis/plot_episode_bias_fes.py)
- [analysis/plot_episode_bias_fes_3d.py](../analysis/plot_episode_bias_fes_3d.py)
- [analysis/stacked_imshow_slices.py](../analysis/stacked_imshow_slices.py)

Use post-processing when you need:

- a new aggregated analysis run
- many trajectories processed together
- a standard output bundle

## 8. Important Notes

### 8.1 Config-module dependency

The script imports the chosen config module to resolve:

- topology path
- trajectory directory
- CV atom indices
- output roots

### 8.2 Legacy metrics support

The script still contains support for older metrics and FES files. Not every function is central to the current PPO workflow.

### 8.3 Episode-surface plotting dependency

Per-episode surface generation depends on episode-level CSV and metadata files being present and consistent.

## 9. Practical Commands

Basic:

```powershell
python scripts\post_process.py --config-module combined_2d
```

Reuse an existing run directory:

```powershell
python scripts\post_process.py --config-module combined_2d --run results_PPO\analysis_runs\my_run
```

Limit cost:

```powershell
python scripts\post_process.py --config-module combined_2d --stride 5 --max-traj-plots 20
```

Fuller visualization pass:

```powershell
python scripts\post_process.py --config-module combined_2d --episode-surfaces --all-trajectories --all-traj-stride 5
```
