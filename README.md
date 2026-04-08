# PPO-Driven 2D Biasing for Mg-Phosphate Unbinding

## Project Summary

This project trains a PPO agent to bias OpenMM molecular dynamics in a 2D collective-variable space while keeping Mg-phosphate unbinding as the main scientific objective.

The two collective variables are distance-based:

- `CV1`: Mg-phosphate P distance, atom `7799` to atom `7840`
- `CV2`: Mg-phosphate O2 distance, atom `7799` to atom `7842`

Current implementation highlights:

- persistent OpenMM `Simulation` per episode with in-place force updates
- selectable Gaussian bias mode: default `1D` on `CV1`, optional `2D` in `(CV1, CV2)`
- fresh-start episodic training for cleaner RL benchmarking
- CV1 milestone curriculum for training progress
- automatic episode metadata, bias profiles, and episode-level analysis plots
- dedicated phosphate-pathway PCA and TICA pipelines
- automatic CV2 candidate ranking for Mg-phosphate oxygen and pathway CVs
- Adaptive-CVgen-inspired TICA seed-candidate diagnostics for restart/evaluation planning
- hybrid PPO/TICA restart selection with exported restart PDBs and two-pathway TICA F/kT plotting
- PCA-space restart-candidate diagnostics and two-pathway PCA F/kT plotting
- automated round-level hybrid PPO training with PCA/TICA-selected restart states

## Project Logic

High-level flow:

1. [main_2d.py](./main_2d.py) runs PPO training.
2. [combined_2d.py](./combined_2d.py) defines the environment, PPO agent, reward, curriculum, and bias logic.
3. Training writes DCD segments, CV trajectories, metadata, bias profiles, checkpoints, and optional episode plots.
4. Analysis scripts re-read those outputs for post-processing, episode visualization, PCA, and TICA.
5. Optional hybrid automation uses PCA/TICA-selected restart PDBs to start a fraction of later PPO episodes.

Current training philosophy:

- every episode starts from the initial structure by default
- CV1 is the primary progress and reward coordinate
- CV2 acts as an auxiliary corridor / shaping coordinate
- the physical final target remains the full unbinding target

## Inputs

Core input files:

- `step3_input.psf`
- `traj_0.restart.pdb`
- `toppar.str`

These are resolved by the configuration in [combined_2d.py](./combined_2d.py).

## Outputs

Training outputs are written under `results_PPO/`.

Important locations:

- `results_PPO/dcd_trajs/`: per-action DCD segments
- `results_PPO/full_trajectories/`: per-episode CV CSVs and plots
- `results_PPO/episode_meta/`: per-episode bias logs and metadata
- `results_PPO/bias_profiles/`: saved 2D bias surfaces
- `results_PPO/episode_pdbs/`: optional end-of-episode structures
- `results_PPO/checkpoints/`: PPO checkpoints
- `results_PPO/analysis_runs/`: post-processing, PCA, and TICA outputs

## Main Code Structure

- [combined_2d.py](./combined_2d.py): PPO agent, OpenMM environment, selectable 1D/2D Gaussian bias logic, rewards, curriculum, export helpers
- [main_2d.py](./main_2d.py): training loop, curriculum promotion, checkpointing, automatic episode plotting
- [analysis/](./analysis): plotting, post-processing, PCA, TICA, hybrid restart selection, and analysis utilities
- [scripts/](./scripts): lightweight wrappers for analysis entrypoints, including TICA/PCA hybrid restart export
- [docs/](./docs): user documentation and workflow guides

## Dependencies

Core training:

- `openmm`
- `torch`
- `numpy`
- `matplotlib`
- `tqdm`

Analysis:

- `MDAnalysis`
- `scikit-learn`
- `scipy`

## Documentation

Detailed instructions have been moved into dedicated documents:

- [docs/TRAINING_MODEL_AND_EPISODE_PLOTTING.md](./docs/TRAINING_MODEL_AND_EPISODE_PLOTTING.md)
- [docs/POST_PROCESSING_ANALYSIS.md](./docs/POST_PROCESSING_ANALYSIS.md)
- [docs/PCA_ANALYSIS.md](./docs/PCA_ANALYSIS.md)
- [docs/TICA_ANALYSIS.md](./docs/TICA_ANALYSIS.md)
- [docs/CV_SELECTION.md](./docs/CV_SELECTION.md)
- [docs/HYBRID_RESTART_SELECTION.md](./docs/HYBRID_RESTART_SELECTION.md)
- [docs/AUTOMATED_HYBRID_TRAINING.md](./docs/AUTOMATED_HYBRID_TRAINING.md)
- [docs/EVALUATION_REPORTING.md](./docs/EVALUATION_REPORTING.md)
- [docs/CLI_WORKFLOW.md](./docs/CLI_WORKFLOW.md)

Use those files for:

- configuration parameters and training logic
- episode plotting and training behavior
- post-processing commands and outputs
- PCA and TICA commands, outputs, and interpretation
- CV2 auto-selection and Mg-phosphate oxygen candidate ranking
- hybrid PPO/TICA restart selection and two-pathway F/kT plotting
- PCA-space candidate scoring and PCA hybrid restart export
- closed-loop hybrid training across PPO rounds
- checkpoint evaluation reports and dashboards
- command order for a normal workflow

## Minimal Quick Start

Train:

```powershell
python main_2d.py
```

Post-process:

```powershell
python scripts\post_process.py --config-module combined_2d
```

Rank CV2 candidates:

```powershell
python scripts\cv2_autoselect.py --config-module combined_2d --max-traj 0
```

Run PCA:

```powershell
python scripts\pca.py --config-module combined_2d --max-traj 0
```

Export PCA-space hybrid restart candidates:

```powershell
python scripts\pca_hybrid_restart.py --config-module combined_2d --max-episode 10 --top-restarts 10
```

Run TICA:

```powershell
python scripts\tica.py --config-module combined_2d --max-traj 0 --lag 5
```

Select hybrid restart candidates from the TICA run:

```powershell
python scripts\hybrid_restart.py --config-module combined_2d --max-episode 10 --top-restarts 10
```

Run automated hybrid PPO training:

```powershell
python scripts\hybrid_auto.py --rounds 2 --initial-episodes 10 --episodes-per-round 10 --restart-fraction 0.30 --analysis-space both --top-restarts 10
```

Evaluate checkpoints:

```powershell
python scripts\evaluate.py --config-module combined_2d
```

## Notes

- Do not modify anything under `Archive/`.
- Avoid editing `results_PPO/` manually except to create new analysis outputs.
- If a run shows MD failures or NaN warnings, do not resume from a corrupted checkpoint.
