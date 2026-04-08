# CLI Workflow

This document lists the main commands in the recommended order of use.

## 1. Training

Run training:

```powershell
python main_2d.py
```

What it does:

1. trains PPO
2. writes DCD segments
3. writes per-episode CV CSVs
4. writes episode metadata JSON
5. writes bias profiles
6. auto-generates 2D reweighted episode plots if enabled

## 2. Inspect One Episode Directly

### 2.1 2D reweighted FES + bias + trajectory

```powershell
python analysis\plot_episode_bias_fes.py --episode 10 --temperature 300 --bins 120
```

### 2.2 3D unbiased FES + projected bias + trajectory

```powershell
python analysis\plot_episode_bias_fes_3d.py --episode 10 --temperature 300 --bins 140
```

### 2.3 Stacked slice plot

```powershell
python analysis\stacked_imshow_slices.py --episode 10 --temperature 300 --bins 180
```

## 3. Run Post-Processing on the Full Dataset

### 3.1 Basic post-processing

```powershell
python scripts\post_process.py --config-module combined_2d
```

### 3.2 With episode surfaces

```powershell
python scripts\post_process.py --config-module combined_2d --episode-surfaces
```

### 3.3 With episode surfaces and pooled trajectories

```powershell
python scripts\post_process.py --config-module combined_2d --episode-surfaces --all-trajectories
```

## 4. Run PCA

### 4.1 Full PCA on all trajectories

```powershell
python scripts\pca.py --config-module combined_2d --max-traj 0
```

### 4.2 Practical full PCA with explicit controls

```powershell
python scripts\pca.py --config-module combined_2d --max-traj 0 --cutoff 6.0 --max-residues 60 --residue-repr com
```

### 4.3 Lower-cost PCA

```powershell
python scripts\pca.py --config-module combined_2d --max-traj 0 --contact-stride 5 --stride 5
```

### 4.4 PCA-space restart export

Run after `scripts\pca.py`:

```powershell
python scripts\pca_hybrid_restart.py --config-module combined_2d --max-episode 10 --top-restarts 10
```

## 5. Rank CV2 Candidates

### 5.1 Full CV2 auto-selection

```powershell
python scripts\cv2_autoselect.py --config-module combined_2d --max-traj 0
```

### 5.2 Lower-cost CV2 auto-selection

```powershell
python scripts\cv2_autoselect.py --config-module combined_2d --max-traj 20 --stride 5
```

## 6. Run TICA

### 6.1 Full TICA on all trajectories

```powershell
python scripts\tica.py --config-module combined_2d --max-traj 0 --lag 5
```

### 6.2 Lower-cost TICA

```powershell
python scripts\tica.py --config-module combined_2d --max-traj 20 --contact-stride 5 --stride 5 --lag 5 --max-residues 40
```

### 6.3 Ligand-focused TICA

```powershell
python scripts\tica.py --config-module combined_2d --max-traj 0 --feature-set phosphate-distance --lag 5
```

## 7. Select Hybrid Restart Candidates

The hybrid workflow uses PPO trajectories for exploration and TICA/Adaptive-CVgen-style scoring for restart selection.

### 7.1 Ten-episode test selection

```powershell
python scripts\hybrid_restart.py --config-module combined_2d --max-episode 10 --top-restarts 10
```

### 7.2 Select from a specific TICA run

```powershell
python scripts\hybrid_restart.py --config-module combined_2d --tica-run results_PPO\analysis_runs\<tica_run> --max-episode 10 --top-restarts 10
```

### 7.3 Full restart selection after the test

```powershell
python scripts\hybrid_restart.py --config-module combined_2d --max-episode 0 --top-restarts 20
```

Main outputs:

- `hybrid_restart/data/hybrid_restart_candidates.csv`
- `hybrid_restart/data/hybrid_transition_pathways.csv`
- `hybrid_restart/data/hybrid_restart_report.md`
- `hybrid_restart/figs/analysis/hybrid_two_transition_pathways_f_over_kT.png`
- `hybrid_restart/restart_candidates/seed_rank_*.pdb`

For PCA-space restart export, use:

```powershell
python scripts\pca_hybrid_restart.py --config-module combined_2d --max-episode 10 --top-restarts 10
```

PCA-space outputs are written under:

- `pca_hybrid_restart/data/hybrid_restart_candidates.csv`
- `pca_hybrid_restart/data/hybrid_transition_pathways.csv`
- `pca_hybrid_restart/data/hybrid_restart_report.md`
- `pca_hybrid_restart/figs/analysis/pca_hybrid_two_transition_pathways_f_over_kT.png`
- `pca_hybrid_restart/restart_candidates/seed_rank_*.pdb`

## 8. Evaluate Trained Checkpoints

### 8.1 Evaluate all checkpoints

```powershell
python scripts\evaluate.py --config-module combined_2d
```

### 8.2 Evaluate one checkpoint

```powershell
python scripts\evaluate.py --config-module combined_2d --episode 50
```

### 8.3 Stronger long-run estimate

```powershell
python scripts\evaluate.py --config-module combined_2d --episodes 20
```

## 9. Automated Hybrid PPO Training

This runs the closed-loop workflow where PCA/TICA-selected restart PDBs affect later PPO rounds:

```powershell
python scripts\hybrid_auto.py --rounds 2 --initial-episodes 10 --episodes-per-round 10 --restart-fraction 0.30 --analysis-space both --top-restarts 10
```

Lower-cost test:

```powershell
python scripts\hybrid_auto.py --rounds 2 --initial-episodes 10 --episodes-per-round 5 --restart-fraction 0.30 --analysis-space tica --top-restarts 5 --max-traj 20 --stride 5 --contact-stride 5 --max-residues 40
```

Outputs:

- `results_PPO/hybrid_auto/<timestamp>/hybrid_auto_report.md`
- `results_PPO/analysis_runs/hybrid_auto_round_XX_pca/`
- `results_PPO/analysis_runs/hybrid_auto_round_XX_tica/`

## 10. Common Option Patterns

### Training

Training runtime control is edited directly in [main_2d.py](../main_2d.py):

- `RESUME`
- `resume_ep`
- `total_target`

### Episode plotting

Common options:

- `--episode`
- `--temperature`
- `--bins`

### Post-processing

Common options:

- `--config-module`
- `--run`
- `--runs-root`
- `--traj-glob`
- `--top`
- `--stride`

### PCA

Common options:

- `--config-module`
- `--max-traj`
- `--sample`
- `--seed`
- `--phosphate-sel`
- `--residue-sel`
- `--cutoff`
- `--contact-stride`
- `--stride`
- `--max-residues`
- `--residue-repr`
- `--n-components`
- `--top-loadings`
- `--clusters`
- `--top-candidates`
- `--per-cluster-candidates`
- `--progress-weight`
- `--cv2-weight`
- `--novelty-weight`
- `--structural-mode-weight`

### CV2 Auto-Selection

Common options:

- `--config-module`
- `--max-traj`
- `--sample`
- `--seed`
- `--stride`
- `--lag`
- `--mg-sel`
- `--phosphate-o-sel`
- `--max-pathway-atoms`
- `--top-candidates`

### TICA

Common options:

- `--config-module`
- `--max-traj`
- `--sample`
- `--seed`
- `--phosphate-sel`
- `--residue-sel`
- `--feature-set`
- `--lag`
- `--stride`
- `--max-residues`
- `--clusters`
- `--top-candidates`

### Hybrid Restart Selection

Common options:

- `--config-module`
- `--space`
- `--tica-run`
- `--runs-root`
- `--traj-glob`
- `--max-episode`
- `--top-restarts`
- `--bins`
- `--topology`
- `--atom-sel`
- `--no-export-pdb`
- `--temperature`

### Automated Hybrid Training

Common options:

- `--rounds`
- `--initial-episodes`
- `--episodes-per-round`
- `--restart-fraction`
- `--analysis-space`
- `--top-restarts`
- `--max-traj`
- `--stride`
- `--contact-stride`
- `--max-residues`
- `--lag`

### Evaluation

Common options:

- `--config-module`
- `--checkpoint-dir`
- `--checkpoint-glob`
- `--episode`
- `--episodes`
- `--target-center`
- `--target-half-width`
- `--run`

## 11. Recommended Real Workflow

1. Train:

```powershell
python main_2d.py
```

2. Inspect a few episodes directly:

```powershell
python analysis\plot_episode_bias_fes.py --episode 1
python analysis\plot_episode_bias_fes.py --episode 10
python analysis\stacked_imshow_slices.py --episode 10
```

3. Create a post-processing run:

```powershell
python scripts\post_process.py --config-module combined_2d --episode-surfaces --all-trajectories
```

4. Run PCA on all trajectories:

```powershell
python scripts\pca.py --config-module combined_2d --max-traj 0
```

5. Export PCA-space restart candidates:

```powershell
python scripts\pca_hybrid_restart.py --config-module combined_2d --max-episode 10 --top-restarts 10
```

6. Rank CV2 candidates:

```powershell
python scripts\cv2_autoselect.py --config-module combined_2d --max-traj 0
```

7. Run TICA slow-mode analysis:

```powershell
python scripts\tica.py --config-module combined_2d --max-traj 0 --lag 5
```

8. Select TICA-space hybrid restart candidates from the first 10 episodes:

```powershell
python scripts\hybrid_restart.py --config-module combined_2d --max-episode 10 --top-restarts 10
```

9. Evaluate saved checkpoints:

```powershell
python scripts\evaluate.py --config-module combined_2d --episodes 20
```

10. Optional automated hybrid PPO training:

```powershell
python scripts\hybrid_auto.py --rounds 2 --initial-episodes 10 --episodes-per-round 10 --restart-fraction 0.30 --analysis-space both --top-restarts 10
```

## 12. Related Documents

- [TRAINING_MODEL_AND_EPISODE_PLOTTING.md](./TRAINING_MODEL_AND_EPISODE_PLOTTING.md)
- [POST_PROCESSING_ANALYSIS.md](./POST_PROCESSING_ANALYSIS.md)
- [CV_SELECTION.md](./CV_SELECTION.md)
- [PCA_ANALYSIS.md](./PCA_ANALYSIS.md)
- [TICA_ANALYSIS.md](./TICA_ANALYSIS.md)
- [HYBRID_RESTART_SELECTION.md](./HYBRID_RESTART_SELECTION.md)
- [AUTOMATED_HYBRID_TRAINING.md](./AUTOMATED_HYBRID_TRAINING.md)
- [EVALUATION_REPORTING.md](./EVALUATION_REPORTING.md)
