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

## 5. Evaluate Trained Checkpoints

### 5.1 Evaluate all checkpoints

```powershell
python scripts\evaluate.py --config-module combined_2d
```

### 5.2 Evaluate one checkpoint

```powershell
python scripts\evaluate.py --config-module combined_2d --episode 50
```

### 5.3 Stronger long-run estimate

```powershell
python scripts\evaluate.py --config-module combined_2d --episodes 20
```

## 6. Common Option Patterns

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

## 7. Recommended Real Workflow

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

5. Evaluate saved checkpoints:

```powershell
python scripts\evaluate.py --config-module combined_2d --episodes 20
```

## 8. Related Documents

- [TRAINING_MODEL_AND_EPISODE_PLOTTING.md](./TRAINING_MODEL_AND_EPISODE_PLOTTING.md)
- [POST_PROCESSING_ANALYSIS.md](./POST_PROCESSING_ANALYSIS.md)
- [PCA_ANALYSIS.md](./PCA_ANALYSIS.md)
- [EVALUATION_REPORTING.md](./EVALUATION_REPORTING.md)
