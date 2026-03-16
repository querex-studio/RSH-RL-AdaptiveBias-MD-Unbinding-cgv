# Training Model and Episode Plotting

This document explains the training model, the main configuration parameters in the training code, the reward and bias logic, and the episode-level plotting tools.

Primary source files:

- [combined_2d.py](../combined_2d.py)
- [main_2d.py](../main_2d.py)
- [analysis/plot_episode_bias_fes.py](../analysis/plot_episode_bias_fes.py)
- [analysis/plot_episode_bias_fes_3d.py](../analysis/plot_episode_bias_fes_3d.py)
- [analysis/stacked_imshow_slices.py](../analysis/stacked_imshow_slices.py)

## 1. Training Model Overview

The project trains a PPO agent to bias OpenMM MD in a 2D collective-variable space:

- `CV1`: main unbinding coordinate
- `CV2`: auxiliary coordinate / corridor coordinate

High-level flow:

1. `main_2d.py` creates `PPOAgent` and `ProteinEnvironmentRedesigned`.
2. Each episode resets to the initial structure by default.
3. PPO selects one discrete action per RL step.
4. The action is converted into up to 3 2D Gaussian hills.
5. The cumulative bias is applied in a persistent OpenMM `Simulation`.
6. MD is propagated.
7. Reward is computed from CV1 progress, milestones, stability, and optional CV2 shaping.
8. PPO updates from the collected rollout.

## 2. Main Configuration Location

Most model and training parameters are defined near the top of [combined_2d.py](../combined_2d.py).

Main run control is in [main_2d.py](../main_2d.py).

## 3. Core Training Parameters

### 3.1 Input structure and reproducibility

Defined in [combined_2d.py](../combined_2d.py):

| Parameter | Meaning |
|---|---|
| `SEED` | Global seed for Python, NumPy, and PyTorch |
| `psf_file` | Topology PSF file |
| `pdb_file` | Initial coordinates used for fresh-start reset |
| `toppar_file` | CHARMM parameter list file |

Rules:

- If `pdb_file` changes, verify `CURRENT_DISTANCE` and `CURRENT_DISTANCE2`.
- If topology changes, verify CV atom indices.

### 3.2 Collective variables

Defined in [combined_2d.py](../combined_2d.py):

| Parameter | Meaning |
|---|---|
| `ATOM1_INDEX`, `ATOM2_INDEX` | CV1 atom pair |
| `ATOM3_INDEX`, `ATOM4_INDEX` | CV2 atom pair |
| `ATOM_PAIRS` | Optional override pair list |
| `CV1_LABEL`, `CV2_LABEL` | Plot labels |

Distance calculation:

```text
CV1 = ||r(atom1) - r(atom2)|| * 10
CV2 = ||r(atom3) - r(atom4)|| * 10
```

OpenMM coordinates are in nm, so the factor `10` converts to Angstrom.

### 3.3 Targets and target zones

CV1 target parameters:

| Parameter | Meaning |
|---|---|
| `CURRENT_DISTANCE` | Fresh-start CV1 anchor |
| `FINAL_TARGET` | Final CV1 target center |
| `TARGET_CENTER` | Alias of final target center |
| `TARGET_ZONE_HALF_WIDTH` | Final target half-width |
| `TARGET_MIN`, `TARGET_MAX` | Derived final target bounds |

CV2 corridor parameters:

| Parameter | Meaning |
|---|---|
| `CURRENT_DISTANCE2` | Fresh-start CV2 anchor |
| `FINAL_TARGET2` | CV2 target center |
| `TARGET2_CENTER` | Active CV2 corridor center |
| `TARGET2_ZONE_HALF_WIDTH` | CV2 corridor half-width |
| `TARGET2_MIN`, `TARGET2_MAX` | Derived CV2 corridor bounds |

Rules:

- `CURRENT_DISTANCE` and `CURRENT_DISTANCE2` are used in state and progress formulas.
- Keep the final physical target separate from curriculum targets.

### 3.4 Curriculum and milestone logic

Defined in [combined_2d.py](../combined_2d.py) and used in [main_2d.py](../main_2d.py).

Parameters:

| Parameter | Meaning |
|---|---|
| `DISTANCE_INCREMENTS` | CV1 milestone list and stage centers |
| `CURRICULUM_SUCCESS_WINDOW` | Window size for stage promotion |
| `CURRICULUM_PROMOTION_THRESHOLD` | Success fraction required to promote |
| `CURRICULUM_ZONE_SCALE` | Scales stage-zone width by distance from initial state |
| `CURRICULUM_MIN_HALF_WIDTH` | Minimum stage-zone half-width |

Derived stage target list:

```text
curriculum_targets = DISTANCE_INCREMENTS plus FINAL_TARGET if missing
```

Stage-zone half-width:

```text
gap = max(0, target_center_A - CURRENT_DISTANCE)
stage_half_width = clip(
    gap * CURRICULUM_ZONE_SCALE,
    CURRICULUM_MIN_HALF_WIDTH,
    TARGET_ZONE_HALF_WIDTH
)
```

Rules:

- The first stage zone must not already contain the initial CV1.
- Promotion should reflect repeated fresh-start success, not one lucky episode.
- The milestone ladder is intentionally denser around the historical `5–6 A` barrier so reward and curriculum signals do not go sparse there.
- Curriculum stages are shaping targets only.
- The true phase-2 stabilization target and terminal success zone remain the final physical target (`TARGET_CENTER`, `TARGET_MIN`, `TARGET_MAX`).

### 3.5 Fresh-start training mode

Important flags in [combined_2d.py](../combined_2d.py):

| Parameter | Meaning |
|---|---|
| `TRAIN_FRESH_START_EVERY_EPISODE` | Reset each episode from the initial structure |
| `CARRY_STATE_ACROSS_EPISODES` | Legacy carry-state behavior |
| `PERSIST_LOCKS_ACROSS_EPISODES` | Legacy lock carryover |

Current intended behavior:

- `TRAIN_FRESH_START_EVERY_EPISODE = True`
- episodes are independent for RL benchmarking
- curriculum is implemented in reward/target logic, not by continuing coordinates

### 3.6 Observation and action space

Defined in [combined_2d.py](../combined_2d.py).

Observation parameters:

| Parameter | Meaning |
|---|---|
| `STATE_SIZE` | Observation dimension |
| `OBS_CLIP` | Clip after observation normalization |

Action parameters:

| Parameter | Meaning |
|---|---|
| `AMP_BINS` | Discrete amplitude values |
| `DX_BINS` | Discrete CV1 offset bins |
| `DY_BINS` | Discrete CV2 offset bins |
| `SIGMA_BINS` | Discrete width bins |
| `ACTION_SIZE` | Number of discrete actions |
| `MIN_AMP`, `MAX_AMP` | Internal amplitude clamp |
| `APPLIED_MAX_AMP` | Final per-hill amplitude cap written into the episode bias |
| `MIN_WIDTH`, `MAX_WIDTH` | Width clamp |
| `IN_ZONE_MAX_AMP` | Action masking threshold in-zone |
| `MAX_ESCALATION_FACTOR` | Max stall-based amplitude escalation |

Action-space size:

```text
ACTION_SIZE = len(AMP_BINS) * len(DX_BINS) * len(DY_BINS) * len(SIGMA_BINS)
```

Current state vector from `get_state()`:

```text
state[0]  = d1 / FINAL_TARGET
state[1]  = max(0, overall_progress)
state[2]  = |d1 - active_target_center| / active_target_half_width
state[3]  = clip(overall_progress, 0, 1)
state[4]  = trend1 / 0.1
state[5]  = 1 if CV1 is inside active target zone else 0
state[6]  = 1 if no_improve_counter > 0 else 0
state[7]  = stability
state[8]  = d2 / cv2_center
state[9]  = trend2 / 0.1
state[10] = 1 if CV2 is inside its corridor else 0
```

Supporting formulas:

```text
trend1 = (d1[t] - d1[t-2]) / 2
trend2 = (d2[t] - d2[t-2]) / 2
stability = 1 / (1 + std(last_5_cv1_values))
```

```text
p1 = clip((d1 - CURRENT_DISTANCE) / (active_target_center - CURRENT_DISTANCE), 0, 1)
p2 = clip((CURRENT_DISTANCE2 - d2) / (CURRENT_DISTANCE2 - FINAL_TARGET2), 0, 1)
overall_progress = p1
```

Rules:

- `STATE_SIZE` must match the actual state length.
- If state order changes, update action masking logic in `PPOAgent`.

### 3.7 PPO hyperparameters

Defined in [combined_2d.py](../combined_2d.py):

| Parameter | Meaning |
|---|---|
| `N_STEPS` | Rollout length needed before PPO update |
| `BATCH_SIZE` | PPO minibatch size |
| `N_EPOCHS` | PPO epochs per update |
| `GAMMA` | Discount factor |
| `GAE_LAMBDA` | GAE lambda |
| `CLIP_RANGE` | PPO clip epsilon |
| `ENT_COEF` | Entropy weight |
| `VF_COEF` | Value loss weight |
| `MAX_GRAD_NORM` | Gradient clip norm |
| `LR` | Learning rate |
| `PPO_TARGET_KL` | KL stability target |

Network architecture:

- Actor MLP: `256 -> 128 -> 64 -> ACTION_SIZE`
- Critic MLP: `256 -> 128 -> 64 -> 1`

### 3.8 MD and episode timing

Defined in [combined_2d.py](../combined_2d.py):

| Parameter | Meaning |
|---|---|
| `MAX_ACTIONS_PER_EPISODE` | RL actions per episode |
| `stepsize` | Langevin integrator timestep |
| `fricCoef` | Default friction |
| `T` | Temperature |
| `propagation_step` | MD steps per RL action |
| `dcdfreq_mfpt` | Sampling interval within one action |

Approximate simulated time per episode:

```text
t_episode_ps = MAX_ACTIONS_PER_EPISODE * propagation_step * stepsize_ps
```

With current defaults:

```text
10 * 4000 * 0.001 ps = 40 ps
```

### 3.9 NaN recovery and failure control

Defined in [combined_2d.py](../combined_2d.py):

| Parameter | Meaning |
|---|---|
| `MAX_INTEGRATOR_RETRIES` | Retry count for NaN recovery |
| `MIN_STEPSIZE` | Minimum timestep during recovery |
| `MD_FAILURE_PENALTY` | Reward penalty when MD fails |

Failure reward:

```text
reward = MD_FAILURE_PENALTY + STEP_PENALTY
```

### 3.10 Reward shaping

Defined in [combined_2d.py](../combined_2d.py):

| Parameter | Meaning |
|---|---|
| `PROGRESS_REWARD` | Reward per A of outward CV1 motion |
| `MILESTONE_REWARD` | Bonus when crossing milestones |
| `BACKTRACK_PENALTY` | Penalty for CV1 inward motion |
| `VELOCITY_BONUS` | Flat bonus if CV1 moved outward |
| `STEP_PENALTY` | Flat per-action penalty |
| `PHASE2_TOL` | Tight target tolerance |
| `CENTER_GAIN` | Phase-2 bowl reward gain |
| `STABILITY_STEPS` | Required in-zone steps for stable success |
| `CONSISTENCY_BONUS` | Phase-2 stability bonus |
| `CV2_DEVIATION_PENALTY` | Penalty outside CV2 corridor |
| `CV2_PROGRESS_REWARD_SCALE` | Weight on CV2 progress shaping |

Phase-2 bowl reward:

```text
err = |d - center|
if err >= half_width: reward = 0
else: reward = gain * (1 - (err / half_width)^2)
```

Target logic:

```text
curriculum target:
- used for promotion bookkeeping (`episode_target_hit`)
- does not terminate the episode
- does not trigger phase 2

final target:
- used in state features
- used for bias progress scaling
- used for phase-2 switching
- used for stable success / episode termination
```

## 4. 2D Gaussian Bias Logic

Location: `ProteinEnvironmentRedesigned.smart_progressive_bias()` in [combined_2d.py](../combined_2d.py)

Bias construction parameters:

| Parameter | Meaning |
|---|---|
| `MAX_GAUSSIANS_PER_ACTION` | Max hills deposited per action |
| `MAX_BIAS_TERMS_PER_EPISODE` | Total preallocated bias slots |
| `RIBBON_CV1_NEAR_OFFSET_A` | Near-behind offset for the first hill |
| `RIBBON_CV1_NEAR_OFFSET_SCALE` | How `dx` moves the first hill farther back |
| `RIBBON_CV1_SPAN_A` | Base CV1 ladder span across all hills |
| `RIBBON_CV1_SPAN_SCALE` | How `dx` widens the CV1 ladder |
| `RIBBON_CV1_STALL_SPAN_SCALE` | Extra CV1 ladder span when progress stalls |
| `RIBBON_CV1_BACK_OFFSET_MAX_A` | Max rearward extent of the ladder |
| `RIBBON_CV2_SPREAD_A` | Base lateral CV2 spread |
| `RIBBON_CV2_SPREAD_SCALE` | Extra CV2 spread from `dy` |
| `RIBBON_CV2_SHIFT_SCALE` | Moves the ladder center in CV2 using `dy` |
| `RIBBON_SECONDARY_AMP_FRACTION` | Amplitude fraction for hill 2 |
| `RIBBON_TERTIARY_AMP_FRACTION` | Amplitude fraction for hill 3 |
| `RIBBON_SIGMA_Y_SCALE` | Makes CV2 width broader than CV1 width |
| `RIBBON_SECONDARY_SIGMA_X_SCALE` | Broadens hill 2 along CV1 |
| `RIBBON_TERTIARY_SIGMA_X_SCALE` | Broadens hill 3 along CV1 |
| `STRONG_BIAS_STALL_TRIGGER` | Stall count before barrier boost can activate |
| `STRONG_BIAS_PLATEAU_MIN_A` | Lower bound of the barrier region on CV1 |
| `STRONG_BIAS_PLATEAU_MAX_A` | Upper bound of the barrier region on CV1 |
| `STRONG_BIAS_REMAINING_MIN_A` | Minimum remaining distance to target before boosting |
| `STRONG_BIAS_GAIN` | Extra multiplicative gain inside the barrier when stalled |
| `PRIMARY_OFFSET_SIGMA_SCALE` | Primary hill offset as a fraction of `sigma` |
| `PRIMARY_OFFSET_SIGMA_SCALE_PLATEAU` | Stronger primary offset scale inside the plateau region |
| `BIAS_FRONTIER_STEP_A` | Minimum forward march of the primary hill between actions |
| `BIAS_FRONTIER_STALL_STEP_A` | Extra forward march while stalled |
| `BIAS_FRONTIER_PLATEAU_STEP_A` | Extra forward march inside the plateau region |

Bias scaling:

```text
amp = base_amp * (3 - 2 * progress)
sigma = base_sigma * (1.5 - progress)
```

Placement logic:

```text
Each action deposits a coherent 3-hill ladder in 2D CV space:
- hill 1: closest-behind hill on CV1
- hill 2: farther-behind hill with broader sigma_x
- hill 3: farthest-behind support hill with the broadest sigma_x

The CV2 centers remain 2D:
- the ladder center shifts with dy
- hills 2 and 3 are spread above and below the CV2 center

Across steps, a primary-hill frontier is tracked:
- repeated actions do not keep depositing the primary hill at one fixed CV1 center
- while stalled, the primary hill is forced to march forward between actions
- this is intended to improve cross-step coordination and build a coherent repulsive wall
```

Stall-based escalation:

```text
if no_improve_counter >= 2:
    amp *= min(1 + 0.7 * no_improve_counter, MAX_ESCALATION_FACTOR)
```

Barrier-aware strength boost:

```text
if stalled and CV1 is inside the plateau region and target is still far away:
    amp *= (1 + STRONG_BIAS_GAIN * min(no_improve_counter, 3))
```

Bias potential:

```text
V_bias(cv1, cv2) = sum_i A_i * exp(
    -((cv1 - mu_x_i)^2 / (2 sigma_x_i^2))
    -((cv2 - mu_y_i)^2 / (2 sigma_y_i^2))
)
```

Checkpoint / resume rule:

- checkpoints must persist curriculum state (`curriculum_stage` and recent stage-success history)
- resumed jobs must restore that state before continuing training
- otherwise long cluster runs silently fall back to early-stage shaping

## 5. Episode Plotting

### 5.1 2D reweighted FES plot

Script: [analysis/plot_episode_bias_fes.py](../analysis/plot_episode_bias_fes.py)

Command:

```powershell
python analysis\plot_episode_bias_fes.py --episode 10 --temperature 300 --bins 120
```

CLI options:

| Option | Meaning |
|---|---|
| `--episode` | Episode number |
| `--temperature` | Reweighting temperature in K |
| `--bins` | Histogram/grid bins |

### 5.2 3D unbiased FES plot

Script: [analysis/plot_episode_bias_fes_3d.py](../analysis/plot_episode_bias_fes_3d.py)

Command:

```powershell
python analysis\plot_episode_bias_fes_3d.py --episode 10 --temperature 300 --bins 140
```

### 5.3 Stacked slice visualization

Script: [analysis/stacked_imshow_slices.py](../analysis/stacked_imshow_slices.py)

Command:

```powershell
python analysis\stacked_imshow_slices.py --episode 10 --temperature 300 --bins 180
```

Current layer order:

- top: time-colored trajectory only
- middle: unbiased FES
- bottom: bias

## 6. Runtime Controls in `main_2d.py`

Defined near the bottom of [main_2d.py](../main_2d.py):

| Variable | Meaning |
|---|---|
| `WARMUP_EPISODES` | Initial episodes with stronger restrictions |
| `RESUME` | Resume or fresh-start training |
| `resume_ep` | Checkpoint episode to load |
| `total_target` | Total desired episode count |

Resume details:

- `load_agent_from_episode()` restores PPO weights and checkpoint payload metadata
- checkpoints are written after curriculum promotion is updated
- resumed runs continue from the saved curriculum stage instead of restarting from stage 0

## 7. Safe Tuning Guidance

### If CV1 is stuck

Check:

- whether the first curriculum zone excludes the initial state
- whether bias hills are actually deposited
- whether amplitudes are being masked in-zone
- whether `CURRENT_DISTANCE` still matches the real fresh-start structure

### If MD is unstable

Reduce first:

- `MAX_ESCALATION_FACTOR`
- `MAX_AMP`
- `RIBBON_SECONDARY_AMP_FRACTION`
- `RIBBON_TERTIARY_AMP_FRACTION`

### If critic loss explodes

Check first:

- reward scale
- MD failure frequency
- observation normalization
- invalid transition filtering

## 8. Outputs Produced by Training

Important output folders under `results_PPO/`:

- `dcd_trajs/`: per-action DCD segments
- `full_trajectories/`: per-episode CV CSVs and plots
- `episode_meta/`: per-episode bias logs and metadata
- `bias_profiles/`: saved bias surfaces
- `episode_pdbs/`: optional end-of-episode structures
- `checkpoints/`: PPO checkpoints
- `analysis_runs/episode_XXXX/`: per-episode analysis figures
