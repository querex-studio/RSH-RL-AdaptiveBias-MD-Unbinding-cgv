import os
import numpy as np
import torch
import combined_2d as config
from combined_2d import (
    PPOAgent,
    ProteinEnvironmentRedesigned,
    _ensure,
    save_checkpoint,
    plot_distance_trajectory,
    save_episode_bias_profiles,
    write_episode_pdb,
    export_episode_metadata,
)
from torch.serialization import add_safe_globals
import openmm.unit as mm_unit
import torch
from torch.serialization import add_safe_globals
import openmm.unit as mm_unit
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")  # use non-GUI backend that writes files only
from collections import deque

try:
    from analysis.plot_episode_bias_fes import make_plot as make_reweighted_episode_plot
except Exception:
    make_reweighted_episode_plot = None

WARMUP_EPISODES = 10  # small warm-up with locks to validate stability

# Allow-list OpenMM classes used inside the checkpoint
add_safe_globals([mm_unit.quantity.Quantity, mm_unit.unit.Unit])

def load_agent_from_episode(ep_idx: int):
    results_dir = _ensure(config.RESULTS_DIR)
    ckpt_dir = _ensure(os.path.join(results_dir, "checkpoints"))
    ckpt_path = os.path.join(ckpt_dir, f"ckpt_ep_{ep_idx:04d}.pt")

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"No checkpoint found for episode {ep_idx} at {ckpt_path}")

    # First try the safe weights-only loader
    try:
        payload = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    except Exception:
        # Fallback: legacy behavior (only do this if you trust the file, which you do)
        payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    agent = PPOAgent(config.STATE_SIZE, config.ACTION_SIZE, config.SEED)
    agent.actor.load_state_dict(payload["actor"])
    agent.critic.load_state_dict(payload["critic"])
    if "obs_norm" in payload:
        agent.load_obs_norm_state(payload["obs_norm"])

    print(f"[resume] Loaded checkpoint from episode {ep_idx}: {ckpt_path}")
    return agent, payload

def train_progressive(n_episodes=400, start_ep=1, agent=None, resume_training_meta=None):
    np.random.seed(config.SEED)
    torch.manual_seed(config.SEED)

    # re-use agent if passed (for resume), otherwise start from scratch
    if agent is None:
        agent = PPOAgent(config.STATE_SIZE, config.ACTION_SIZE, config.SEED)

    env = ProteinEnvironmentRedesigned()
    curriculum_targets = list(config.curriculum_cv1_targets())
    success_window = int(getattr(config, "CURRICULUM_SUCCESS_WINDOW", 4))
    curriculum_stage = 0
    recent_stage_success = deque(maxlen=success_window)
    if resume_training_meta:
        curriculum_stage = int(resume_training_meta.get("curriculum_stage", 0))
        curriculum_stage = max(0, min(curriculum_stage, len(curriculum_targets) - 1))
        recent_vals = list(resume_training_meta.get("recent_stage_success", []))
        recent_stage_success = deque(
            [float(x) for x in recent_vals[-success_window:]],
            maxlen=success_window,
        )
        print(
            f"[resume] restored curriculum stage {curriculum_stage + 1}/{len(curriculum_targets)} "
            f"target={curriculum_targets[curriculum_stage]:.2f} A "
            f"window={list(recent_stage_success)}"
        )

    results_dir = _ensure(config.RESULTS_DIR)
    ckpt_dir = _ensure(os.path.join(results_dir, "checkpoints"))

    # start_ep lets us resume from a later episode number
    for ep_idx in range(start_ep, start_ep + n_episodes):
        # warm-up schedule then true training
        if ep_idx <= WARMUP_EPISODES:
            config.ENABLE_MILESTONE_LOCKS = True
            config.IN_ZONE_MAX_AMP = 1.0
        else:
            config.ENABLE_MILESTONE_LOCKS = False
            config.IN_ZONE_MAX_AMP = 1e9

        target_center_A = float(curriculum_targets[curriculum_stage])
        target_half_width_A = float(config.curriculum_half_width_for_target(target_center_A))

        carry_state = not bool(getattr(config, "TRAIN_FRESH_START_EVERY_EPISODE", True))
        state = env.reset(
            seed_from_max_A=None,
            carry_state=carry_state,
            episode_index=ep_idx,
            target_center_A=target_center_A,
            target_half_width_A=target_half_width_A,
            target_stage=curriculum_stage,
        )
        done = False
        steps = 0

        while not done and steps < config.MAX_ACTIONS_PER_EPISODE:
            action, logp, value = agent.act(state, training=True)
            next_state, reward, done, dists = env.step(action)
            agent.save_experience(
                state, action, logp, value, reward, done, next_state
            )
            state = next_state
            steps += 1

        metrics = agent.update()

        # plotting fallback if no segments were recorded
        if not getattr(env, "episode_trajectory_segments", []):
            if len(getattr(env, "distance_history", [])) > 1:
                env.episode_trajectory_segments = [
                    list(map(float, env.distance_history[1:]))
                ]

        if not getattr(env, "episode_trajectory_segments_cv2", []):
            if len(getattr(env, "distance2_history", [])) > 1:
                env.episode_trajectory_segments_cv2 = [
                    list(map(float, env.distance2_history[1:]))
                ]

        # plot trajectory
        if getattr(env, "episode_trajectory_segments", []):
            plot_distance_trajectory(
                env.episode_trajectory_segments,
                ep_idx,
                distance_history=getattr(env, "distance_history", None),
                episode_trajectories_cv2=getattr(env, "episode_trajectory_segments_cv2", None),
            )
        else:
            print(
                f"[Warning] No trajectory data recorded for episode "
                f"{ep_idx}. Nothing to plot."
            )

        # bias profile per episode
        if getattr(config, "SAVE_BIAS_PROFILE", False):
            if (ep_idx % int(getattr(config, "BIAS_PROFILE_EVERY", 1))) == 0:
                save_episode_bias_profiles(getattr(env, "all_biases_in_episode", []), ep_idx)

        # save PDB per episode
        if getattr(config, "SAVE_EPISODE_PDB", False):
            if (ep_idx % int(getattr(config, "EPISODE_PDB_EVERY", 1))) == 0:
                write_episode_pdb(env, getattr(config, "EPISODE_PDB_DIR", config.RESULTS_DIR), ep_idx)

        # persist bias log for later unbiasing/reweighting
        export_episode_metadata(
            ep_idx,
            getattr(env, "bias_log", []),
            getattr(env, "backstops_A", []),
            getattr(env, "backstop_events", None),
            curriculum_target_center_A=getattr(env, "episode_target_center_A", config.TARGET_CENTER),
            curriculum_target_zone=[
                getattr(env, "episode_target_min_A", config.TARGET_MIN),
                getattr(env, "episode_target_max_A", config.TARGET_MAX),
            ],
            target_stage=getattr(env, "episode_target_stage", None),
        )

        stage_success = bool(getattr(env, "episode_target_hit", False))
        episode_stage = int(getattr(env, "episode_target_stage", curriculum_stage))
        episode_target_center = float(getattr(env, "episode_target_center_A", target_center_A))
        recent_stage_success.append(1.0 if stage_success else 0.0)
        if (
            curriculum_stage < (len(curriculum_targets) - 1)
            and len(recent_stage_success) == recent_stage_success.maxlen
            and float(np.mean(recent_stage_success)) >= float(getattr(config, "CURRICULUM_PROMOTION_THRESHOLD", 0.75))
        ):
            curriculum_stage += 1
            recent_stage_success.clear()
            print(
                f"[curriculum] promoted to stage {curriculum_stage + 1}/{len(curriculum_targets)} "
                f"target={curriculum_targets[curriculum_stage]:.2f} A"
            )

        # checkpoint after curriculum state is updated, so resumes restore the true stage.
        if ep_idx % config.SAVE_CHECKPOINT_EVERY == 0:
            save_checkpoint(
                agent,
                env,
                ckpt_dir,
                ep_idx,
                training_meta={
                    "curriculum_stage": int(curriculum_stage),
                    "recent_stage_success": list(recent_stage_success),
                    "curriculum_targets": list(map(float, curriculum_targets)),
                },
            )

        if (
            make_reweighted_episode_plot is not None
            and getattr(config, "AUTO_REWEIGHTED_FES_PLOT", False)
            and (ep_idx % int(getattr(config, "AUTO_REWEIGHTED_FES_EVERY", 1))) == 0
        ):
            try:
                out_png = make_reweighted_episode_plot(
                    ep_idx,
                    temperature=float(getattr(config, "AUTO_REWEIGHTED_FES_TEMPERATURE", 300.0)),
                    bins=int(getattr(config, "AUTO_REWEIGHTED_FES_BINS", 120)),
                )
                print(f"[plot] saved reweighted FES plot: {out_png}")
            except Exception as e:
                print(f"[plot] failed to generate reweighted FES plot for episode {ep_idx}: {e}")

        # simple console log
        if metrics:
            print(
                f"[ep {ep_idx:04d}] steps={steps} "
                f"stage={episode_stage + 1}/{len(curriculum_targets)} "
                f"target={episode_target_center:.2f}A "
                f"hit={int(stage_success)} "
                f"loss={metrics.get('loss', 0):.3f} "
                f"actor={metrics.get('actor_loss', 0):.3f} "
                f"critic={metrics.get('critic_loss', 0):.3f} "
                f"ent={metrics.get('entropy', 0):.3f} "
                f"kl={metrics.get('approx_kl', 0):.3f} "
                f"clip={metrics.get('clip_frac', 0):.2f} "
                f"lr={metrics.get('lr', 0):.2e}"
            )

    print("Training complete.")


if __name__ == "__main__":
    # ---------------------------------------------------------------------
    # Run control: start fresh OR resume from an on-disk checkpoint.
    #
    # Set RESUME=False to start fresh training from episode 1 with a new agent.
    # Set RESUME=True  to resume from `resume_ep` if that checkpoint exists.
    # If the checkpoint is missing, it will automatically fall back to fresh.
    # ---------------------------------------------------------------------

    RESUME = False          # True -> resume from checkpoint; False -> start fresh
    resume_ep = 295        # checkpoint episode index to load (only used if RESUME=True)
    total_target = 50     # total episodes you want to complete in this run

    if RESUME:
        try:
            agent, payload = load_agent_from_episode(resume_ep)
            resume_training_meta = payload.get("training_meta", {})
            remaining = total_target - resume_ep
            start_ep = resume_ep + 1
        except FileNotFoundError as e:
            print(f"[resume] {e}")
            print("[resume] Falling back to fresh training from episode 1.")
            agent = None
            resume_training_meta = None
            remaining = total_target
            start_ep = 1
    else:
        agent = None
        resume_training_meta = None
        remaining = total_target
        start_ep = 1

    train_progressive(
        n_episodes=remaining,
        start_ep=start_ep,
        agent=agent,
        resume_training_meta=resume_training_meta,
    )
