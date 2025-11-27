import os
import numpy as np
import torch
import config
from agent import PPOAgent
from env_protein import ProteinEnvironmentRedesigned
from util_protein import _ensure, save_checkpoint, plot_distance_trajectory

WARMUP_EPISODES = 10  # small warm-up with locks to validate stability


def train_progressive(n_episodes=400):
    np.random.seed(config.SEED)
    torch.manual_seed(config.SEED)

    agent = PPOAgent(config.STATE_SIZE, config.ACTION_SIZE, config.SEED)
    env = ProteinEnvironmentRedesigned()

    results_dir = _ensure(config.RESULTS_DIR)
    ckpt_dir = _ensure(os.path.join(results_dir, "checkpoints"))

    for ep_idx in range(1, n_episodes + 1):
        # warm-up schedule then true training
        if ep_idx <= WARMUP_EPISODES:
            config.ENABLE_MILESTONE_LOCKS = True
            config.IN_ZONE_MAX_AMP = 1.0
        else:
            config.ENABLE_MILESTONE_LOCKS = False
            config.IN_ZONE_MAX_AMP = 1e9

        # fresh or curriculum start
        state = env.reset(seed_from_max_A=None,
                          carry_state=False,
                          episode_index=ep_idx)
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

        # checkpoint
        if ep_idx % config.SAVE_CHECKPOINT_EVERY == 0:
            save_checkpoint(agent, env, ckpt_dir, ep_idx)

        # plotting fallback if no segments were recorded
        if not getattr(env, "episode_trajectory_segments", []):
            if len(getattr(env, "distance_history", [])) > 1:
                env.episode_trajectory_segments = [
                    list(map(float, env.distance_history[1:]))
                ]

        # plot trajectory
        if getattr(env, "episode_trajectory_segments", []):
            plot_distance_trajectory(
                env.episode_trajectory_segments,
                ep_idx,
                distance_history=getattr(env, "distance_history", None),
            )
        else:
            print(
                f"[Warning] No trajectory data recorded for episode "
                f"{ep_idx}. Nothing to plot."
            )

        # simple console log
        if metrics:
            print(
                f"[ep {ep_idx:04d}] steps={steps} "
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
    train_progressive(n_episodes=400)
