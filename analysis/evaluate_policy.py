import argparse
import csv
import glob
import importlib
import json
import os
import re
import sys
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import openmm.unit as mm_unit
import torch
from torch.serialization import add_safe_globals

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

add_safe_globals([mm_unit.quantity.Quantity, mm_unit.unit.Unit])


def _load_config(module_name):
    try:
        return importlib.import_module(module_name)
    except Exception:
        return importlib.import_module("combined_2d")


def _ensure(path):
    os.makedirs(path, exist_ok=True)
    return path


def _time_tag():
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def _default_eval_root(cfg):
    return os.path.join(getattr(cfg, "RESULTS_DIR", os.path.join(ROOT_DIR, "results_PPO")), "evaluation_runs")


def _parse_ckpt_episode(path):
    m = re.search(r"ckpt_ep_(\d+)\.pt$", os.path.basename(path))
    return int(m.group(1)) if m else -1


def _find_checkpoints(cfg, checkpoint_dir=None, checkpoint_glob=None, episode=None):
    ckpt_dir = checkpoint_dir or os.path.join(getattr(cfg, "RESULTS_DIR", os.path.join(ROOT_DIR, "results_PPO")), "checkpoints")
    if checkpoint_glob:
        pattern = checkpoint_glob if os.path.isabs(checkpoint_glob) else os.path.join(ROOT_DIR, checkpoint_glob)
    else:
        pattern = os.path.join(ckpt_dir, "ckpt_ep_*.pt")
    paths = sorted(glob.glob(pattern), key=_parse_ckpt_episode)
    if episode is not None:
        paths = [p for p in paths if _parse_ckpt_episode(p) == int(episode)]
    return paths


def load_agent_from_checkpoint(cfg, ckpt_path):
    try:
        payload = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    except Exception:
        payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    agent = cfg.PPOAgent(cfg.STATE_SIZE, cfg.ACTION_SIZE, cfg.SEED)
    agent.actor.load_state_dict(payload["actor"])
    agent.critic.load_state_dict(payload["critic"])
    if "obs_norm" in payload:
        agent.load_obs_norm_state(payload["obs_norm"])
    return agent, payload


def _run_greedy_episode(cfg, agent, env, episode_idx, target_center_A, target_half_width_A):
    state = env.reset(
        seed_from_max_A=None,
        carry_state=False,
        episode_index=None,
        target_center_A=float(target_center_A),
        target_half_width_A=float(target_half_width_A),
        target_stage=None,
    )
    total_reward = 0.0
    steps = 0
    max_cv1 = float(getattr(env, "current_distance", np.nan))
    done = False

    while not done and steps < int(cfg.MAX_ACTIONS_PER_EPISODE):
        action, _, _ = agent.act(state, training=False)
        next_state, reward, done, _ = env.step(action)
        total_reward += float(reward)
        state = next_state
        steps += 1
        max_cv1 = max(max_cv1, float(getattr(env, "current_distance", max_cv1)))

    target_center, target_min, target_max, _ = env.current_target_zone()
    stable_target_stay = bool(getattr(env, "in_zone_count", 0) >= int(cfg.STABILITY_STEPS))
    center_tolerance_success = bool(abs(float(env.current_distance) - float(target_center)) < float(cfg.PHASE2_TOL))

    return {
        "eval_episode": int(episode_idx),
        "steps": int(steps),
        "total_reward": float(total_reward),
        "final_cv1_A": float(env.current_distance),
        "final_cv2_A": float(env.current_distance2),
        "max_cv1_A": float(max_cv1),
        "target_center_A": float(target_center),
        "target_min_A": float(target_min),
        "target_max_A": float(target_max),
        "target_zone_hit": int(bool(getattr(env, "episode_target_hit", False))),
        "stable_target_stay": int(stable_target_stay),
        "center_tolerance_success": int(center_tolerance_success),
        "md_failed": int(bool(getattr(env, "episode_md_failed", False))),
        "in_zone_count": int(getattr(env, "in_zone_count", 0)),
    }


def evaluate_checkpoint(cfg, ckpt_path, n_episodes, target_center_A=None, target_half_width_A=None):
    agent, payload = load_agent_from_checkpoint(cfg, ckpt_path)
    ckpt_episode = int(payload.get("episode", _parse_ckpt_episode(ckpt_path)))
    env = cfg.ProteinEnvironmentRedesigned()

    target_center_A = float(cfg.TARGET_CENTER if target_center_A is None else target_center_A)
    target_half_width_A = float(cfg.TARGET_ZONE_HALF_WIDTH if target_half_width_A is None else target_half_width_A)

    episode_rows = []
    for eval_idx in range(1, int(n_episodes) + 1):
        row = _run_greedy_episode(cfg, agent, env, eval_idx, target_center_A, target_half_width_A)
        row["checkpoint_episode"] = int(ckpt_episode)
        row["checkpoint_path"] = ckpt_path
        episode_rows.append(row)

    hit_rate = float(np.mean([row["target_zone_hit"] for row in episode_rows]))
    stable_rate = float(np.mean([row["stable_target_stay"] for row in episode_rows]))
    md_failure_rate = float(np.mean([row["md_failed"] for row in episode_rows]))
    center_tol_rate = float(np.mean([row["center_tolerance_success"] for row in episode_rows]))
    mean_max_cv1 = float(np.mean([row["max_cv1_A"] for row in episode_rows]))
    mean_final_cv1 = float(np.mean([row["final_cv1_A"] for row in episode_rows]))
    mean_reward = float(np.mean([row["total_reward"] for row in episode_rows]))

    summary = {
        "checkpoint_episode": int(ckpt_episode),
        "checkpoint_path": ckpt_path,
        "evaluation_episodes": int(n_episodes),
        "target_center_A": float(target_center_A),
        "target_min_A": float(target_center_A - target_half_width_A),
        "target_max_A": float(target_center_A + target_half_width_A),
        "long_run_success_rate": stable_rate,
        "target_zone_hit_rate": hit_rate,
        "stable_target_zone_stay_rate": stable_rate,
        "center_tolerance_success_rate": center_tol_rate,
        "md_failure_rate": md_failure_rate,
        "mean_max_cv1_A": mean_max_cv1,
        "mean_final_cv1_A": mean_final_cv1,
        "mean_total_reward": mean_reward,
    }
    return summary, episode_rows


def _write_csv(path, rows):
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _plot_dashboard(summary_rows, out_path):
    episodes = [int(row["checkpoint_episode"]) for row in summary_rows]
    hit_rate = [float(row["target_zone_hit_rate"]) for row in summary_rows]
    stable_rate = [float(row["stable_target_zone_stay_rate"]) for row in summary_rows]
    md_rate = [float(row["md_failure_rate"]) for row in summary_rows]
    mean_max_cv1 = [float(row["mean_max_cv1_A"]) for row in summary_rows]

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    ax = axes[0, 0]
    ax.plot(episodes, hit_rate, marker="o", linewidth=1.8, color="#1f77b4")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title("Target-Zone Hit Rate")
    ax.set_xlabel("Checkpoint Episode")
    ax.set_ylabel("Rate")
    ax.grid(alpha=0.2)

    ax = axes[0, 1]
    ax.plot(episodes, stable_rate, marker="o", linewidth=1.8, color="#2ca02c")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title("Stable Target-Zone Stay Rate")
    ax.set_xlabel("Checkpoint Episode")
    ax.set_ylabel("Rate")
    ax.grid(alpha=0.2)

    ax = axes[1, 0]
    ax.plot(episodes, md_rate, marker="o", linewidth=1.8, color="#d62728")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title("MD Failure Rate")
    ax.set_xlabel("Checkpoint Episode")
    ax.set_ylabel("Rate")
    ax.grid(alpha=0.2)

    ax = axes[1, 1]
    ax.axis("off")
    ranked = sorted(summary_rows, key=lambda row: (-float(row["stable_target_zone_stay_rate"]), int(row["checkpoint_episode"])))
    top_rows = ranked[: min(8, len(ranked))]
    cell_text = []
    for row in top_rows:
        cell_text.append([
            f'{int(row["checkpoint_episode"]):04d}',
            f'{float(row["target_zone_hit_rate"]):.2f}',
            f'{float(row["stable_target_zone_stay_rate"]):.2f}',
            f'{float(row["md_failure_rate"]):.2f}',
            f'{float(row["mean_max_cv1_A"]):.2f}',
        ])
    table = ax.table(
        cellText=cell_text,
        colLabels=["ckpt", "hit", "stable", "md fail", "mean max CV1"],
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.15, 1.35)
    ax.set_title("Per-Checkpoint Evaluation Performance")

    fig.suptitle("Greedy Fresh-Start Evaluation Dashboard", fontsize=14)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def _write_markdown_report(path, summary_rows, eval_episode_rows, target_center_A, target_half_width_A, episodes_per_checkpoint):
    if summary_rows:
        overall_long_run_success = float(np.mean([row["long_run_success_rate"] for row in summary_rows]))
        overall_hit_rate = float(np.mean([row["target_zone_hit_rate"] for row in summary_rows]))
        overall_stable_rate = float(np.mean([row["stable_target_zone_stay_rate"] for row in summary_rows]))
        overall_md_failure_rate = float(np.mean([row["md_failure_rate"] for row in summary_rows]))
        best_row = max(summary_rows, key=lambda row: (float(row["stable_target_zone_stay_rate"]), float(row["target_zone_hit_rate"]), -float(row["md_failure_rate"])))
    else:
        overall_long_run_success = 0.0
        overall_hit_rate = 0.0
        overall_stable_rate = 0.0
        overall_md_failure_rate = 0.0
        best_row = None

    with open(path, "w", encoding="utf-8") as fh:
        fh.write("# PPO Evaluation Report\n\n")
        fh.write("This report uses greedy fresh-start episodes against the final target zone.\n\n")
        fh.write("## Evaluation Setup\n\n")
        fh.write(f"- Episodes per checkpoint: {int(episodes_per_checkpoint)}\n")
        fh.write(f"- Target center (A): {float(target_center_A):.3f}\n")
        fh.write(f"- Target zone min (A): {float(target_center_A - target_half_width_A):.3f}\n")
        fh.write(f"- Target zone max (A): {float(target_center_A + target_half_width_A):.3f}\n")
        fh.write(f"- Checkpoints evaluated: {len(summary_rows)}\n")
        fh.write("\n## Aggregate Results\n\n")
        fh.write(f"- Long-run success rate: {overall_long_run_success:.4f}\n")
        fh.write(f"- Target-zone hit rate: {overall_hit_rate:.4f}\n")
        fh.write(f"- Stable target-zone stay rate: {overall_stable_rate:.4f}\n")
        fh.write(f"- MD failure rate: {overall_md_failure_rate:.4f}\n")
        if best_row is not None:
            fh.write(f"- Best checkpoint: episode {int(best_row['checkpoint_episode']):04d}\n")
            fh.write(f"- Best checkpoint stable target-zone stay rate: {float(best_row['stable_target_zone_stay_rate']):.4f}\n")
        fh.write("\n## Definitions\n\n")
        fh.write("- `target-zone hit rate`: fraction of evaluation episodes where CV1 entered the target zone at least once.\n")
        fh.write("- `stable target-zone stay rate`: fraction of evaluation episodes where the agent remained in-zone for at least `STABILITY_STEPS`.\n")
        fh.write("- `long-run success rate`: reported here as the stable target-zone stay rate under greedy fresh-start evaluation.\n")
        fh.write("- `MD failure rate`: fraction of evaluation episodes terminated by MD failure / NaN handling.\n")
        fh.write("\n## Per-Checkpoint Summary\n\n")
        fh.write("| Checkpoint | Hit Rate | Stable Stay Rate | MD Failure Rate | Mean Max CV1 (A) | Mean Final CV1 (A) |\n")
        fh.write("|---|---:|---:|---:|---:|---:|\n")
        for row in summary_rows:
            fh.write(
                f"| {int(row['checkpoint_episode']):04d} | "
                f"{float(row['target_zone_hit_rate']):.4f} | "
                f"{float(row['stable_target_zone_stay_rate']):.4f} | "
                f"{float(row['md_failure_rate']):.4f} | "
                f"{float(row['mean_max_cv1_A']):.4f} | "
                f"{float(row['mean_final_cv1_A']):.4f} |\n"
            )


def main():
    parser = argparse.ArgumentParser(description="Evaluate PPO checkpoints with greedy fresh-start episodes and write summary reports.")
    parser.add_argument("--config-module", default="combined_2d", help="Config module to import.")
    parser.add_argument("--checkpoint-dir", default=None, help="Checkpoint directory. Defaults to results_PPO/checkpoints.")
    parser.add_argument("--checkpoint-glob", default=None, help="Optional checkpoint glob override.")
    parser.add_argument("--episode", type=int, default=None, help="Evaluate only one checkpoint episode.")
    parser.add_argument("--episodes", type=int, default=None, help="Greedy fresh-start evaluation episodes per checkpoint.")
    parser.add_argument("--target-center", type=float, default=None, help="Override final evaluation target center.")
    parser.add_argument("--target-half-width", type=float, default=None, help="Override final evaluation target half-width.")
    parser.add_argument("--run", default=None, help="Output evaluation run directory.")
    args = parser.parse_args()

    cfg = _load_config(args.config_module)
    n_episodes = int(args.episodes if args.episodes is not None else getattr(cfg, "N_EVAL_EPISODES", 3))
    target_center_A = float(args.target_center if args.target_center is not None else getattr(cfg, "TARGET_CENTER", getattr(cfg, "FINAL_TARGET", 0.0)))
    target_half_width_A = float(args.target_half_width if args.target_half_width is not None else getattr(cfg, "TARGET_ZONE_HALF_WIDTH", 0.35))

    ckpt_paths = _find_checkpoints(cfg, checkpoint_dir=args.checkpoint_dir, checkpoint_glob=args.checkpoint_glob, episode=args.episode)
    if not ckpt_paths:
        raise FileNotFoundError("No checkpoints found for evaluation.")

    run_dir = args.run
    if run_dir is None:
        run_dir = os.path.join(_default_eval_root(cfg), f"eval_{_time_tag()}")
    elif not os.path.isabs(run_dir):
        run_dir = os.path.join(ROOT_DIR, run_dir)
    _ensure(run_dir)

    summary_rows = []
    episode_rows = []
    for ckpt_path in ckpt_paths:
        print(f"[eval] evaluating checkpoint: {ckpt_path}")
        summary, rows = evaluate_checkpoint(
            cfg,
            ckpt_path,
            n_episodes=n_episodes,
            target_center_A=target_center_A,
            target_half_width_A=target_half_width_A,
        )
        summary_rows.append(summary)
        episode_rows.extend(rows)

    summary_csv = os.path.join(run_dir, "checkpoint_eval_summary.csv")
    episodes_csv = os.path.join(run_dir, "checkpoint_eval_episodes.csv")
    summary_json = os.path.join(run_dir, "checkpoint_eval_summary.json")
    dashboard_png = os.path.join(run_dir, "checkpoint_eval_dashboard.png")
    report_md = os.path.join(run_dir, "checkpoint_eval_report.md")

    _write_csv(summary_csv, summary_rows)
    _write_csv(episodes_csv, episode_rows)
    with open(summary_json, "w", encoding="utf-8") as fh:
        json.dump(
            {
                "config_module": args.config_module,
                "evaluation_episodes_per_checkpoint": int(n_episodes),
                "target_center_A": float(target_center_A),
                "target_half_width_A": float(target_half_width_A),
                "summary_rows": summary_rows,
            },
            fh,
            indent=2,
        )
    _plot_dashboard(summary_rows, dashboard_png)
    _write_markdown_report(
        report_md,
        summary_rows,
        episode_rows,
        target_center_A=target_center_A,
        target_half_width_A=target_half_width_A,
        episodes_per_checkpoint=n_episodes,
    )

    print(f"[eval] saved summary CSV: {summary_csv}")
    print(f"[eval] saved episode CSV: {episodes_csv}")
    print(f"[eval] saved dashboard: {dashboard_png}")
    print(f"[eval] saved report: {report_md}")


if __name__ == "__main__":
    main()
