"""
Closed-loop hybrid PPO controller.

This automates the round-level loop:

PPO training -> PCA/TICA analysis -> restart PDB selection -> next PPO round

PCA/TICA remain post-processing tools. They influence learning by changing the
start-state distribution between PPO rounds, not by entering each PPO update.
"""

import argparse
import csv
import glob
import json
import os
import subprocess
import sys
import time

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

import combined_2d as config
from main_2d import train_progressive


def _run(cmd):
    print("[hybrid-auto] " + " ".join(cmd))
    subprocess.run(cmd, cwd=ROOT_DIR, check=True)


def _read_restart_candidates(csv_path, space):
    if not os.path.exists(csv_path):
        return []
    rows = []
    with open(csv_path, "r", newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            pdb_path = row.get("pdb_path")
            if not pdb_path or not os.path.exists(pdb_path):
                continue
            item = dict(row)
            item["space"] = space
            item["pdb_path"] = os.path.abspath(pdb_path)
            item["source_rank"] = row.get("hybrid_rank") or row.get("rank") or ""
            rows.append(item)
    return rows


def _analysis_run_dir(name):
    root = getattr(config, "RUNS_DIR", os.path.join(config.RESULTS_DIR, "analysis_runs"))
    os.makedirs(root, exist_ok=True)
    return os.path.join(root, name)


def _run_pca_round(args, round_idx, max_episode):
    run_dir = _analysis_run_dir(f"hybrid_auto_round_{round_idx:02d}_pca")
    cmd = [
        sys.executable, "scripts/pca.py",
        "--config-module", args.config_module,
        "--max-traj", str(args.max_traj),
        "--contact-stride", str(args.contact_stride),
        "--stride", str(args.stride),
        "--max-residues", str(args.max_residues),
        "--run", run_dir,
    ]
    _run(cmd)
    _run([
        sys.executable, "scripts/pca_hybrid_restart.py",
        "--config-module", args.config_module,
        "--tica-run", run_dir,
        "--max-episode", str(max_episode),
        "--top-restarts", str(args.top_restarts),
    ])
    candidate_csv = os.path.join(run_dir, "pca_hybrid_restart", "data", "hybrid_restart_candidates.csv")
    return run_dir, _read_restart_candidates(candidate_csv, "pca")


def _run_tica_round(args, round_idx, max_episode):
    run_dir = _analysis_run_dir(f"hybrid_auto_round_{round_idx:02d}_tica")
    cmd = [
        sys.executable, "scripts/tica.py",
        "--config-module", args.config_module,
        "--max-traj", str(args.max_traj),
        "--contact-stride", str(args.contact_stride),
        "--stride", str(args.stride),
        "--max-residues", str(args.max_residues),
        "--lag", str(args.lag),
        "--run", run_dir,
    ]
    _run(cmd)
    _run([
        sys.executable, "scripts/hybrid_restart.py",
        "--config-module", args.config_module,
        "--tica-run", run_dir,
        "--max-episode", str(max_episode),
        "--top-restarts", str(args.top_restarts),
    ])
    candidate_csv = os.path.join(run_dir, "hybrid_restart", "data", "hybrid_restart_candidates.csv")
    return run_dir, _read_restart_candidates(candidate_csv, "tica")


def _dedupe_candidates(candidates):
    seen = set()
    out = []
    for item in candidates:
        key = os.path.abspath(item["pdb_path"])
        if key in seen:
            continue
        seen.add(key)
        out.append(item)
    return out


def _write_report(out_dir, summary):
    os.makedirs(out_dir, exist_ok=True)
    json_path = os.path.join(out_dir, "hybrid_auto_summary.json")
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    md_path = os.path.join(out_dir, "hybrid_auto_report.md")
    with open(md_path, "w", encoding="utf-8") as fh:
        fh.write("# Closed-Loop Hybrid PPO Report\n\n")
        fh.write("This controller trains PPO in rounds and uses PCA/TICA restart candidates between rounds.\n\n")
        fh.write("## Scope\n\n")
        fh.write("- PPO remains the online RL method.\n")
        fh.write("- PCA/TICA affect the next round by selecting restart structures.\n")
        fh.write("- PCA/TICA are not used inside each PPO step.\n\n")
        fh.write("## Rounds\n\n")
        for row in summary["rounds"]:
            fh.write(
                f"- Round {row['round']}: episodes `{row['start_episode']}-{row['end_episode']}`, "
                f"restart pool in `{row['restart_pool_size']}`, restart pool out `{row['selected_restart_count']}`\n"
            )
        fh.write("\n## Final Restart Pool\n\n")
        for item in summary.get("final_restart_pool", [])[:20]:
            fh.write(f"- `{item.get('space')}` rank `{item.get('source_rank')}`: `{item.get('pdb_path')}`\n")
    return json_path, md_path


def _parse_args():
    parser = argparse.ArgumentParser(description="Closed-loop PPO + PCA/TICA restart-selection controller.")
    parser.add_argument("--config-module", default="combined_2d")
    parser.add_argument("--rounds", type=int, default=2, help="Number of PPO training rounds.")
    parser.add_argument("--initial-episodes", type=int, default=10, help="Episodes in round 1 before restart candidates exist.")
    parser.add_argument("--episodes-per-round", type=int, default=10, help="Episodes in later hybrid rounds.")
    parser.add_argument("--restart-fraction", type=float, default=0.30, help="Fraction of episodes started from selected restart PDBs in rounds after round 1.")
    parser.add_argument("--analysis-space", choices=["tica", "pca", "both"], default="both", help="Analysis spaces used to select restart PDBs.")
    parser.add_argument("--top-restarts", type=int, default=10)
    parser.add_argument("--max-traj", type=int, default=0)
    parser.add_argument("--contact-stride", type=int, default=5)
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--max-residues", type=int, default=40)
    parser.add_argument("--lag", type=int, default=5)
    parser.add_argument("--out", default=None, help="Controller report directory.")
    return parser.parse_args()


def main():
    args = _parse_args()
    if args.config_module != "combined_2d":
        raise ValueError("The current training loop imports combined_2d directly; use --config-module combined_2d.")

    out_dir = args.out or os.path.join(config.RESULTS_DIR, "hybrid_auto", time.strftime("%Y%m%d-%H%M%S"))
    os.makedirs(out_dir, exist_ok=True)

    agent = None
    training_meta = None
    restart_pool = []
    start_episode = 1
    summary = {
        "config_module": args.config_module,
        "rounds_requested": int(args.rounds),
        "restart_fraction": float(args.restart_fraction),
        "analysis_space": args.analysis_space,
        "rounds": [],
    }

    for round_idx in range(1, int(args.rounds) + 1):
        n_episodes = int(args.initial_episodes if round_idx == 1 else args.episodes_per_round)
        end_episode = start_episode + n_episodes - 1
        input_restart_count = len(restart_pool)
        print(
            f"[hybrid-auto] round {round_idx}/{args.rounds}: training episodes "
            f"{start_episode}-{end_episode} with restart_pool={input_restart_count}"
        )
        agent, training_meta = train_progressive(
            n_episodes=n_episodes,
            start_ep=start_episode,
            agent=agent,
            resume_training_meta=training_meta,
            restart_pool=restart_pool,
            restart_fraction=0.0 if round_idx == 1 else float(args.restart_fraction),
            restart_use_final_target=True,
            hybrid_round=round_idx,
        )

        selected = []
        analysis_runs = {}
        if args.analysis_space in ("pca", "both"):
            run_dir, rows = _run_pca_round(args, round_idx, end_episode)
            selected.extend(rows)
            analysis_runs["pca"] = run_dir
        if args.analysis_space in ("tica", "both"):
            run_dir, rows = _run_tica_round(args, round_idx, end_episode)
            selected.extend(rows)
            analysis_runs["tica"] = run_dir

        restart_pool = _dedupe_candidates(selected)
        summary["rounds"].append({
            "round": int(round_idx),
            "start_episode": int(start_episode),
            "end_episode": int(end_episode),
            "restart_pool_size": int(input_restart_count),
            "selected_restart_count": int(len(selected)),
            "analysis_runs": analysis_runs,
        })
        start_episode = end_episode + 1

    summary["final_restart_pool"] = restart_pool
    json_path, md_path = _write_report(out_dir, summary)
    print(f"[hybrid-auto] wrote {json_path}")
    print(f"[hybrid-auto] wrote {md_path}")


if __name__ == "__main__":
    main()
