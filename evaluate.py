# evaluate.py — robust evaluation (final)
import os, re, glob, json
import numpy as np
import matplotlib.pyplot as plt
import torch
import config
from agent import PPOAgent
from env_protein import ProteinEnvironmentRedesigned

import torch
from torch.serialization import add_safe_globals
import openmm.unit as mm_unit

# Allow-list OpenMM Quantity for safe unpickling when weights_only=True
add_safe_globals([mm_unit.quantity.Quantity])


def _safe_torch_load(path, map_location="cpu"):
    """
    PyTorch 2.6+ changed default weights_only=True.
    Try safe path first, then trusted fallback.
    """
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except Exception:
        # Trusted local artifact: allow full unpickling if needed
        return torch.load(path, map_location=map_location, weights_only=False)


def _ensure(path): os.makedirs(path, exist_ok=True); return path


def _latest_checkpoint(ckpt_dir):
    files = glob.glob(os.path.join(ckpt_dir, "ckpt_ep_*.pt"))
    if not files: return None

    def epnum(p):
        m = re.search(r"ckpt_ep_(\d+)\.pt$", os.path.basename(p));
        return int(m.group(1)) if m else -1

    files.sort(key=epnum);
    return files[-1]


def set_global_seeds(seed: int):
    import random
    random.seed(seed);
    np.random.seed(seed);
    torch.manual_seed(seed)


def load_latest_agent():
    ckpt_dir = os.path.join(config.RESULTS_DIR, "checkpoints")
    ckpt_path = _latest_checkpoint(ckpt_dir)
    agent = PPOAgent(state_size=config.STATE_SIZE, action_size=config.ACTION_SIZE, seed=config.SEED)
    if ckpt_path is None:
        print("[eval] No checkpoint found.")
        return agent, None
    # payload = torch.load(ckpt_path, map_location="cpu")
    payload = _safe_torch_load(ckpt_path, map_location="cpu")
    agent.actor.load_state_dict(payload["actor"])
    agent.critic.load_state_dict(payload["critic"])
    if "obs_norm" in payload: agent.load_obs_norm_state(payload["obs_norm"])
    print(f"[eval] Loaded checkpoint: {ckpt_path}")
    return agent, ckpt_path


def in_zone(d): return config.TARGET_MIN <= float(d) <= config.TARGET_MAX


def success_final(end_distance): return in_zone(end_distance)


def success_anytime(traj): return any(in_zone(x) for x in (traj or []))


def one_episode_eval(env, agent, max_steps):
    traj = [];
    ret = 0.0;
    steps = 0;
    done = False
    s = env.get_state()
    while not done and steps < max_steps:
        a, _, _ = agent.act(s, training=False)
        s, r, done, dists = env.step(a)
        ret += r;
        steps += 1
        if dists: traj.extend([float(x) for x in dists])
    return dict(reward=float(ret), steps=int(steps), end=float(env.current_distance), traj=traj)


def evaluate_policy(agent, starts_A, episodes_per_start=10, max_steps=None):
    saved = config.ENABLE_MILESTONE_LOCKS
    config.ENABLE_MILESTONE_LOCKS = False
    try:
        if max_steps is None: max_steps = config.MAX_ACTIONS_PER_EPISODE
        rows = [];
        succ_final = {s: 0 for s in starts_A};
        succ_any = {s: 0 for s in starts_A}
        for sA in starts_A:
            for ep in range(episodes_per_start):
                env = ProteinEnvironmentRedesigned()
                env.reset(seed_from_max_A=None, carry_state=False)
                env.current_distance = float(sA)
                env.previous_distance = float(sA)
                env.distance_history.clear();
                env.distance_history.append(float(sA))
                epi = one_episode_eval(env, agent, max_steps=max_steps)
                rows.append({'start_A': float(sA), 'episode_idx': ep + 1,
                             'steps': int(epi['steps']), 'reward': float(epi['reward']),
                             'end_A': float(epi['end']),
                             'success_final': int(success_final(epi['end'])),
                             'success_anytime': int(success_anytime(epi['traj']))})
                succ_final[sA] += rows[-1]['success_final']
                succ_any[sA] += rows[-1]['success_anytime']
        n = float(episodes_per_start)
        rates_final = [succ_final[s] / n for s in starts_A]
        rates_any = [succ_any[s] / n for s in starts_A]
        return {'rows': rows, 'starts': list(starts_A), 'rates_final': rates_final, 'rates_any': rates_any}
    finally:
        config.ENABLE_MILESTONE_LOCKS = saved


def main():
    set_global_seeds(config.SEED)
    eval_dir = _ensure(os.path.join(config.RESULTS_DIR, "eval"))
    agent, ckpt_path = load_latest_agent()
    result = evaluate_policy(
        agent,
        starts_A=(3.2, 4.0, 5.0, 6.0),
        episodes_per_start=10,
        max_steps=config.MAX_ACTIONS_PER_EPISODE
    )
    out_csv = os.path.join(eval_dir, "eval_detailed.csv")
    write_header = not os.path.exists(out_csv)
    import csv
    with open(out_csv, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=['start_A', 'episode_idx', 'steps', 'reward', 'end_A', 'success_final',
                                          'success_anytime'])
        if write_header: w.writeheader()
        for r in result['rows']: w.writerow(r)
    print(f"[eval] CSV: {out_csv}")

    print("\n=== Evaluation Summary (locks OFF) ===")
    for s, rf, ra in zip(result['starts'], result['rates_final'], result['rates_any']):
        print(f"Start {s:>4.1f} Å → success(final)={rf * 100:5.1f}% | success(any)={ra * 100:5.1f}%")
    overall_final = float(np.mean(result['rates_final']))
    overall_any = float(np.mean(result['rates_any']))
    print(f"OVERALL → success(final)={overall_final * 100:5.1f}% | success(any)={overall_any * 100:5.1f}%")


if __name__ == "__main__":
    main()
