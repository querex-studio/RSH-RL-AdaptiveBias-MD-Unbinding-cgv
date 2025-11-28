import os
import json
import csv
import math
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import openmm.unit as omm_unit

import config


def _ensure(path: str):
    os.makedirs(path, exist_ok=True)
    return path


# ======================== METRICS CSV ========================

def append_metrics_row(path, row_dict):
    _ensure(os.path.dirname(path))
    write_header = not os.path.exists(path)
    with open(path, 'a', newline='') as f:
        w = csv.DictWriter(f, fieldnames=sorted(row_dict.keys()))
        if write_header:
            w.writeheader()
        w.writerow(row_dict)


# ======================== COVERAGE ===========================

def compute_coverage(dist_segments, bin_edges):
    if not dist_segments:
        return np.zeros(len(bin_edges) - 1, dtype=float)
    arrs = [np.asarray(seg, dtype=float) for seg in dist_segments
            if len(seg) > 0]
    if not arrs:
        return np.zeros(len(bin_edges) - 1, dtype=float)
    data = np.concatenate(arrs)
    hist, _ = np.histogram(data, bins=bin_edges)
    cov = hist.astype(float)
    if cov.sum() > 0:
        cov = cov / cov.sum()
    return cov


def plot_coverage_histogram(dist_segments, episode_num: int, bin_size=0.25):
    out_dir = _ensure(f"{config.RESULTS_DIR}/coverage/")
    lo = max(0.5, config.CURRENT_DISTANCE - 1.0)
    hi = config.FINAL_TARGET + 1.5
    bins = np.arange(lo, hi + bin_size, bin_size)
    cov = compute_coverage(dist_segments, bins)

    centers = 0.5 * (bins[1:] + bins[:-1])
    plt.figure(figsize=(10, 4))
    plt.bar(centers, cov, width=bin_size * 0.9, align='center')
    plt.axvspan(config.TARGET_MIN, config.TARGET_MAX,
                alpha=0.18, label='Target Zone')
    for m in config.DISTANCE_INCREMENTS:
        plt.axvline(x=m, linestyle=':', alpha=0.4)
    plt.xlabel('Distance (Å)')
    plt.ylabel('Visit fraction')
    plt.title(f'Distance Coverage — Episode {episode_num}')
    plt.legend(loc='upper right')
    out = os.path.join(out_dir,
                       f"coverage_ep_{episode_num:04d}.png")
    plt.tight_layout()
    plt.savefig(out, dpi=250)
    plt.close()
    print(f"Saved coverage histogram: {out}")


# ======================== KPIs & TRAJ ========================

def plot_distance_trajectory(episode_trajectories,
                             episode_num: int,
                             distance_history=None):
    plot_dir = _ensure(f"{config.RESULTS_DIR}/full_trajectories/")
    # fallback
    if ((not episode_trajectories or len(episode_trajectories) == 0)
            and distance_history and len(distance_history) > 1):
        episode_trajectories = [list(map(float,
                                         distance_history[1:]))]
    if not episode_trajectories:
        print(f"[Warning] No trajectory data recorded for episode "
              f"{episode_num}. Nothing to plot.")
        return

    # flatten segments
    segs = [np.asarray(seg, dtype=np.float32) for seg in episode_trajectories
            if len(seg) > 0]
    full_traj = (np.concatenate(segs)
                 if len(segs) > 1 else segs[0])
    if full_traj.ndim == 0:
        full_traj = full_traj.reshape(1)

    # simple x-axis in pseudo-time units (frames)
    time_axis = np.arange(len(full_traj), dtype=np.float32)

    # plot
    plt.figure(figsize=(9, 4.5))
    plt.plot(time_axis, full_traj, linewidth=1.8)
    plt.axhspan(config.TARGET_MIN, config.TARGET_MAX,
                color='green', alpha=0.15, label='Target zone')
    plt.axhline(config.TARGET_CENTER,
                linestyle='--', linewidth=1.0, label='Target 8.5 Å')
    plt.xlabel("Frames")
    plt.ylabel("P–Mg distance (Å)")
    plt.title(f"Episode {episode_num:04d} trajectory")
    plt.legend()
    out_png = os.path.join(plot_dir,
                           f"progressive_traj_ep_{episode_num:04d}.png")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()
    print(f"Saved trajectory plot: {out_png}")

    # CSV export
    out_csv = os.path.join(plot_dir,
                           f"progressive_traj_ep_{episode_num:04d}.csv")
    np.savetxt(out_csv, np.c_[time_axis, full_traj],
               delimiter=",",
               header="frame,distance_A", comments="")
    print(f"Saved trajectory CSV: {out_csv}")


# ======================== BIAS LANDSCAPES ====================

def _bias_energy_components(bias_log, backstops_A, xA_grid):
    r_nm = xA_grid / 10.0
    per_bias = []
    total = np.zeros_like(xA_grid, dtype=float)

    for (_, amp_kcal, center_A, width_A) in bias_log:
        A_kJ = amp_kcal * 4.184
        mu_nm = center_A / 10.0
        sig_nm = max(1e-6, width_A / 10.0)
        e = A_kJ * np.exp(-((r_nm - mu_nm) ** 2) /
                          (2.0 * sig_nm ** 2))
        per_bias.append(e)
        total += e

    if backstops_A:
        for m_eff_A in backstops_A:
            m_nm = m_eff_A / 10.0
            mask = r_nm < m_nm
            e = np.zeros_like(xA_grid, dtype=float)
            e[mask] = config.BACKSTOP_K * (m_nm - r_nm[mask]) ** 2
            total += e
            per_bias.append(e)

    return per_bias, total


def plot_bias_components_and_sum(bias_log, backstops_A, episode_num):
    if not bias_log and not backstops_A:
        return
    plot_dir = _ensure(f"{config.RESULTS_DIR}/bias_profiles/")
    lo = max(0.5, config.CURRENT_DISTANCE - 2.0)
    hi = config.FINAL_TARGET + 2.0
    xA = np.linspace(lo, hi, 1000)
    per_bias, total = _bias_energy_components(bias_log,
                                              backstops_A, xA)

    plt.figure(figsize=(12, 7))
    for i, e in enumerate(per_bias):
        label = (f"Bias {i+1}" if i < len(bias_log)
                 else f"Backstop {i - len(bias_log) + 1}")
        plt.plot(xA, e, linewidth=1.5, alpha=0.9, label=label)
    plt.plot(xA, total, linewidth=2.5, alpha=1.0,
             label='Cumulative Bias')

    plt.axvline(x=config.CURRENT_DISTANCE,
                linestyle='--', linewidth=2, label='Start')
    plt.axvspan(config.TARGET_MIN, config.TARGET_MAX,
                alpha=0.18, label='Target Zone')
    plt.axvline(x=config.TARGET_CENTER,
                linestyle='--', linewidth=3, label='Target Center')

    plt.xlabel('Position (Å)')
    plt.ylabel('Bias Energy (kJ/mol)')
    plt.title(f'Bias Potentials in Episode {episode_num}')
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.04, 1), loc='upper left')
    plt.tight_layout()
    out = os.path.join(plot_dir,
                       f"bias_components_ep_{episode_num:04d}.png")
    plt.savefig(out, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved bias components plot: {out}")


def plot_bias_sum_only(bias_log, backstops_A, episode_num):
    if not bias_log and not backstops_A:
        return
    plot_dir = _ensure(f"{config.RESULTS_DIR}/bias_profiles/")
    lo = max(0.5, config.CURRENT_DISTANCE - 2.0)
    hi = config.FINAL_TARGET + 2.0
    xA = np.linspace(lo, hi, 1000)
    _, total = _bias_energy_components(bias_log, backstops_A, xA)

    plt.figure(figsize=(12, 7))
    plt.plot(xA, total, linewidth=2.5)
    plt.axvline(x=config.CURRENT_DISTANCE,
                linestyle='--', linewidth=2, label='Start')
    plt.axvspan(config.TARGET_MIN, config.TARGET_MAX,
                alpha=0.18, label='Target Zone')
    plt.axvline(x=config.TARGET_CENTER,
                linestyle='--', linewidth=3, label='Target Center')

    idx_min = np.argmin(total)
    plt.scatter([xA[idx_min]], [total[idx_min]], s=60,
                label=f"Min: {xA[idx_min]:.2f} Å")

    plt.xlabel('Position (Å)')
    plt.ylabel('Bias Energy (kJ/mol)')
    plt.title(f'Cumulative Bias Landscape — Episode {episode_num}')
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.04, 1), loc='upper left')
    plt.tight_layout()
    out = os.path.join(plot_dir,
                       f"bias_sum_ep_{episode_num:04d}.png")
    plt.savefig(out, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved cumulative bias plot: {out}")


# ======================== TIMELINES & LOCKS ==================

def plot_bias_timeline(bias_log, backstop_events, episode_num):
    if not bias_log:
        return
    tl_dir = _ensure(f"{config.RESULTS_DIR}/bias_timeline/")
    steps = [b[0] for b in bias_log]
    amps = [b[1] for b in bias_log]
    cents = [b[2] for b in bias_log]
    widths = [b[3] for b in bias_log]

    plt.figure(figsize=(10, 5))
    plt.plot(steps, amps, marker='o')
    plt.xlabel('Bias index (episode step)')
    plt.ylabel('Amplitude (kcal/mol)')
    plt.title(f'Bias Amplitude Timeline — Episode {episode_num}')
    plt.grid(True, alpha=0.3)
    out1 = os.path.join(tl_dir,
                        f"timeline_amp_ep_{episode_num:04d}.png")
    plt.savefig(out1, dpi=300, bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(steps, cents, marker='o')
    for (s, mA) in backstop_events:
        plt.axhline(y=mA, linestyle=':', alpha=0.5)
    plt.axvspan(config.TARGET_MIN, config.TARGET_MAX,
                alpha=0.18, label='Target Zone')
    plt.axvline(y=config.TARGET_CENTER,
                linestyle='--', linewidth=2, label='Target Center')
    plt.xlabel('Bias index (episode step)')
    plt.ylabel('Center (Å)')
    plt.title(f'Bias Center Timeline — Episode {episode_num}')
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    out2 = os.path.join(tl_dir,
                        f"timeline_center_ep_{episode_num:04d}.png")
    plt.savefig(out2, dpi=300, bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(steps, widths, marker='o')
    plt.xlabel('Bias index (episode step)')
    plt.ylabel('Width (Å)')
    plt.title(f'Bias Width Timeline — Episode {episode_num}')
    plt.grid(True, alpha=0.3)
    out3 = os.path.join(tl_dir,
                        f"timeline_width_ep_{episode_num:04d}.png")
    plt.savefig(out3, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Saved bias timelines: {out1}, {out2}, {out3}")


def plot_lock_snapshot(backstops_A, episode_num):
    if not backstops_A:
        return
    lock_dir = _ensure(f"{config.RESULTS_DIR}/locks/")
    plt.figure(figsize=(12, 3))
    for i, m in enumerate(backstops_A):
        plt.axvline(x=m, linestyle='--',
                    label=f"Lock {i+1} @ {m:.2f} Å")
    plt.axvspan(config.TARGET_MIN, config.TARGET_MAX,
                alpha=0.18, label='Target Zone')
    plt.axvline(x=config.TARGET_CENTER,
                linestyle='--', linewidth=2,
                label='Target Center')
    plt.xlim(max(0.5, config.CURRENT_DISTANCE - 1.5),
             config.FINAL_TARGET + 1.5)
    plt.ylim(0, 1)
    plt.yticks([])
    plt.xlabel('Position (Å)')
    plt.title(f'Milestone Locks Snapshot — Episode {episode_num}')
    plt.legend(bbox_to_anchor=(1.04, 1),
               loc='upper left')
    plt.tight_layout()
    out = os.path.join(lock_dir,
                       f"locks_ep_{episode_num:04d}.png")
    plt.savefig(out, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved locks snapshot: {out}")


# ======================== EPISODE EXPORT =====================

def export_episode_metadata(episode_num,
                            bias_log,
                            backstops_A,
                            backstop_events):
    meta_dir = _ensure(f"{config.RESULTS_DIR}/episode_meta/")
    meta = {
        'episode': int(episode_num),
        'bias_log_columns': ['step', 'amp_kcal', 'center_A', 'width_A'],
        'bias_log': [list(x) for x in bias_log],
        'backstops_A': list(map(float, backstops_A)),
        'backstop_events': [list(map(float, x))
                            for x in backstop_events],
        'start_A': float(config.CURRENT_DISTANCE),
        'target_center_A': float(config.TARGET_CENTER),
        'target_zone': [float(config.TARGET_MIN),
                        float(config.TARGET_MAX)]
    }
    with open(os.path.join(meta_dir,
                           f"episode_{episode_num:04d}.json"),
              'w') as f:
        json.dump(meta, f, indent=2)
    print(f"Saved episode metadata JSON for episode {episode_num}.")


# ======================== Checkpoints ========================

def save_checkpoint(agent, env, ckpt_dir, episode):
    _ensure(ckpt_dir)
    path = os.path.join(ckpt_dir, f"ckpt_ep_{episode:04d}.pt")
    payload = {
        'episode': episode,
        'actor': agent.actor.state_dict(),
        'critic': agent.critic.state_dict(),
        'optimizer': agent.optimizer.state_dict(),
        'scheduler': agent.scheduler.state_dict(),
        'obs_norm': agent.obs_norm_state(),
        'config': {k: getattr(config, k)
                   for k in dir(config) if k.isupper()},
        'env_meta': {
            'best_distance_ever': float(
                getattr(env, 'best_distance_ever', 0.0)
            ),
            'phase': int(getattr(env, 'phase', 1))
        }
    }
    try:
        import torch
        torch.save(payload, path)
        print(f"Saved checkpoint: {path}")
    except Exception as e:
        print(f"[warn] failed to save checkpoint: {e}")


# ======================== Monitoring helpers =================

def _safe_import_mdanalysis():
    try:
        import MDAnalysis
        from MDAnalysis.analysis import distances
        return MDAnalysis, distances
    except Exception:
        print("[monitor] MDAnalysis not available; "
              "skipping trajectory analysis.")
        return None, None


def analyze_mg_coordination_for_dcd(psf_file, dcd_file,
                                    out_dir, run_name):
    MDAnalysis, distances = _safe_import_mdanalysis()
    if MDAnalysis is None:
        return None

    _ensure(out_dir)
    out_path = os.path.join(out_dir, f"{run_name}.txt")

    u = MDAnalysis.Universe(psf_file, dcd_file)
    mg_sel = u.select_atoms("resname MG")
    coordinating_atoms_updating = u.select_atoms(
        "name O* and around 5 resname MG", updating=True
    )

    coord_counts = []
    site_counts = Counter()

    with open(out_path, "w") as output:
        fr = 0
        for ts in u.trajectory:
            if len(coordinating_atoms_updating) == 0 or len(mg_sel) == 0:
                continue

            dist_arr = distances.distance_array(mg_sel,
                                                coordinating_atoms_updating)
            d_sorted = np.sort(dist_arr)[0]
            if len(d_sorted) == 0:
                continue
            if len(d_sorted) >= 7:
                cutoff = d_sorted[5] + (d_sorted[6] - d_sorted[5]) / 2.0
            elif len(d_sorted) >= 6:
                cutoff = d_sorted[5]
            else:
                cutoff = d_sorted[-1]

            temp_sele = u.select_atoms(
                f"name O* and around {cutoff} resname MG",
                updating=True
            )

            n = min(6, len(temp_sele))
            coord_counts.append(n)

            for ik in range(n):
                atom = temp_sele[ik]
                key = (atom.resid, atom.resname, atom.name)
                site_counts[key] += 1
                output.write(f"{atom.resid}_{atom.resname}_{atom.name}")
                if ik < n - 1:
                    output.write(",")
                else:
                    output.write("\n")
            fr += 1

    metrics = {}
    if coord_counts:
        arr = np.asarray(coord_counts, dtype=float)
        metrics["mg_coordination_mean"] = float(arr.mean())
        metrics["mg_coordination_std"] = float(arr.std())
        metrics["mg_unique_sites"] = int(len(site_counts))
    print(f"[monitor] Wrote Mg coordination file: {out_path}")
    return metrics


def analyze_pi_path_for_dcd(psf_file, dcd_file,
                            out_dir, run_name):
    MDAnalysis, _ = _safe_import_mdanalysis()
    if MDAnalysis is None:
        return None

    _ensure(out_dir)
    out_path = os.path.join(out_dir, f"Pi-{run_name}.txt")

    u = MDAnalysis.Universe(psf_file, dcd_file)
    coord1 = u.select_atoms(
        "not segid HETC and around 2 "
        "(segid HETC and (name O2 or name O3))",
        updating=True
    )
    coord2 = u.select_atoms(
        "not segid HETC and around 2 "
        "(segid HETC and (name H1 or name H2))",
        updating=True
    )

    site_counts = Counter()
    frame_counts = []

    with open(out_path, "w") as output:
        fr = 0
        for ts in u.trajectory:
            coordinating_atoms_updating = coord1 + coord2
            n_tot = len(coordinating_atoms_updating)
            frame_counts.append(n_tot)

            for ik in range(n_tot):
                atom = coordinating_atoms_updating[ik]
                key = (atom.resid, atom.resname, atom.name)
                site_counts[key] += 1
                output.write(f"{atom.resid}_{atom.resname}_{atom.name}")
                if ik < n_tot - 1:
                    output.write(",")
                else:
                    output.write("\n")
            fr += 1

    metrics = {}
    if frame_counts:
        arr = np.asarray(frame_counts, dtype=float)
        metrics["pi_contacts_mean"] = float(arr.mean())
        metrics["pi_contacts_std"] = float(arr.std())
        metrics["pi_unique_sites"] = int(len(site_counts))
    print(f"[monitor] Wrote Pi-path file: {out_path}")
    return metrics


def run_mdanalysis_monitoring(run_name, psf_file, dcd_file):
    if not os.path.isfile(dcd_file):
        print(f"[monitor] DCD file not found for run {run_name}: "
              f"{dcd_file}")
        return None

    mg_metrics = analyze_mg_coordination_for_dcd(
        psf_file, dcd_file, config.MG_MONITOR_DIR, run_name
    )
    pi_metrics = analyze_pi_path_for_dcd(
        psf_file, dcd_file, config.PI_MONITOR_DIR, run_name
    )

    return {
        "mg": mg_metrics,
        "pi": pi_metrics,
    }
