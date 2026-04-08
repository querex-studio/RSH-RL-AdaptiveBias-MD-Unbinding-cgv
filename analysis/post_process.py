import argparse
import glob
import json
import os
import re
import sys
import importlib

import numpy as np
import matplotlib.pyplot as plt

try:
    import MDAnalysis as mda
except Exception:
    mda = None

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from analysis import run_utils

KB_KCAL = 0.0019872041


def _resolve_runs_root(cfg, root=None):
    base = root or getattr(cfg, "RUNS_DIR", "analysis_runs")
    if not os.path.isabs(base):
        base = os.path.join(ROOT_DIR, base)
    return base


def _resolve_default_traj_glob(cfg):
    base = getattr(cfg, "RESULTS_TRAJ_DIR", os.path.join(ROOT_DIR, "results_PPO", "dcd_trajs"))
    if not os.path.isabs(base):
        base = os.path.join(ROOT_DIR, base)
    return os.path.join(base, "*.dcd")


def _snapshot_cfg(cfg):
    keys = [
        "ATOM1_INDEX", "ATOM2_INDEX", "ATOM3_INDEX", "ATOM4_INDEX",
        "CURRENT_DISTANCE", "FINAL_TARGET", "CURRENT_DISTANCE_2", "FINAL_TARGET_2",
        "TARGET_MIN", "TARGET_MAX", "TARGET2_MIN", "TARGET2_MAX",
        "stepsize", "dcdfreq_mfpt", "DCD_REPORT_INTERVAL",
        "RESULTS_DIR", "RESULTS_TRAJ_DIR",
    ]
    unit_mod = None
    if hasattr(cfg, "unit"):
        unit_mod = cfg.unit
    elif hasattr(cfg, "u"):
        unit_mod = cfg.u
    snap = {}
    for k in keys:
        if not hasattr(cfg, k):
            continue
        val = getattr(cfg, k)
        try:
            if hasattr(val, "value_in_unit") and unit_mod is not None:
                val = float(val.value_in_unit(unit_mod.picoseconds))
        except Exception:
            pass
        snap[k] = val
    return snap


def find_latest_run(root):
    if not os.path.isdir(root):
        return None
    candidates = [
        os.path.join(root, name)
        for name in os.listdir(root)
        if os.path.isdir(os.path.join(root, name))
    ]
    if not candidates:
        return None
    return sorted(candidates)[-1]


def load_fes(run_dir):
    fes_path = os.path.join(run_dir, "data", "fes.npy")
    if os.path.exists(fes_path):
        return np.load(fes_path)
    return None


def plot_fes(fes, out_path, title="FES"):
    plt.figure()
    plt.imshow(fes, cmap="coolwarm", origin="lower")
    plt.colorbar()
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def _metrics_paths(run_dir):
    metrics_dir = os.path.join(run_dir, "metrics")
    if os.path.isdir(metrics_dir):
        return {
            "mfpt": os.path.join(metrics_dir, "total_steps_mfpt.csv"),
            "unbiased": os.path.join(metrics_dir, "total_steps_unbiased.csv"),
            "metad": os.path.join(metrics_dir, "total_steps_metaD.csv"),
        }
    # fallback: project root (legacy)
    return {
        "mfpt": os.path.join(ROOT_DIR, "total_steps_mfpt.csv"),
        "unbiased": os.path.join(ROOT_DIR, "total_steps_unbiased.csv"),
        "metad": os.path.join(ROOT_DIR, "total_steps_metaD.csv"),
    }


def plot_total_steps(run_dir, out_dir, cfg):
    paths = _metrics_paths(run_dir)
    data = {}
    unit_mod = None
    if hasattr(cfg, "unit"):
        unit_mod = cfg.unit
    elif hasattr(cfg, "u"):
        unit_mod = cfg.u
    if unit_mod is None or not hasattr(cfg, "stepsize"):
        return

    for key, path in paths.items():
        if os.path.exists(path):
            values = np.genfromtxt(path, delimiter=",")
            values = np.atleast_1d(values).reshape(-1)
            dt = float(cfg.stepsize.value_in_unit(unit_mod.picoseconds))
            if key == "unbiased" and hasattr(cfg, "stepsize_unbias"):
                dt = float(cfg.stepsize_unbias.value_in_unit(unit_mod.picoseconds))
            data[key] = values * dt

    if not data:
        return

    labels = []
    series = []
    for key in ["mfpt", "unbiased", "metad"]:
        if key in data:
            labels.append(key)
            series.append(data[key])

    plt.figure()
    plt.boxplot(series, labels=labels)
    plt.yscale("log")
    plt.ylabel("Time (ps)")
    plt.title("Time to reach destination")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "total_steps_boxplot_log.png"))
    plt.close()

    plt.figure()
    plt.violinplot(series)
    plt.xticks(np.arange(1, len(labels) + 1), labels)
    plt.yscale("log")
    plt.ylabel("Time (ps)")
    plt.title("Time to reach destination")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "total_steps_violin_log.png"))
    plt.close()


def plot_reconstructed_fes(run_dir, out_dir, cfg):
    if not hasattr(cfg, "num_bins"):
        return
    paths = sorted(glob.glob(os.path.join(run_dir, "data", "*_reconstructed_fes_*.npy")))
    for path in paths:
        fes = np.load(path)
        name = os.path.splitext(os.path.basename(path))[0]
        plot_fes(fes.reshape(cfg.num_bins, cfg.num_bins), os.path.join(out_dir, f"{name}.png"), title=name)


def plot_bias_surfaces(run_dir, out_dir, cfg):
    if not hasattr(cfg, "num_bins"):
        return
    param_paths = sorted(glob.glob(os.path.join(run_dir, "params", "*_gaussian_fes_param_*.txt")))
    if not param_paths:
        return
    x, y = np.meshgrid(np.linspace(0, 2 * np.pi, cfg.num_bins), np.linspace(0, 2 * np.pi, cfg.num_bins))
    for path in param_paths:
        params = np.loadtxt(path)
        from util import get_total_bias_2d
        total_bias = get_total_bias_2d(x, y, params)
        name = os.path.splitext(os.path.basename(path))[0]
        plot_fes(total_bias, os.path.join(out_dir, f"{name}_bias.png"), title=f"Bias {name}")


def plot_metaD_energy(run_dir, out_dir):
    paths = sorted(glob.glob(os.path.join(run_dir, "visited_states", "*_metaD_potential_energy.npy")))
    for path in paths:
        energy = np.load(path)
        name = os.path.splitext(os.path.basename(path))[0]
        plt.figure()
        plt.plot(energy)
        plt.xlabel("Step")
        plt.ylabel("Potential energy (kJ/mol)")
        plt.title(name)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"{name}.png"))
        plt.close()


def _iter_episode_cv2d_csvs(traj_dir):
    pattern = os.path.join(traj_dir, "progressive_traj_ep_*_cv1_cv2_2d.csv")
    paths = sorted(glob.glob(pattern))
    for path in paths:
        name = os.path.basename(path)
        m = re.search(r"ep_(\d+)_cv1_cv2_2d\.csv$", name)
        if not m:
            continue
        yield int(m.group(1)), path


def _load_episode_bias_terms(meta_dir, ep_idx):
    meta_path = os.path.join(meta_dir, f"episode_{ep_idx:04d}.json")
    if not os.path.exists(meta_path):
        return []
    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
    except Exception:
        return []
    raw = payload.get("bias_log", [])
    terms = []
    for entry in raw:
        if len(entry) < 7:
            continue
        kind = str(entry[1])
        amp = float(entry[2])
        c1 = float(entry[3])
        sx = float(entry[5])
        if kind in ("gaussian2d", "bias2d") or entry[6] is not None:
            c2 = float(entry[4])
            sy = float(entry[6])
            terms.append({"kind": "2d", "amp": amp, "c1": c1, "c2": c2, "sx": sx, "sy": sy})
        else:
            c2 = None if entry[4] is None else float(entry[4])
            terms.append({"kind": "1d", "amp": amp, "c1": c1, "c2": c2, "sx": sx, "sy": None})
    return terms


def _edges_from_centers(centers):
    centers = np.asarray(centers, dtype=float)
    if centers.size < 2:
        c0 = float(centers[0]) if centers.size else 0.0
        width = 0.5
        return np.array([c0 - width, c0 + width], dtype=float)
    edges = np.empty(centers.size + 1, dtype=float)
    edges[1:-1] = 0.5 * (centers[:-1] + centers[1:])
    edges[0] = centers[0] - (centers[1] - centers[0]) / 2.0
    edges[-1] = centers[-1] + (centers[-1] - centers[-2]) / 2.0
    return edges


def _bias_surface_from_terms(terms, cfg):
    if not terms:
        return None, None, None
    centers1 = np.array([t["c1"] for t in terms], dtype=float)
    widths1 = np.array([max(1e-6, t["sx"]) for t in terms], dtype=float)

    pad = float(getattr(cfg, "BIAS_PROFILE_PAD_SIGMA", 3.0))
    bins = int(getattr(cfg, "BIAS_PROFILE_BINS", 250))
    lo1 = float(np.min(centers1 - pad * widths1))
    hi1 = float(np.max(centers1 + pad * widths1))
    terms_2d = [t for t in terms if t["kind"] == "2d"]
    if terms_2d:
        centers2 = np.array([t["c2"] for t in terms_2d], dtype=float)
        widths2 = np.array([max(1e-6, t["sy"]) for t in terms_2d], dtype=float)
        lo2 = float(np.min(centers2 - pad * widths2))
        hi2 = float(np.max(centers2 + pad * widths2))
    else:
        lo2 = min(float(getattr(cfg, "TARGET2_MIN", 0.0)), float(getattr(cfg, "FINAL_TARGET2", 0.0)) - 1.0)
        hi2 = max(float(getattr(cfg, "TARGET2_MAX", 1.0)), float(getattr(cfg, "CURRENT_DISTANCE2", 1.0)) + 1.0)

    x = np.linspace(lo1, hi1, bins)
    y = np.linspace(lo2, hi2, bins)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X, dtype=np.float64)
    for term in terms:
        sx = max(1e-6, float(term["sx"]))
        if term["kind"] == "2d":
            sy = max(1e-6, float(term["sy"]))
            Z += float(term["amp"]) * np.exp(-((X - float(term["c1"])) ** 2) / (2.0 * sx ** 2)
                                             -((Y - float(term["c2"])) ** 2) / (2.0 * sy ** 2))
        else:
            Z += float(term["amp"]) * np.exp(-((X - float(term["c1"])) ** 2) / (2.0 * sx ** 2))
    return x, y, Z


def _fes_from_samples(cv1, cv2, xedges, yedges, temperature_k, epsilon=1e-12):
    counts = np.histogram2d(cv1, cv2, bins=[xedges, yedges])[0]
    total = float(np.sum(counts))
    if total <= 0:
        return np.zeros_like(counts), counts
    prob = counts / total
    kT = KB_KCAL * float(temperature_k)
    fes = -kT * np.log(prob + epsilon)
    if np.isfinite(fes).any():
        fes = fes - np.nanmin(fes)
    return fes, counts


def plot_episode_bias_potential(run_dir, cfg, max_episodes=50, stride=1):
    traj_dir = os.path.join(ROOT_DIR, "results_PPO", "full_trajectories")
    meta_dir = os.path.join(ROOT_DIR, "results_PPO", "episode_meta")
    if not os.path.isdir(traj_dir):
        return
    out_dir = os.path.join(run_dir, "figs", "analysis")
    os.makedirs(out_dir, exist_ok=True)

    temp_k = 300.0
    if cfg is not None and hasattr(cfg, "T"):
        try:
            unit_mod = cfg.unit if hasattr(cfg, "unit") else getattr(cfg, "u", None)
            if unit_mod is not None and hasattr(cfg.T, "value_in_unit"):
                temp_k = float(cfg.T.value_in_unit(unit_mod.kelvin))
        except Exception:
            pass

    plotted = 0
    for ep_idx, csv_path in _iter_episode_cv2d_csvs(traj_dir):
        if max_episodes is not None and plotted >= max_episodes:
            break
        try:
            arr = np.loadtxt(csv_path, delimiter=",", skiprows=1)
        except Exception:
            continue
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        if arr.shape[1] < 3:
            continue
        if stride > 1:
            arr = arr[::stride]
        time_ps = arr[:, 0]
        cv1 = arr[:, 1]
        cv2 = arr[:, 2]
        if cv1.size == 0:
            continue

        terms = _load_episode_bias_terms(meta_dir, ep_idx)
        x = y = Z = None
        if terms and cfg is not None:
            x, y, Z = _bias_surface_from_terms(terms, cfg)

        if x is None or y is None:
            # fallback grid from trajectory extent
            margin = 0.5
            bins = int(getattr(cfg, "BIAS_PROFILE_BINS", 250)) if cfg is not None else 200
            x = np.linspace(float(np.min(cv1) - margin), float(np.max(cv1) + margin), bins)
            y = np.linspace(float(np.min(cv2) - margin), float(np.max(cv2) + margin), bins)
            Z = np.zeros((y.size, x.size), dtype=float)

        xedges = _edges_from_centers(x)
        yedges = _edges_from_centers(y)
        fes, counts = _fes_from_samples(cv1, cv2, xedges, yedges, temp_k)
        fes_masked = np.array(fes, copy=True)
        fes_masked[counts == 0] = np.nan
        finite_fes = np.isfinite(fes_masked)
        if np.any(finite_fes):
            vmin_fes = float(np.nanpercentile(fes_masked[finite_fes], 5))
            vmax_fes = float(np.nanpercentile(fes_masked[finite_fes], 95))
        else:
            vmin_fes = vmax_fes = None

        xcenters = 0.5 * (xedges[:-1] + xedges[1:])
        ycenters = 0.5 * (yedges[:-1] + yedges[1:])

        fig = plt.figure(figsize=(7.8, 6.0))
        gs = fig.add_gridspec(1, 4, width_ratios=[1.0, 0.02, 0.065, 0.065], wspace=0.10)
        ax = fig.add_subplot(gs[0, 0])
        spacer_ax = fig.add_subplot(gs[0, 1])
        spacer_ax.axis("off")
        cax_bias = fig.add_subplot(gs[0, 2])
        cax_fes = fig.add_subplot(gs[0, 3])

        fes_cmap = plt.cm.get_cmap("coolwarm").copy()
        fes_cmap.set_bad(color="white", alpha=0.0)
        im_fes = ax.imshow(
            fes_masked.T,
            origin="lower",
            aspect="auto",
            cmap=fes_cmap,
            extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
            vmin=vmin_fes,
            vmax=vmax_fes,
        )

        # FES contours
        if np.any(finite_fes):
            levels = np.percentile(fes_masked[finite_fes], [50, 75, 90, 95])
            levels = sorted(set(float(l) for l in levels if np.isfinite(l)))
            if len(levels) >= 2:
                ax.contour(
                    xcenters,
                    ycenters,
                    fes_masked.T,
                    levels=levels,
                    colors="white",
                    linewidths=0.8,
                    alpha=0.8,
                )

        # Bias surface overlay (semi-transparent)
        zb = Z[Z > 0]
        if zb.size > 0:
            vmax_bias = float(np.percentile(zb, 99))
        else:
            vmax_bias = None
        levels_bias = None
        if zb.size > 0:
            levels_bias = np.linspace(0, vmax_bias, 8)
        cf = ax.contourf(
            x,
            y,
            Z,
            levels=levels_bias,
            cmap="magma",
            alpha=0.35,
            vmin=0.0,
            vmax=vmax_bias,
        )
        if zb.size > 0:
            levels_b = np.percentile(zb, [50, 75, 90, 95])
            levels_b = sorted(set(float(l) for l in levels_b if np.isfinite(l)))
            if len(levels_b) >= 2:
                ax.contour(
                    x,
                    y,
                    Z,
                    levels=levels_b,
                    colors="black",
                    linewidths=0.7,
                    alpha=0.7,
                )

        # Trajectory overlay
        ax.plot(cv1, cv2, color="black", linewidth=0.9, alpha=0.7)
        ax.scatter(cv1, cv2, s=8, color="black", alpha=0.6, linewidths=0)

        # Start/Target markers
        ax.scatter([cv1[0]], [cv2[0]], s=40, color="green", edgecolors="white", linewidths=0.6, zorder=5, label="Start")
        if cfg is not None and hasattr(cfg, "TARGET_MIN") and hasattr(cfg, "TARGET_MAX"):
            tgt1 = 0.5 * (float(cfg.TARGET_MIN) + float(cfg.TARGET_MAX))
        else:
            tgt1 = float(cv1[-1])
        if cfg is not None and hasattr(cfg, "TARGET2_MIN") and hasattr(cfg, "TARGET2_MAX"):
            tgt2 = 0.5 * (float(cfg.TARGET2_MIN) + float(cfg.TARGET2_MAX))
        else:
            tgt2 = float(cv2[-1])
        ax.scatter([tgt1], [tgt2], s=70, marker="*", color="red", edgecolors="white", linewidths=0.6, zorder=6, label="Target")

        # Target zones (if available)
        if cfg is not None and hasattr(cfg, "TARGET_MIN") and hasattr(cfg, "TARGET_MAX"):
            ax.axvspan(cfg.TARGET_MIN, cfg.TARGET_MAX, alpha=0.10, color="white")
        if cfg is not None and hasattr(cfg, "TARGET2_MIN") and hasattr(cfg, "TARGET2_MAX"):
            ax.axhspan(cfg.TARGET2_MIN, cfg.TARGET2_MAX, alpha=0.08, color="white")

        ax.set_title(f"Episode {ep_idx:04d}")
        ax.set_xlabel(getattr(cfg, "CV1_AXIS_LABEL", f"{getattr(cfg, 'CV1_LABEL', 'CV1')} (A)") if cfg else "CV1 (A)")
        ax.set_ylabel(getattr(cfg, "CV2_AXIS_LABEL", f"{getattr(cfg, 'CV2_LABEL', 'CV2')} (A)") if cfg else "CV2 (A)")
        ax.legend(loc="upper right", frameon=True)

        cbar_bias = fig.colorbar(cf, cax=cax_bias)
        cbar_bias.set_label("Bias (kcal/mol)")
        cbar_bias.ax.yaxis.set_label_position("left")
        cbar_bias.ax.yaxis.tick_left()
        cbar_bias.ax.tick_params(pad=2)
        cbar_bias.outline.set_linewidth(0.8)
        cbar_fes = fig.colorbar(im_fes, cax=cax_fes)
        cbar_fes.set_label("Potential (FES)")
        cbar_fes.ax.yaxis.set_label_position("right")
        cbar_fes.ax.yaxis.tick_right()
        cbar_fes.ax.tick_params(pad=2)
        cbar_fes.outline.set_linewidth(0.8)
        out_path = os.path.join(out_dir, f"episode_{ep_idx:04d}_fes_bias_traj.png")
        fig.savefig(out_path, dpi=220)
        plt.close(fig)
        plotted += 1


def plot_all_trajectories(run_dir, cfg, stride=5, max_episodes=None):
    traj_dir = os.path.join(ROOT_DIR, "results_PPO", "full_trajectories")
    if not os.path.isdir(traj_dir):
        return
    out_dir = os.path.join(run_dir, "figs", "analysis")
    os.makedirs(out_dir, exist_ok=True)

    entries = list(_iter_episode_cv2d_csvs(traj_dir))
    if not entries:
        return
    if max_episodes is not None:
        entries = entries[:max_episodes]

    episodes = [ep for ep, _ in entries]
    ep_min = min(episodes)
    ep_max = max(episodes) if len(episodes) > 1 else ep_min + 1
    cmap = plt.get_cmap("turbo")
    norm = plt.Normalize(vmin=ep_min, vmax=ep_max)

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    for ep_idx, csv_path in entries:
        try:
            arr = np.loadtxt(csv_path, delimiter=",", skiprows=1)
        except Exception:
            continue
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        if arr.shape[1] < 3:
            continue
        if stride > 1:
            arr = arr[::stride]
        cv1 = arr[:, 1]
        cv2 = arr[:, 2]
        if cv1.size == 0:
            continue
        ax.plot(cv1, cv2, color=cmap(norm(ep_idx)), linewidth=0.9, alpha=0.6)

    if cfg is not None and hasattr(cfg, "TARGET_MIN") and hasattr(cfg, "TARGET_MAX"):
        ax.axvspan(cfg.TARGET_MIN, cfg.TARGET_MAX, alpha=0.12, color="grey")
    if cfg is not None and hasattr(cfg, "TARGET2_MIN") and hasattr(cfg, "TARGET2_MAX"):
        ax.axhspan(cfg.TARGET2_MIN, cfg.TARGET2_MAX, alpha=0.10, color="grey")

    ax.set_xlabel(getattr(cfg, "CV1_AXIS_LABEL", f"{getattr(cfg, 'CV1_LABEL', 'CV1')} (A)") if cfg else "CV1 (A)")
    ax.set_ylabel(getattr(cfg, "CV2_AXIS_LABEL", f"{getattr(cfg, 'CV2_LABEL', 'CV2')} (A)") if cfg else "CV2 (A)")
    ax.set_title("All Episode Trajectories (CV1 vs CV2)")
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label("Episode")
    fig.tight_layout()
    out_path = os.path.join(out_dir, "all_trajectories_cv1_cv2.png")
    fig.savefig(out_path, dpi=220)
    plt.close(fig)

def plot_cv_trajectories(out_dir, traj_glob, top_path, cfg, max_plots=50, stride=1):
    if mda is None:
        return
    if not os.path.exists(top_path):
        return

    dcd_paths = sorted(glob.glob(traj_glob))
    if not dcd_paths:
        return
    dcd_paths = dcd_paths[:max_plots]

    atom1 = int(getattr(cfg, "ATOM1_INDEX", 0))
    atom2 = int(getattr(cfg, "ATOM2_INDEX", 0))
    atom3 = int(getattr(cfg, "ATOM3_INDEX", 0))
    atom4 = int(getattr(cfg, "ATOM4_INDEX", 0))

    unit_mod = None
    if hasattr(cfg, "unit"):
        unit_mod = cfg.unit
    elif hasattr(cfg, "u"):
        unit_mod = cfg.u
    if unit_mod is None or not hasattr(cfg, "stepsize"):
        return
    stepsize_ps = float(cfg.stepsize.value_in_unit(unit_mod.picoseconds))
    report_interval = int(getattr(cfg, "DCD_REPORT_INTERVAL", getattr(cfg, "dcdfreq_mfpt", 1)))
    dt_ps = stepsize_ps * report_interval * max(1, stride)

    for dcd_path in dcd_paths:
        u = mda.Universe(top_path, dcd_path)
        cv1 = []
        cv2 = []
        for i, ts in enumerate(u.trajectory):
            if stride > 1 and (i % stride) != 0:
                continue
            pos = u.atoms.positions.astype(np.float64)
            cv1.append(float(np.linalg.norm(pos[atom1] - pos[atom2])))
            cv2.append(float(np.linalg.norm(pos[atom3] - pos[atom4])))
        cv1 = np.asarray(cv1, dtype=np.float32)
        cv2 = np.asarray(cv2, dtype=np.float32)

        time_ps = np.arange(len(cv1), dtype=np.float32) * dt_ps
        base = os.path.splitext(os.path.basename(dcd_path))[0]

        # CV1
        plt.figure(figsize=(9, 4.5))
        plt.plot(time_ps, cv1, linewidth=1.6)
        plt.xlabel("Time (ps)")
        label_cv1 = getattr(cfg, "CV1_AXIS_LABEL", f"{getattr(cfg, 'CV1_LABEL', 'CV1 distance')} (A)")
        plt.ylabel(label_cv1)
        plt.title(f"{base} CV1 trajectory")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"{base}_cv1.png"))
        plt.close()

        # CV2
        plt.figure(figsize=(9, 4.5))
        plt.plot(time_ps, cv2, linewidth=1.6)
        plt.xlabel("Time (ps)")
        label_cv2 = getattr(cfg, "CV2_AXIS_LABEL", f"{getattr(cfg, 'CV2_LABEL', 'CV2 distance')} (A)")
        plt.ylabel(label_cv2)
        plt.title(f"{base} CV2 trajectory")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"{base}_cv2.png"))
        plt.close()

        # 2D path
        plt.figure(figsize=(6.5, 5.5))
        sc = plt.scatter(cv1, cv2, c=time_ps, s=8, cmap="viridis")
        cbar = plt.colorbar(sc)
        cbar.set_label("Time (ps)")
        label_cv1 = getattr(cfg, "CV1_AXIS_LABEL", f"{getattr(cfg, 'CV1_LABEL', 'CV1 distance')} (A)")
        label_cv2 = getattr(cfg, "CV2_AXIS_LABEL", f"{getattr(cfg, 'CV2_LABEL', 'CV2 distance')} (A)")
        plt.xlabel(label_cv1)
        plt.ylabel(label_cv2)
        plt.title(f"{base} CV1 vs CV2")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"{base}_cv1_cv2.png"))
        plt.close()

        # CSV
        csv_path = os.path.join(out_dir, f"{base}_cv.csv")
        np.savetxt(
            csv_path,
            np.c_[time_ps, cv1, cv2],
            delimiter=",",
            header="time_ps,cv1_A,cv2_A",
            comments="",
        )


def main():
    parser = argparse.ArgumentParser(description="Post-process PPO trajectories and generate plots.")
    parser.add_argument("--config-module", default=None, help="Config module to use (default: combined_2d or config).")
    parser.add_argument("--run", dest="run_dir", default=None, help="Existing analysis run directory.")
    parser.add_argument("--runs-root", default=None, help="Root folder that contains analysis_runs/")
    parser.add_argument("--traj-glob", default=None, help="Glob for DCD trajectories.")
    parser.add_argument("--top", default=None, help="Topology file (PSF/PDB).")
    parser.add_argument("--max-traj-plots", type=int, default=50)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--episode-surfaces", action="store_true",
                        help="Generate per-episode FES + bias surface plots with trajectories.")
    parser.add_argument("--max-episode-plots", type=int, default=50,
                        help="Maximum number of episodes to render in per-episode plots.")
    parser.add_argument("--episode-plot-stride", type=int, default=1,
                        help="Stride for per-episode CV1/CV2 samples in bias/FES plots.")
    parser.add_argument("--all-trajectories", action="store_true",
                        help="Generate a combined plot of all episode trajectories.")
    parser.add_argument("--all-traj-stride", type=int, default=5,
                        help="Stride for combined trajectories plot.")
    args = parser.parse_args()

    cfg_name = args.config_module or ("combined_2d" if os.path.exists(os.path.join(ROOT_DIR, "combined_2d.py")) else "config")
    try:
        cfg = importlib.import_module(cfg_name)
    except Exception:
        cfg = importlib.import_module("config")

    runs_root = _resolve_runs_root(cfg, args.runs_root)
    run_dir = args.run_dir
    if run_dir is not None and not os.path.isabs(run_dir):
        run_dir = os.path.join(ROOT_DIR, run_dir)

    if run_dir is None:
        latest = find_latest_run(runs_root)
        if latest is None:
            time_tag = run_utils.default_time_tag()
            run_dir = run_utils.prepare_run_dir(time_tag, root=runs_root)
            run_utils.write_run_metadata(
                run_dir,
                {
                    "script": "analysis/post_process.py",
                    "config_module": cfg_name,
                    "config": _snapshot_cfg(cfg),
                },
            )
        else:
            run_dir = latest

    out_dir = os.path.join(run_dir, "figs", "analysis")
    os.makedirs(out_dir, exist_ok=True)

    fes = load_fes(run_dir)
    if fes is not None:
        plot_fes(fes, os.path.join(out_dir, "fes.png"))

    plot_total_steps(run_dir, out_dir, cfg)
    plot_reconstructed_fes(run_dir, out_dir, cfg)
    plot_bias_surfaces(run_dir, out_dir, cfg)
    plot_metaD_energy(run_dir, out_dir)

    traj_glob = args.traj_glob or _resolve_default_traj_glob(cfg)
    top_path = args.top or getattr(cfg, "psf_file", None)
    top_path = os.path.join(ROOT_DIR, top_path) if top_path and not os.path.isabs(top_path) else top_path
    plot_cv_trajectories(out_dir, traj_glob, top_path, cfg, max_plots=args.max_traj_plots, stride=args.stride)

    if args.episode_surfaces:
        plot_episode_bias_potential(run_dir, cfg, max_episodes=args.max_episode_plots, stride=args.episode_plot_stride)
    if args.all_trajectories:
        plot_all_trajectories(run_dir, cfg, stride=args.all_traj_stride, max_episodes=args.max_episode_plots)

    run_utils.cleanup_empty_dirs(run_dir)


if __name__ == "__main__":
    main()
