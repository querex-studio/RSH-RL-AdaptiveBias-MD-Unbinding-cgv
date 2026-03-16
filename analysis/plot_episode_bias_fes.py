import argparse
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
import numpy as np

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

import combined_2d as config

KB_KCAL = 0.00198720425864083


def _episode_paths(episode_num: int):
    traj_csv = os.path.join(
        config.RESULTS_DIR,
        "full_trajectories",
        f"progressive_traj_ep_{episode_num:04d}_cv1_cv2_2d.csv",
    )
    meta_json = os.path.join(
        config.RESULTS_DIR,
        "episode_meta",
        f"episode_{episode_num:04d}.json",
    )
    return traj_csv, meta_json


def _load_traj(csv_path):
    arr = np.loadtxt(csv_path, delimiter=",", skiprows=1)
    if arr.ndim != 2 or arr.shape[1] < 3:
        raise ValueError(f"Unexpected trajectory CSV format: {csv_path}")
    return arr[:, 0], arr[:, 1], arr[:, 2]


def _load_bias_log(meta_path):
    with open(meta_path, "r") as fh:
        meta = json.load(fh)
    return meta


def _frames_per_action():
    return max(1, int(config.propagation_step // max(1, int(config.dcdfreq_mfpt))))


def _bias_value_from_terms(cv1_A, cv2_A, terms):
    val = 0.0
    for _, _, amp_kcal, center1_A, center2_A, sigma_x_A, sigma_y_A in terms:
        sx = max(1e-6, float(sigma_x_A))
        sy = max(1e-6, float(sigma_y_A))
        val += float(amp_kcal) * np.exp(
            -((float(cv1_A) - float(center1_A)) ** 2) / (2.0 * sx * sx)
            -((float(cv2_A) - float(center2_A)) ** 2) / (2.0 * sy * sy)
        )
    return val


def _frame_biases(cv1, cv2, bias_log):
    frames_per_action = _frames_per_action()
    frame_bias = np.zeros(len(cv1), dtype=np.float64)
    if not bias_log:
        return frame_bias

    for idx in range(len(cv1)):
        action_idx = int(idx // frames_per_action) + 1
        active_terms = [row for row in bias_log if int(row[0]) <= action_idx]
        frame_bias[idx] = _bias_value_from_terms(cv1[idx], cv2[idx], active_terms)
    return frame_bias


def _grid_bias(x_grid, y_grid, bias_log):
    X, Y = np.meshgrid(x_grid, y_grid)
    Z = np.zeros_like(X, dtype=np.float64)
    for _, _, amp_kcal, center1_A, center2_A, sigma_x_A, sigma_y_A in bias_log:
        sx = max(1e-6, float(sigma_x_A))
        sy = max(1e-6, float(sigma_y_A))
        Z += float(amp_kcal) * np.exp(
            -((X - float(center1_A)) ** 2) / (2.0 * sx * sx)
            -((Y - float(center2_A)) ** 2) / (2.0 * sy * sy)
        )
    return Z


def _bias_surface_like_profile(bias_log):
    if not bias_log:
        x = np.linspace(0.0, 10.0, int(getattr(config, "BIAS_PROFILE_BINS", 250)))
        y = np.linspace(
            min(float(config.TARGET2_MIN), float(config.FINAL_TARGET2) - 1.0),
            max(float(config.TARGET2_MAX), float(config.CURRENT_DISTANCE2) + 1.0),
            int(getattr(config, "BIAS_PROFILE_BINS", 250)),
        )
        return x, y, np.zeros((len(y), len(x)), dtype=np.float64)

    amps = np.array([float(row[2]) for row in bias_log], dtype=np.float64)
    centers1 = np.array([float(row[3]) for row in bias_log], dtype=np.float64)
    centers2 = np.array([float(row[4]) for row in bias_log], dtype=np.float64)
    widths1 = np.array([max(1e-6, float(row[5])) for row in bias_log], dtype=np.float64)
    widths2 = np.array([max(1e-6, float(row[6])) for row in bias_log], dtype=np.float64)

    pad = float(getattr(config, "BIAS_PROFILE_PAD_SIGMA", 3.0))
    bins = int(getattr(config, "BIAS_PROFILE_BINS", 250))
    lo1 = float(np.min(centers1 - pad * widths1))
    hi1 = float(np.max(centers1 + pad * widths1))
    lo2 = float(np.min(centers2 - pad * widths2))
    hi2 = float(np.max(centers2 + pad * widths2))

    x = np.linspace(
        min(lo1, float(config.CURRENT_DISTANCE) - 1.0, 0.0),
        max(hi1, float(config.FINAL_TARGET) + 1.0, 10.0),
        bins,
    )
    y = np.linspace(
        min(lo2, float(config.FINAL_TARGET2) - 1.0),
        max(hi2, float(config.CURRENT_DISTANCE2) + 1.0),
        bins,
    )
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X, dtype=np.float64)
    for amp, c1, c2, sx, sy in zip(amps, centers1, centers2, widths1, widths2):
        Z += float(amp) * np.exp(
            -((X - float(c1)) ** 2) / (2.0 * float(sx) ** 2)
            -((Y - float(c2)) ** 2) / (2.0 * float(sy) ** 2)
        )
    return x, y, Z


def _plot_limits(cv1, cv2, target_min, target_max, target2_min, target2_max):
    x_all = np.array([float(np.min(cv1)), float(np.max(cv1)), float(target_min), float(target_max)], dtype=np.float64)
    y_all = np.array([float(np.min(cv2)), float(np.max(cv2)), float(target2_min), float(target2_max)], dtype=np.float64)
    x_span = max(0.5, float(np.max(x_all) - np.min(x_all)))
    y_span = max(0.5, float(np.max(y_all) - np.min(y_all)))
    x_pad = max(0.25, 0.12 * x_span)
    y_pad = max(0.25, 0.12 * y_span)
    return (
        float(np.min(x_all) - x_pad),
        float(np.max(x_all) + x_pad),
        float(np.min(y_all) - y_pad),
        float(np.max(y_all) + y_pad),
    )


def _reweighted_fes(cv1, cv2, frame_bias_kcal, temperature, bins):
    beta = 1.0 / (KB_KCAL * float(temperature))
    weights = np.exp(beta * frame_bias_kcal)

    x_edges = np.linspace(float(np.min(cv1)), float(np.max(cv1)), int(bins) + 1)
    y_edges = np.linspace(float(np.min(cv2)), float(np.max(cv2)), int(bins) + 1)
    counts, _, _ = np.histogram2d(cv1, cv2, bins=[x_edges, y_edges], weights=weights)
    visits, _, _ = np.histogram2d(cv1, cv2, bins=[x_edges, y_edges])

    prob = counts / max(np.sum(counts), 1.0)
    with np.errstate(divide="ignore"):
        fes = -(KB_KCAL * float(temperature)) * np.log(prob)
    if np.isfinite(fes).any():
        fes = fes - np.nanmin(fes[np.isfinite(fes)])
    fes[visits <= 0] = np.nan

    x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
    y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])
    return x_centers, y_centers, fes.T, visits.T


def _finite_level_range(values, n_levels, low_q=None, high_q=None):
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return None
    vmin = float(np.nanmin(finite)) if low_q is None else float(np.nanpercentile(finite, low_q))
    vmax = float(np.nanmax(finite)) if high_q is None else float(np.nanpercentile(finite, high_q))
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        return None
    return np.linspace(vmin, vmax, int(n_levels))


def _grid_edges(centers):
    centers = np.asarray(centers, dtype=np.float64)
    if centers.ndim != 1 or centers.size == 0:
        raise ValueError("Grid centers must be a non-empty 1D array.")
    if centers.size == 1:
        step = 0.5
        return np.array([centers[0] - step, centers[0] + step], dtype=np.float64)

    mids = 0.5 * (centers[:-1] + centers[1:])
    left = centers[0] - 0.5 * (centers[1] - centers[0])
    right = centers[-1] + 0.5 * (centers[-1] - centers[-2])
    return np.concatenate([[left], mids, [right]])


def _time_colored_line(ax, x, y, t):
    if len(x) < 2:
        return None, None

    points = np.column_stack([x, y]).reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    norm = Normalize(vmin=float(np.min(t)), vmax=float(np.max(t)))

    line = LineCollection(
        segments,
        cmap="viridis",
        norm=norm,
        linewidths=0.95,
        alpha=0.98,
        capstyle="round",
        joinstyle="round",
        zorder=5,
    )
    line.set_array(np.asarray(t[:-1], dtype=np.float64))
    ax.add_collection(line)
    return line, norm


def make_plot(episode_num: int, temperature: float = 300.0, bins: int = 120):
    traj_csv, meta_json = _episode_paths(episode_num)
    if not os.path.exists(traj_csv):
        raise FileNotFoundError(traj_csv)
    if not os.path.exists(meta_json):
        raise FileNotFoundError(meta_json)

    time_ps, cv1, cv2 = _load_traj(traj_csv)
    meta = _load_bias_log(meta_json)
    bias_log = meta.get("bias_log", [])
    target_center = float(meta.get("target_center_A", config.TARGET_CENTER))
    target_zone = meta.get("target_zone", [config.TARGET_MIN, config.TARGET_MAX])
    target_min = float(target_zone[0])
    target_max = float(target_zone[1])

    frame_bias = _frame_biases(cv1, cv2, bias_log)
    x_grid, y_grid, fes, visits = _reweighted_fes(cv1, cv2, frame_bias, temperature, bins)
    hit_target = bool(
        (target_min <= cv1[-1] <= target_max)
        and (config.TARGET2_MIN <= cv2[-1] <= config.TARGET2_MAX)
    )
    status_text = "HIT target" if hit_target else "MISS target"
    status_color = "#1b7f3b" if hit_target else "#8b0000"

    out_dir = os.path.join(config.RESULTS_DIR, "analysis_runs", f"episode_{episode_num:04d}")
    os.makedirs(out_dir, exist_ok=True)
    out_png = os.path.join(out_dir, f"episode_{episode_num:04d}_reweighted_fes_bias_traj.png")

    fig = plt.figure(figsize=(10.4, 7.1), constrained_layout=True)
    gs = fig.add_gridspec(1, 4, width_ratios=[1.0, 0.055, 0.055, 0.055], wspace=0.16)
    ax = fig.add_subplot(gs[0, 0])
    cax_fes = fig.add_subplot(gs[0, 1])
    cax_bias = fig.add_subplot(gs[0, 2])
    cax_time = fig.add_subplot(gs[0, 3])
    ax.set_facecolor("#f5f1ea")

    x_edges = _grid_edges(x_grid)
    y_edges = _grid_edges(y_grid)
    x_min, x_max, y_min, y_max = _plot_limits(
        cv1,
        cv2,
        target_min,
        target_max,
        config.TARGET2_MIN,
        config.TARGET2_MAX,
    )
    bias_x, bias_y, total_bias = _bias_surface_like_profile(bias_log)

    fes_masked = np.ma.masked_invalid(fes)
    fes_map = ax.pcolormesh(
        x_edges,
        y_edges,
        fes_masked,
        shading="auto",
        cmap="YlGnBu_r",
        antialiased=True,
        linewidth=0.0,
        rasterized=True,
        zorder=1,
    )

    bias_positive = total_bias[np.isfinite(total_bias) & (total_bias > 0.0)]
    bias_low = None
    if bias_positive.size > 0:
        bias_low = float(np.nanpercentile(bias_positive, 8.0))
    if bias_low is None or not np.isfinite(bias_low) or bias_low <= 0.0:
        bias_low = 1e-8
    bias_display = np.ma.masked_less(np.array(total_bias, copy=True), bias_low)
    bias_cmap = matplotlib.colormaps["coolwarm"].copy()
    bias_cmap.set_bad((1.0, 1.0, 1.0, 0.0))
    bias_map = ax.imshow(
        bias_display,
        origin="lower",
        aspect="auto",
        cmap=bias_cmap,
        extent=[float(np.min(bias_x)), float(np.max(bias_x)), float(np.min(bias_y)), float(np.max(bias_y))],
        interpolation="bicubic",
        alpha=0.26,
        zorder=2,
    )
    bias_levels = _finite_level_range(np.asarray(bias_display.filled(np.nan), dtype=np.float64), n_levels=9, low_q=18.0, high_q=98.8)
    if bias_levels is not None:
        ax.contour(
            bias_x,
            bias_y,
            bias_display,
            levels=bias_levels,
            cmap=bias_cmap,
            linewidths=0.60,
            alpha=0.30,
            zorder=3,
        )

    ax.plot(cv1, cv2, color="#111111", linewidth=1.55, alpha=0.14, zorder=4)
    time_line, _ = _time_colored_line(ax, cv1, cv2, time_ps)
    ax.scatter([cv1[0]], [cv2[0]], s=84, color="#2a9d8f", edgecolor="white", linewidth=0.8, label="Start", zorder=7)
    ax.scatter(
        [target_center],
        [config.TARGET2_CENTER],
        s=180,
        marker="*",
        color="#e63946",
        edgecolor="white",
        linewidth=0.9,
        label="Target",
        zorder=7,
    )
    ax.scatter([cv1[-1]], [cv2[-1]], s=90, color="#f4a261", edgecolor="white", linewidth=0.8, label="Final", zorder=7)

    ax.axvspan(target_min, target_max, alpha=0.05, color="#6c8ebf", zorder=1)
    ax.axhspan(config.TARGET2_MIN, config.TARGET2_MAX, alpha=0.05, color="#6c8ebf", zorder=1)
    ax.add_patch(
        Rectangle(
            (target_min, config.TARGET2_MIN),
            target_max - target_min,
            config.TARGET2_MAX - config.TARGET2_MIN,
            fill=False,
            edgecolor="#ef476f",
            linewidth=1.25,
            linestyle="--",
            alpha=0.92,
            zorder=4,
        )
    )
    ax.text(
        0.02,
        0.98,
        status_text,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=13,
        fontweight="bold",
        color=status_color,
        bbox={"facecolor": "white", "alpha": 0.74, "edgecolor": "none", "pad": 4.5},
        zorder=8,
    )
    ax.set_xlabel("CV1 (atom 7799 - atom 7840 distance) (Å)")
    ax.set_ylabel("CV2 (atom 487 - atom 3789 distance) (Å)")
    ax.set_title(
        f"Episode {episode_num} | {status_text} | final CV1={cv1[-1]:.3f} | frames={len(cv1)}"
    )
    ax.set_xlim(0.0, 10.0)
    ax.set_ylim(y_min, y_max)
    ax.grid(color="#d8d4cf", linewidth=0.45, alpha=0.22)
    ax.legend(loc="upper right", frameon=True, framealpha=0.90, facecolor="white", edgecolor="#d9d4cc")

    cbar_fes = fig.colorbar(fes_map, cax=cax_fes)
    cbar_fes.set_label("Reweighted FES (kcal/mol)")
    if bias_map is None:
        bias_map = fes_map
    cbar_bias = fig.colorbar(bias_map, cax=cax_bias)
    cbar_bias.set_label("Bias (sum of hills)")
    cbar_time = fig.colorbar(time_line, cax=cax_time)
    cbar_time.set_label("Time (ps)")

    fig.savefig(out_png, dpi=220)
    plt.close(fig)
    return out_png


def main():
    parser = argparse.ArgumentParser(description="Plot reweighted FES + bias + trajectory for one episode.")
    parser.add_argument("--episode", type=int, required=True, help="Episode number, e.g. 4")
    parser.add_argument("--temperature", type=float, default=300.0)
    parser.add_argument("--bins", type=int, default=120)
    args = parser.parse_args()

    out_png = make_plot(args.episode, temperature=args.temperature, bins=args.bins)
    print(f"Saved plot: {out_png}")


if __name__ == "__main__":
    main()
