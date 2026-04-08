import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import numpy as np

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

import combined_2d as config
from analysis.plot_episode_bias_fes import (
    KB_KCAL,
    _bias_surface_like_profile,
    _episode_paths,
    _frame_biases,
    _axis_label,
    _load_bias_log,
    _load_traj,
)


def _histogram_fes(cv1, cv2, temperature, bins, weights=None, x_limits=None, y_limits=None):
    if x_limits is None:
        x_limits = (float(np.min(cv1)), float(np.max(cv1)))
    if y_limits is None:
        y_limits = (float(np.min(cv2)), float(np.max(cv2)))

    x_edges = np.linspace(float(x_limits[0]), float(x_limits[1]), int(bins) + 1)
    y_edges = np.linspace(float(y_limits[0]), float(y_limits[1]), int(bins) + 1)
    counts, _, _ = np.histogram2d(cv1, cv2, bins=[x_edges, y_edges], weights=weights)
    visits, _, _ = np.histogram2d(cv1, cv2, bins=[x_edges, y_edges])

    prob = counts / max(float(np.sum(counts)), 1.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        fes = -(KB_KCAL * float(temperature)) * np.log(prob)
    if np.isfinite(fes).any():
        fes -= np.nanmin(fes[np.isfinite(fes)])
    fes[visits <= 0] = np.nan

    x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
    y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])
    return x_centers, y_centers, fes.T


def _sample_surface_nearest(x_vals, y_vals, x_grid, y_grid, surface, fallback_z):
    z = np.full(len(x_vals), float(fallback_z), dtype=np.float64)
    finite = np.isfinite(surface)
    if not np.any(finite):
        return z

    for idx, (xv, yv) in enumerate(zip(x_vals, y_vals)):
        ix = int(np.argmin(np.abs(x_grid - float(xv))))
        iy = int(np.argmin(np.abs(y_grid - float(yv))))
        if np.isfinite(surface[iy, ix]):
            z[idx] = float(surface[iy, ix])
            continue
        dist2 = (x_grid[None, :] - float(xv)) ** 2 + (y_grid[:, None] - float(yv)) ** 2
        dist2 = np.where(finite, dist2, np.inf)
        iy2, ix2 = np.unravel_index(int(np.argmin(dist2)), dist2.shape)
        if np.isfinite(surface[iy2, ix2]):
            z[idx] = float(surface[iy2, ix2])
    return z


def _combined_limits(cv1, cv2, target_x, target_y, bias_x, bias_y, bias_z):
    bias_pos = np.isfinite(bias_z) & (bias_z > 0.0)
    xs = [0.0, 10.0, float(np.min(cv1)), float(np.max(cv1)), float(target_x)]
    ys = [float(np.min(cv2)), float(np.max(cv2)), float(target_y)]
    if np.any(bias_pos):
        rows, cols = np.where(bias_pos)
        xs.extend([float(bias_x[np.min(cols)]), float(bias_x[np.max(cols)])])
        ys.extend([float(bias_y[np.min(rows)]), float(bias_y[np.max(rows)])])
    y_min = float(np.min(ys))
    y_max = float(np.max(ys))
    y_pad = max(0.25, 0.10 * max(0.5, y_max - y_min))
    return 0.0, 10.0, y_min - y_pad, y_max + y_pad


def _projection_mask(surface, low_q=10.0):
    positive = surface[np.isfinite(surface) & (surface > 0.0)]
    low = float(np.nanpercentile(positive, low_q)) if positive.size > 0 else 1e-8
    if not np.isfinite(low) or low <= 0.0:
        low = 1e-8
    masked = np.ma.masked_less(np.array(surface, copy=True), low)
    return masked


def _cleanup_old_outputs(out_dir, episode_num, keep_name):
    old_names = [
        f"episode_{int(episode_num):04d}_3d_biased_fes_bias_projection.png",
        f"episode_{int(episode_num):04d}_3d_biased_fes_unbiased_projection.png",
        f"episode_{int(episode_num):04d}_reweighted_fes_bias_traj_3d.png",
    ]
    for name in old_names:
        path = os.path.join(out_dir, name)
        if os.path.exists(path) and os.path.basename(path) != keep_name:
            os.remove(path)


def make_plot(episode_num: int, temperature: float = 300.0, bins: int = 140):
    traj_csv, meta_json = _episode_paths(episode_num)
    if not os.path.exists(traj_csv):
        raise FileNotFoundError(traj_csv)
    if not os.path.exists(meta_json):
        raise FileNotFoundError(meta_json)

    time_ps, cv1, cv2 = _load_traj(traj_csv)
    meta = _load_bias_log(meta_json)
    bias_log = meta.get("bias_log", [])
    frame_bias = _frame_biases(cv1, cv2, bias_log)
    beta = 1.0 / (KB_KCAL * float(temperature))
    weights = np.exp(beta * frame_bias)

    target_x = float(meta.get("target_center_A", config.TARGET_CENTER))
    target_y = float(meta.get("target2_center_A", config.TARGET2_CENTER))
    cv1_axis_label = _axis_label(meta, "cv1", getattr(config, "CV1_LABEL", "CV1"))
    cv2_axis_label = _axis_label(meta, "cv2", getattr(config, "CV2_LABEL", "CV2"))
    bias_x, bias_y, bias_surface = _bias_surface_like_profile(bias_log)
    x_min, x_max, y_min, y_max = _combined_limits(cv1, cv2, target_x, target_y, bias_x, bias_y, bias_surface)
    x_grid, y_grid, fes = _histogram_fes(
        cv1,
        cv2,
        temperature=temperature,
        bins=bins,
        weights=weights,
        x_limits=(x_min, x_max),
        y_limits=(y_min, y_max),
    )

    finite_fes = fes[np.isfinite(fes)]
    if finite_fes.size == 0:
        raise RuntimeError("No finite unbiased FES values were produced for 3D plotting.")

    Xs, Ys = np.meshgrid(x_grid, y_grid)
    z_min = float(np.nanmin(finite_fes))
    z_high = float(np.nanpercentile(finite_fes, 96.0))
    z_max = max(z_min + 1.0, z_high)
    z_range = max(1.0, z_max - z_min)
    z_floor = z_min - 0.35 * z_range
    z_top = z_max + 0.10 * z_range

    traj_z = _sample_surface_nearest(cv1, cv2, x_grid, y_grid, fes, fallback_z=z_min + 0.15)
    target_z = _sample_surface_nearest(
        np.array([target_x], dtype=np.float64),
        np.array([target_y], dtype=np.float64),
        x_grid,
        y_grid,
        fes,
        fallback_z=z_min + 0.2,
    )[0]

    fes_norm = colors.Normalize(vmin=z_min, vmax=z_max, clip=True)
    bias_masked = _projection_mask(bias_surface, low_q=10.0)
    bias_cmap = matplotlib.colormaps["coolwarm"].copy()
    bias_cmap.set_bad((1.0, 1.0, 1.0, 0.0))

    out_dir = os.path.join(config.RESULTS_DIR, "analysis_runs", f"episode_{episode_num:04d}")
    os.makedirs(out_dir, exist_ok=True)
    out_name = f"episode_{episode_num:04d}_3d_unbiased_fes_bias_projection.png"
    out_png = os.path.join(out_dir, out_name)

    fig = plt.figure(figsize=(12.8, 8.6), constrained_layout=True)
    ax = fig.add_subplot(111, projection="3d")

    surf = ax.plot_surface(
        Xs,
        Ys,
        np.ma.masked_invalid(fes),
        cmap="YlGnBu_r",
        norm=fes_norm,
        linewidth=0.0,
        antialiased=True,
        shade=False,
        alpha=0.98,
        rcount=min(180, fes.shape[0]),
        ccount=min(180, fes.shape[1]),
    )
    ax.contour(
        Xs,
        Ys,
        np.ma.masked_invalid(fes),
        zdir="z",
        offset=z_floor,
        levels=np.linspace(z_min, z_max, 10),
        colors="#4f5d75",
        linewidths=0.45,
        alpha=0.32,
    )
    proj = ax.contourf(
        bias_x,
        bias_y,
        bias_masked,
        zdir="z",
        offset=z_floor,
        levels=16,
        cmap=bias_cmap,
        alpha=0.88,
    )

    ax.plot(cv1, cv2, traj_z + 0.05, color="#111111", linewidth=1.7, alpha=0.92, zorder=5)
    time_scatter = ax.scatter(
        cv1,
        cv2,
        traj_z + 0.06,
        c=time_ps,
        cmap="viridis",
        s=12,
        alpha=0.98,
        edgecolors="none",
        depthshade=False,
        zorder=6,
    )
    ax.scatter([cv1[0]], [cv2[0]], [traj_z[0] + 0.12], s=94, color="#2a9d8f", edgecolor="white", linewidth=0.9, depthshade=False, zorder=7)
    ax.scatter([cv1[-1]], [cv2[-1]], [traj_z[-1] + 0.12], s=98, color="#f4a261", edgecolor="white", linewidth=0.9, depthshade=False, zorder=7)
    ax.scatter([target_x], [target_y], [target_z + 0.14], s=155, marker="*", color="#e63946", edgecolor="white", linewidth=1.0, depthshade=False, zorder=7)

    legend_handles = [
        Line2D([0], [0], color="#111111", linewidth=1.7, label="Trajectory"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#2a9d8f", markeredgecolor="white", markersize=8, label="Start"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#f4a261", markeredgecolor="white", markersize=8, label="Final"),
        Line2D([0], [0], marker="*", color="none", markerfacecolor="#e63946", markeredgecolor="white", markersize=12, label="Target"),
    ]
    ax.legend(handles=legend_handles, loc="upper left", bbox_to_anchor=(0.02, 0.98), frameon=True, framealpha=0.94)

    ax.set_title(f"Episode {episode_num} | 3D Unbiased FES + Bias Projection")
    ax.set_xlabel(cv1_axis_label, labelpad=10)
    ax.set_ylabel(cv2_axis_label, labelpad=10)
    ax.set_zlabel("Unbiased FES (kcal/mol)", labelpad=10)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_zlim(z_floor, z_top)
    ax.set_box_aspect((x_max - x_min, y_max - y_min, max(1.0, z_top - z_floor) * 0.62))
    ax.view_init(elev=28, azim=-129)
    ax.xaxis.pane.set_facecolor((0.985, 0.985, 0.985, 1.0))
    ax.yaxis.pane.set_facecolor((0.985, 0.985, 0.985, 1.0))
    ax.zaxis.pane.set_facecolor((0.99, 0.99, 0.99, 1.0))
    ax.grid(False)

    cbar_surface = fig.colorbar(surf, ax=ax, shrink=0.62, pad=0.08, fraction=0.05)
    cbar_surface.set_label("Unbiased FES (kcal/mol)")
    cbar_proj = fig.colorbar(proj, ax=ax, shrink=0.62, pad=0.02, fraction=0.05)
    cbar_proj.set_label("Bias (sum of hills)")
    cbar_time = fig.colorbar(time_scatter, ax=ax, shrink=0.62, pad=0.14, fraction=0.05)
    cbar_time.set_label("Time (ps)")

    fig.savefig(out_png, dpi=220)
    plt.close(fig)

    _cleanup_old_outputs(out_dir, episode_num, out_name)
    return out_png


def main():
    parser = argparse.ArgumentParser(description="Create a 3D unbiased FES plot with projected bias and trajectory.")
    parser.add_argument("--episode", type=int, required=True, help="Episode number, e.g. 10")
    parser.add_argument("--temperature", type=float, default=300.0)
    parser.add_argument("--bins", type=int, default=140)
    args = parser.parse_args()

    out_png = make_plot(args.episode, temperature=args.temperature, bins=args.bins)
    print(f"Saved plot: {out_png}")


if __name__ == "__main__":
    main()
