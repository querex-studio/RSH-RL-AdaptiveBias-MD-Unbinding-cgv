import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colors
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from mpl_toolkits.mplot3d.art3d import Line3DCollection
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
    _grid_bias,
    _axis_label,
    _load_bias_log,
    _load_traj,
)


def _grid_edges(coords):
    coords = np.asarray(coords, dtype=float)
    if coords.ndim != 1 or coords.size < 2:
        raise ValueError("Coordinate arrays must be 1D with at least 2 values.")
    mids = 0.5 * (coords[:-1] + coords[1:])
    left = coords[0] - 0.5 * (coords[1] - coords[0])
    right = coords[-1] + 0.5 * (coords[-1] - coords[-2])
    return np.concatenate(([left], mids, [right]))


def validate_inputs_and_coords(layers, x, y, layer_names):
    if not layers:
        raise ValueError("At least one layer is required.")
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.ndim != 1 or y.ndim != 1:
        raise ValueError("x and y must be 1D arrays.")
    if x.size < 2 or y.size < 2:
        raise ValueError("x and y must contain at least 2 points.")
    if not (np.all(np.diff(x) > 0.0) or np.all(np.diff(x) < 0.0)):
        raise ValueError("x must be strictly monotonic.")
    if not (np.all(np.diff(y) > 0.0) or np.all(np.diff(y) < 0.0)):
        raise ValueError("y must be strictly monotonic.")

    checked = []
    first_shape = np.asarray(layers[0]).shape
    if len(first_shape) != 2:
        raise ValueError("Each layer must be a 2D array.")
    ny, nx = first_shape
    if x.size != nx or y.size != ny:
        raise ValueError(
            f"Coordinate lengths must match layer shape. Got x={x.size}, y={y.size}, layer={first_shape}."
        )
    for idx, layer in enumerate(layers):
        arr = np.asarray(layer, dtype=float)
        if arr.shape != first_shape:
            raise ValueError(f"Layer {idx} shape {arr.shape} does not match first layer shape {first_shape}.")
        checked.append(arr)
    if layer_names is not None and len(layer_names) != len(layers):
        raise ValueError("layer_names must match the number of layers.")
    return checked, x, y


def compute_shared_normalization(layers, per_layer=False, robust_percentile=99.0):
    finite_sets = [np.asarray(layer)[np.isfinite(layer)] for layer in layers]
    finite_sets = [vals for vals in finite_sets if vals.size > 0]
    if not finite_sets:
        raise ValueError("No finite data found in layers.")

    if per_layer:
        norms = []
        for vals in finite_sets:
            vmin = float(np.min(vals))
            vmax = float(np.percentile(vals, robust_percentile))
            if not np.isfinite(vmax) or vmax <= vmin:
                vmax = vmin + 1e-9
            norms.append(colors.Normalize(vmin=vmin, vmax=vmax))
        return norms

    merged = np.concatenate(finite_sets)
    vmin = float(np.min(merged))
    vmax = float(np.percentile(merged, robust_percentile))
    if not np.isfinite(vmax) or vmax <= vmin:
        vmax = vmin + 1e-9
    shared = colors.Normalize(vmin=vmin, vmax=vmax)
    return [shared for _ in layers]


def draw_one_layer(ax, x, y, layer, z_level, cmap, norm, alpha=0.985, edgecolor=None):
    """
    Draw one imshow-like layer in the x-y plane at the given z level.

    Alignment is enforced by using the true coordinate edges derived from x and y,
    so every layer keeps the exact same x/y extent and orientation.
    """
    x_edges = _grid_edges(x)
    y_edges = _grid_edges(y)
    Xe, Ye = np.meshgrid(x_edges, y_edges)
    Ze = np.full_like(Xe, float(z_level), dtype=float)

    arr = np.asarray(layer, dtype=float)
    rgba = cmap(norm(arr))
    rgba[..., -1] = np.where(np.isfinite(arr), alpha, 0.0)

    surf = ax.plot_surface(
        Xe,
        Ye,
        Ze,
        rstride=1,
        cstride=1,
        facecolors=rgba,
        linewidth=0.0,
        antialiased=False,
        shade=False,
    )

    if edgecolor is not None:
        ax.plot(
            [x_edges[0], x_edges[-1], x_edges[-1], x_edges[0], x_edges[0]],
            [y_edges[0], y_edges[0], y_edges[-1], y_edges[-1], y_edges[0]],
            [z_level] * 5,
            color=edgecolor,
            linewidth=0.8,
            alpha=0.65,
        )
    return surf


def _draw_time_colored_trajectory(ax, x, y, t, z_level, cmap, norm, linewidth=0.9):
    if len(x) < 2:
        return None
    pts = np.column_stack([x, y, np.full(len(x), float(z_level), dtype=float)])
    segs = np.stack([pts[:-1], pts[1:]], axis=1)
    lc = Line3DCollection(segs, cmap=cmap, norm=norm, linewidth=linewidth, alpha=1.0)
    lc.set_array(0.5 * (np.asarray(t[:-1], dtype=float) + np.asarray(t[1:], dtype=float)))
    ax.add_collection3d(lc)
    return lc


def _episode_xy_limits(cv1, cv2, target_y, bias_y, bias_z):
    x_min = 0.0
    x_max = 10.0
    y_vals = [float(np.min(cv2)), float(np.max(cv2)), float(target_y), float(config.TARGET2_MIN), float(config.TARGET2_MAX)]
    mask = np.isfinite(bias_z) & (bias_z > 0.0)
    if np.any(mask):
        rows, _ = np.where(mask)
        y_vals.extend([float(bias_y[np.min(rows)]), float(bias_y[np.max(rows)])])
    y_min = float(np.min(y_vals))
    y_max = float(np.max(y_vals))
    y_pad = max(0.35, 0.12 * max(0.5, y_max - y_min))
    return x_min, x_max, y_min - y_pad, y_max + y_pad


def _histogram_on_common_grid(cv1, cv2, x, y, weights=None):
    x_edges = _grid_edges(x)
    y_edges = _grid_edges(y)
    hist, _, _ = np.histogram2d(cv1, cv2, bins=[x_edges, y_edges], weights=weights)
    return hist.T


def _compute_episode_layers(episode_num, temperature=300.0, bins=180):
    traj_csv, meta_json = _episode_paths(episode_num)
    if not os.path.exists(traj_csv):
        raise FileNotFoundError(traj_csv)
    if not os.path.exists(meta_json):
        raise FileNotFoundError(meta_json)

    time_ps, cv1, cv2 = _load_traj(traj_csv)
    meta = _load_bias_log(meta_json)
    bias_log = meta.get("bias_log", [])
    target_x = float(meta.get("target_center_A", config.TARGET_CENTER))
    target_y = float(meta.get("target2_center_A", config.TARGET2_CENTER))
    cv1_axis_label = _axis_label(meta, "cv1", getattr(config, "CV1_LABEL", "CV1"))
    cv2_axis_label = _axis_label(meta, "cv2", getattr(config, "CV2_LABEL", "CV2"))

    frame_bias = _frame_biases(cv1, cv2, bias_log)
    beta = 1.0 / (KB_KCAL * float(temperature))
    weights = np.exp(beta * frame_bias)

    _, bias_y_native, bias_native = _bias_surface_like_profile(bias_log)
    x_min, x_max, y_min, y_max = _episode_xy_limits(cv1, cv2, target_y, bias_y_native, bias_native)
    x = np.linspace(x_min, x_max, int(bins))
    y = np.linspace(y_min, y_max, int(bins))

    weighted_counts = _histogram_on_common_grid(cv1, cv2, x, y, weights=weights)
    visits = _histogram_on_common_grid(cv1, cv2, x, y, weights=None)
    with np.errstate(divide="ignore", invalid="ignore"):
        prob = weighted_counts / max(float(np.sum(weighted_counts)), 1.0)
        fes = -(KB_KCAL * float(temperature)) * np.log(prob)
    if np.isfinite(fes).any():
        fes -= np.nanmin(fes[np.isfinite(fes)])
        finite = fes[np.isfinite(fes)]
        fes_bg = float(np.nanpercentile(finite, 97.0) + 0.15 * max(1e-6, np.nanmax(finite) - np.nanmin(finite)))
    else:
        fes_bg = 1.0
    # Fill unsampled regions with a high-energy background so the slice remains complete.
    fes[visits <= 0.0] = fes_bg

    bias = _grid_bias(x, y, bias_log) if bias_log else np.zeros((len(y), len(x)), dtype=float)
    if np.isfinite(bias).any():
        bias_floor = float(np.nanpercentile(bias[np.isfinite(bias)], 2.0))
        bias = np.maximum(bias, bias_floor)

    # The top slice is a plain aligned plane; only the time-colored trajectory is drawn on it.
    traj_plane = np.zeros_like(fes, dtype=float)

    return {
        "time_ps": time_ps,
        "cv1": cv1,
        "cv2": cv2,
        "x": x,
        "y": y,
        "layers": [traj_plane, fes, bias],
        "layer_names": ["Trajectories", "Unbiased FES", "Bias"],
        "target_x": target_x,
        "target_y": target_y,
        "cv1_axis_label": cv1_axis_label,
        "cv2_axis_label": cv2_axis_label,
    }


def assemble_full_figure(
    layers,
    x,
    y,
    layer_names,
    trajectory_x,
    trajectory_y,
    target_x,
    target_y,
    time_ps=None,
    cv1_axis_label=None,
    cv2_axis_label=None,
    output_path="stacked_episode_slices.png",
    per_layer_norm=False,
    figure_size=(28, 20),
    z_spacing=3.8,
    view_elev=22,
    view_azim=-124,
):
    layers, x, y = validate_inputs_and_coords(layers, x, y, layer_names)
    norms = compute_shared_normalization(layers, per_layer=per_layer_norm, robust_percentile=99.0)
    shared_cmap = matplotlib.colormaps["viridis"]

    fig = plt.figure(figsize=figure_size)
    ax = fig.add_subplot(111, projection="3d")
    ax.set_proj_type("ortho")

    # Fixed order requested by user: top trajectories, middle FES, bottom bias.
    z_levels = np.array([2.0 * z_spacing, 1.0 * z_spacing, 0.0], dtype=float)
    edge_colors = ["#355070", "#4f6d7a", "#7b2d26"]

    for idx, (layer, name, z_level) in enumerate(zip(layers, layer_names, z_levels)):
        if idx != 0:
            draw_one_layer(
                ax,
                x,
                y,
                layer,
                z_level=z_level,
                cmap=shared_cmap,
                norm=norms[idx],
                alpha=0.985,
                edgecolor=edge_colors[idx % len(edge_colors)],
            )
        ax.text(
            float(x[-1]) + 0.025 * (x[-1] - x[0]),
            float(y[0]),
            float(z_level + 0.12 * z_spacing),
            name,
            fontsize=15,
            fontweight="bold",
            color="#1f1f1f",
            ha="left",
            va="bottom",
        )

    traj_times = np.asarray(time_ps if time_ps is not None else np.arange(len(trajectory_x)), dtype=float)
    traj_norm = colors.Normalize(vmin=float(np.min(traj_times)), vmax=float(np.max(traj_times)))
    top_z = float(z_levels[0])
    _draw_time_colored_trajectory(
        ax,
        trajectory_x,
        trajectory_y,
        traj_times,
        z_level=top_z + 0.08 * z_spacing,
        cmap=shared_cmap,
        norm=traj_norm,
        linewidth=0.7,
    )
    ax.scatter(
        [target_x],
        [target_y],
        [top_z + 0.14 * z_spacing],
        s=150,
        marker="*",
        color="#e63946",
        edgecolor="white",
        linewidth=1.0,
        depthshade=False,
        zorder=7,
    )

    x_span = max(1.0, float(np.max(x) - np.min(x)))
    y_span = max(1.0, float(np.max(y) - np.min(y)))
    z_span = max(1.0, float(np.max(z_levels) - np.min(z_levels) + 1.6 * z_spacing))

    ax.set_title("Episode stacked slices: trajectories, unbiased FES, and bias", pad=28, fontsize=20)
    ax.set_xlim(float(np.min(x)), float(np.max(x)))
    ax.set_ylim(float(np.min(y)), float(np.max(y)))
    ax.set_xlabel(cv1_axis_label or getattr(config, "CV1_AXIS_LABEL", f"{config.CV1_LABEL} (A)"), labelpad=18, fontsize=13)
    ax.set_ylabel(cv2_axis_label or getattr(config, "CV2_AXIS_LABEL", f"{config.CV2_LABEL} (A)"), labelpad=18, fontsize=13)
    ax.set_zlim(float(np.min(z_levels) - 0.38 * z_spacing), float(np.max(z_levels) + 0.46 * z_spacing))
    ax.set_zticks([])
    ax.set_box_aspect((x_span, y_span, 1.75 * z_span))
    ax.view_init(elev=view_elev, azim=view_azim)
    ax.grid(False)
    ax.xaxis.pane.set_facecolor((0.985, 0.985, 0.985, 1.0))
    ax.yaxis.pane.set_facecolor((0.985, 0.985, 0.985, 1.0))
    ax.zaxis.pane.set_facecolor((0.995, 0.995, 0.995, 0.0))
    ax.zaxis.line.set_linewidth(0.0)

    # One centered horizontal colorbar for the shared slice normalization.
    cax = fig.add_axes([0.30, 0.06, 0.40, 0.020])
    sm = plt.cm.ScalarMappable(norm=norms[0], cmap=shared_cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cax, orientation="horizontal")
    cbar.set_label("Shared normalized slice value", fontsize=11)
    cbar.ax.tick_params(labelsize=9)

    fig.subplots_adjust(left=0.005, right=0.995, bottom=0.12, top=0.955)
    fig.savefig(output_path, dpi=260)
    plt.close(fig)
    return output_path


def _synthetic_demo():
    x = np.linspace(0.0, 10.0, 220)
    y = np.linspace(1.2, 8.4, 180)
    X, Y = np.meshgrid(x, y)
    fes = 5.0 * np.exp(-((X - 3.2) ** 2 / 1.8 + (Y - 5.8) ** 2 / 2.0))
    fes += 3.5 * np.exp(-((X - 6.7) ** 2 / 2.2 + (Y - 3.4) ** 2 / 1.4))
    fes += 0.8 * np.sin(0.8 * X) * np.cos(0.7 * Y)
    fes -= np.nanmin(fes)
    finite = fes[np.isfinite(fes)]
    fes += 0.0
    bias = 2.5 * np.exp(-((X - 2.8) ** 2 / 1.0 + (Y - 6.2) ** 2 / 1.1))
    bias += 2.1 * np.exp(-((X - 5.8) ** 2 / 0.8 + (Y - 4.1) ** 2 / 0.9))
    if np.isfinite(bias).any():
        bias = np.maximum(bias, float(np.nanpercentile(bias, 2.0)))
    traj_plane = np.zeros_like(fes, dtype=float)

    t = np.linspace(0.0, 1.0, 140)
    tx = 1.1 + 7.3 * t + 0.35 * np.sin(8.0 * t)
    ty = 7.2 - 4.8 * t + 0.25 * np.cos(7.0 * t)
    return x, y, [traj_plane, fes, bias], ["Trajectories", "Unbiased FES", "Bias"], tx, ty


def make_episode_plot(episode_num, temperature=300.0, bins=180):
    payload = _compute_episode_layers(episode_num, temperature=temperature, bins=bins)
    out_dir = os.path.join(config.RESULTS_DIR, "analysis_runs", f"episode_{episode_num:04d}")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"episode_{episode_num:04d}_stacked_slices.png")
    return assemble_full_figure(
        layers=payload["layers"],
        x=payload["x"],
        y=payload["y"],
        layer_names=payload["layer_names"],
        trajectory_x=payload["cv1"],
        trajectory_y=payload["cv2"],
        target_x=payload["target_x"],
        target_y=payload["target_y"],
        time_ps=payload["time_ps"],
        cv1_axis_label=payload["cv1_axis_label"],
        cv2_axis_label=payload["cv2_axis_label"],
        output_path=out_path,
        per_layer_norm=False,
        figure_size=(36, 20),
        z_spacing=3.8,
        view_elev=22,
        view_azim=-124,
    )


def main():
    parser = argparse.ArgumentParser(description="Create a stacked 3D slice plot for one episode or synthetic demo.")
    parser.add_argument("--episode", type=int, default=None, help="Episode number. If omitted, a synthetic demo is generated.")
    parser.add_argument("--temperature", type=float, default=300.0)
    parser.add_argument("--bins", type=int, default=180)
    parser.add_argument("--demo-out", default="stacked_imshow_slices_demo.png")
    args = parser.parse_args()

    if args.episode is None:
        x, y, layers, names, tx, ty = _synthetic_demo()
        out_path = assemble_full_figure(
            layers=layers,
            x=x,
            y=y,
            layer_names=names,
            trajectory_x=tx,
            trajectory_y=ty,
            target_x=7.5,
            target_y=2.9,
            output_path=args.demo_out,
            per_layer_norm=False,
            figure_size=(36, 20),
            z_spacing=3.8,
            view_elev=22,
            view_azim=-124,
        )
    else:
        out_path = make_episode_plot(args.episode, temperature=args.temperature, bins=args.bins)

    print(f"Saved stacked slice figure: {out_path}")


if __name__ == "__main__":
    main()
