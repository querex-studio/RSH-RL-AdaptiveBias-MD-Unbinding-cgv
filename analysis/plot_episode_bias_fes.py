import argparse
import csv
import heapq
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
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
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.ndim != 2 or arr.shape[1] < 3:
        raise ValueError(f"Unexpected trajectory CSV format: {csv_path}")
    return arr[:, 0], arr[:, 1], arr[:, 2]


def _load_bias_log(meta_path):
    with open(meta_path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _frames_per_action():
    return max(1, int(config.propagation_step // max(1, int(config.dcdfreq_mfpt))))


def _is_2d_term(row) -> bool:
    kind = str(row[1]) if len(row) > 1 else ""
    if kind in {"gaussian2d", "bias2d"}:
        return True
    return len(row) >= 7 and row[6] is not None


def _is_1d_term(row) -> bool:
    kind = str(row[1]) if len(row) > 1 else ""
    if kind in {"gaussian1d", "bias1d"}:
        return True
    return not _is_2d_term(row)


def _bias_value_from_terms(cv1_A, cv2_A, terms):
    val = 0.0
    for row in terms:
        amp_kcal = float(row[2])
        center1_A = float(row[3])
        sigma_x_A = max(1e-6, float(row[5]))
        if _is_2d_term(row):
            center2_A = float(row[4])
            sigma_y_A = max(1e-6, float(row[6]))
            val += amp_kcal * np.exp(
                -((float(cv1_A) - center1_A) ** 2) / (2.0 * sigma_x_A * sigma_x_A)
                -((float(cv2_A) - center2_A) ** 2) / (2.0 * sigma_y_A * sigma_y_A)
            )
        else:
            val += amp_kcal * np.exp(
                -((float(cv1_A) - center1_A) ** 2) / (2.0 * sigma_x_A * sigma_x_A)
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
    for row in bias_log:
        amp_kcal = float(row[2])
        center1_A = float(row[3])
        sigma_x_A = max(1e-6, float(row[5]))
        if _is_2d_term(row):
            center2_A = float(row[4])
            sigma_y_A = max(1e-6, float(row[6]))
            Z += amp_kcal * np.exp(
                -((X - center1_A) ** 2) / (2.0 * sigma_x_A * sigma_x_A)
                -((Y - center2_A) ** 2) / (2.0 * sigma_y_A * sigma_y_A)
            )
        else:
            Z += amp_kcal * np.exp(
                -((X - center1_A) ** 2) / (2.0 * sigma_x_A * sigma_x_A)
            )
    return Z


def _bias_surface_like_profile(bias_log):
    bins = int(getattr(config, "BIAS_PROFILE_BINS", 250))
    cv2_refs = [
        float(config.TARGET2_MIN),
        float(config.TARGET2_MAX),
        float(config.FINAL_TARGET2),
        float(config.CURRENT_DISTANCE2),
    ]
    y_min = min(cv2_refs) - 1.0
    y_max = max(cv2_refs) + 1.0
    if not bias_log:
        x = np.linspace(0.0, 10.0, bins)
        y = np.linspace(y_min, y_max, bins)
        return x, y, np.zeros((len(y), len(x)), dtype=np.float64)

    centers1 = np.array([float(row[3]) for row in bias_log], dtype=np.float64)
    widths1 = np.array([max(1e-6, float(row[5])) for row in bias_log], dtype=np.float64)
    pad = float(getattr(config, "BIAS_PROFILE_PAD_SIGMA", 3.0))
    lo1 = float(np.min(centers1 - pad * widths1))
    hi1 = float(np.max(centers1 + pad * widths1))

    if any(_is_2d_term(row) for row in bias_log):
        centers2 = np.array([float(row[4]) for row in bias_log if _is_2d_term(row)], dtype=np.float64)
        widths2 = np.array([max(1e-6, float(row[6])) for row in bias_log if _is_2d_term(row)], dtype=np.float64)
        lo2 = float(np.min(centers2 - pad * widths2))
        hi2 = float(np.max(centers2 + pad * widths2))
        y = np.linspace(min(lo2, y_min), max(hi2, y_max), bins)
    else:
        y = np.linspace(y_min, y_max, bins)

    x = np.linspace(min(lo1, float(config.CURRENT_DISTANCE) - 1.0), max(hi1, float(config.FINAL_TARGET) + 1.0), bins)
    return x, y, _grid_bias(x, y, bias_log)


def _grid_edges(centers):
    centers = np.asarray(centers, dtype=np.float64)
    if centers.ndim != 1 or centers.size == 0:
        raise ValueError("Grid centers must be a non-empty 1D array.")
    if centers.size == 1:
        return np.array([centers[0] - 0.5, centers[0] + 0.5], dtype=np.float64)

    mids = 0.5 * (centers[:-1] + centers[1:])
    left = centers[0] - 0.5 * (centers[1] - centers[0])
    right = centers[-1] + 0.5 * (centers[-1] - centers[-2])
    return np.concatenate([[left], mids, [right]])


def _plot_limits(cv1, cv2, target_min, target_max, target2_min, target2_max):
    x_all = np.array([np.min(cv1), np.max(cv1), target_min, target_max], dtype=np.float64)
    y_all = np.array([np.min(cv2), np.max(cv2), target2_min, target2_max], dtype=np.float64)
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


def _reweighted_fes(cv1, cv2, frame_bias_kcal, temperature, bins, x_limits, y_limits):
    beta = 1.0 / (KB_KCAL * float(temperature))
    weights = np.exp(beta * frame_bias_kcal)

    x_edges = np.linspace(float(x_limits[0]), float(x_limits[1]), int(bins) + 1)
    y_edges = np.linspace(float(y_limits[0]), float(y_limits[1]), int(bins) + 1)
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


def _nearest_valid_index(surface, x_grid, y_grid, x_val, y_val):
    finite = np.isfinite(surface)
    if not np.any(finite):
        return None
    ix = int(np.argmin(np.abs(x_grid - float(x_val))))
    iy = int(np.argmin(np.abs(y_grid - float(y_val))))
    if finite[iy, ix]:
        return iy, ix

    best = None
    best_dist = None
    for yy, xx in np.argwhere(finite):
        dist = (float(x_grid[xx]) - float(x_val)) ** 2 + (float(y_grid[yy]) - float(y_val)) ** 2
        if best_dist is None or dist < best_dist:
            best = (int(yy), int(xx))
            best_dist = float(dist)
    return best


def _neighbor_indices(iy, ix, shape):
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dx == 0 and dy == 0:
                continue
            ny = iy + dy
            nx = ix + dx
            if 0 <= ny < shape[0] and 0 <= nx < shape[1]:
                yield ny, nx


def _minimax_path(surface, start_idx, target_idx):
    finite = np.isfinite(surface)
    if start_idx is None or target_idx is None:
        return []
    if not finite[start_idx] or not finite[target_idx]:
        return []

    costs = {start_idx: float(surface[start_idx])}
    parents = {}
    heap = [(float(surface[start_idx]), start_idx)]
    visited = set()

    while heap:
        cost, node = heapq.heappop(heap)
        if node in visited:
            continue
        visited.add(node)
        if node == target_idx:
            break
        for nxt in _neighbor_indices(node[0], node[1], surface.shape):
            if not finite[nxt]:
                continue
            new_cost = max(float(cost), float(surface[nxt]))
            if new_cost < costs.get(nxt, float("inf")):
                costs[nxt] = new_cost
                parents[nxt] = node
                heapq.heappush(heap, (new_cost, nxt))

    if target_idx not in costs:
        return []

    path = [target_idx]
    while path[-1] != start_idx:
        parent = parents.get(path[-1])
        if parent is None:
            return []
        path.append(parent)
    path.reverse()
    return path


def _trajectory_path_progress(cv1, cv2, path_points):
    traj = np.column_stack([cv1, cv2])
    progress = np.zeros(len(traj), dtype=int)
    min_dist = np.zeros(len(traj), dtype=np.float64)
    for idx, point in enumerate(traj):
        deltas = path_points - point[None, :]
        dist2 = np.sum(deltas * deltas, axis=1)
        best = int(np.argmin(dist2))
        progress[idx] = best
        min_dist[idx] = float(np.sqrt(dist2[best]))
    return progress, min_dist


def _extract_crossings(progress, barrier_idx):
    crossings = []
    in_event = False
    event_start = 0
    side_prev = np.sign(progress[0] - barrier_idx)

    for idx in range(1, len(progress)):
        side_now = np.sign(progress[idx] - barrier_idx)
        if not in_event and side_prev < 0 and side_now >= 0:
            in_event = True
            event_start = idx - 1
        if in_event and side_now > 0:
            window = np.arange(event_start, idx + 1)
            pick = int(window[np.argmin(np.abs(progress[window] - barrier_idx))])
            crossings.append((int(event_start), int(idx), pick))
            in_event = False
        if side_now != 0:
            side_prev = side_now
    return crossings


def _transition_state_analysis(cv1, cv2, time_ps, x_grid, y_grid, fes, target_center, target2_center):
    start_idx = _nearest_valid_index(fes, x_grid, y_grid, cv1[0], cv2[0])
    target_idx = _nearest_valid_index(fes, x_grid, y_grid, target_center, target2_center)
    path = _minimax_path(fes, start_idx, target_idx)
    if not path:
        return {
            "available": False,
            "reason": "No finite minimum-barrier path between start and target.",
            "crossings": [],
        }

    path_points = np.array([[float(x_grid[ix]), float(y_grid[iy])] for iy, ix in path], dtype=np.float64)
    path_energies = np.array([float(fes[iy, ix]) for iy, ix in path], dtype=np.float64)
    barrier_rel_idx = int(np.nanargmax(path_energies))
    ts_iy, ts_ix = path[barrier_rel_idx]
    ts_x = float(x_grid[ts_ix])
    ts_y = float(y_grid[ts_iy])
    ts_energy = float(fes[ts_iy, ts_ix])

    progress, path_dist = _trajectory_path_progress(cv1, cv2, path_points)
    crossing_windows = _extract_crossings(progress, barrier_rel_idx)

    x_spacing = float(np.median(np.diff(x_grid))) if len(x_grid) > 1 else 0.25
    y_spacing = float(np.median(np.diff(y_grid))) if len(y_grid) > 1 else 0.25
    ts_radius = 2.0 * max(abs(x_spacing), abs(y_spacing), 0.25)
    low_energy_margin = 1.0
    ts_energy_tolerance = 0.75

    crossings = []
    for event_start, event_end, frame_idx in crossing_windows:
        grid_idx = _nearest_valid_index(fes, x_grid, y_grid, cv1[frame_idx], cv2[frame_idx])
        local_energy = None if grid_idx is None else float(fes[grid_idx])
        dist_to_ts = float(np.hypot(cv1[frame_idx] - ts_x, cv2[frame_idx] - ts_y))
        near_ts = dist_to_ts <= ts_radius
        energy_near_ts = local_energy is not None and local_energy >= (ts_energy - ts_energy_tolerance)
        lower_energy = local_energy is not None and local_energy <= (ts_energy - low_energy_margin)
        crossings.append(
            {
                "frame_idx": int(frame_idx),
                "time_ps": float(time_ps[frame_idx]),
                "cv1_A": float(cv1[frame_idx]),
                "cv2_A": float(cv2[frame_idx]),
                "event_start_frame": int(event_start),
                "event_end_frame": int(event_end),
                "progress_index": int(progress[frame_idx]),
                "path_distance_A": float(path_dist[frame_idx]),
                "local_fes_kcal": None if local_energy is None else float(local_energy),
                "distance_to_ts_A": dist_to_ts,
                "near_ts_region": bool(near_ts and energy_near_ts),
                "lower_energy_crossing": bool(lower_energy),
            }
        )

    near_ts_count = sum(int(item["near_ts_region"]) for item in crossings)
    lower_energy_count = sum(int(item["lower_energy_crossing"]) for item in crossings)
    crossing_count = len(crossings)

    return {
        "available": True,
        "ts_point_A": {"cv1_A": ts_x, "cv2_A": ts_y},
        "ts_fes_kcal": ts_energy,
        "ts_radius_A": ts_radius,
        "ts_energy_tolerance_kcal": ts_energy_tolerance,
        "low_energy_margin_kcal": low_energy_margin,
        "mfep_path_points_A": path_points.tolist(),
        "crossing_count": crossing_count,
        "near_ts_crossing_count": near_ts_count,
        "near_ts_fraction": 0.0 if crossing_count == 0 else float(near_ts_count / crossing_count),
        "lower_energy_crossing_count": lower_energy_count,
        "lower_energy_fraction": 0.0 if crossing_count == 0 else float(lower_energy_count / crossing_count),
        "crossings": crossings,
    }


def _write_transition_outputs(out_dir, episode_num, analysis):
    summary_json = os.path.join(out_dir, f"episode_{episode_num:04d}_transition_state_summary.json")
    with open(summary_json, "w", encoding="utf-8") as fh:
        json.dump(analysis, fh, indent=2)

    csv_path = os.path.join(out_dir, f"episode_{episode_num:04d}_transition_crossings.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow([
            "frame_idx",
            "time_ps",
            "cv1_A",
            "cv2_A",
            "event_start_frame",
            "event_end_frame",
            "progress_index",
            "path_distance_A",
            "local_fes_kcal",
            "distance_to_ts_A",
            "near_ts_region",
            "lower_energy_crossing",
        ])
        for row in analysis.get("crossings", []):
            writer.writerow([
                row["frame_idx"],
                row["time_ps"],
                row["cv1_A"],
                row["cv2_A"],
                row["event_start_frame"],
                row["event_end_frame"],
                row["progress_index"],
                row["path_distance_A"],
                row["local_fes_kcal"],
                row["distance_to_ts_A"],
                int(row["near_ts_region"]),
                int(row["lower_energy_crossing"]),
            ])


def _axis_label(meta, key, fallback_label):
    stored = meta.get(f"{key}_axis_label")
    if stored:
        return str(stored)
    if key == "cv2":
        target2_center = float(meta.get("target2_center_A", np.nan))
        target2_zone = meta.get("target2_zone")
        old_pair_zone = (
            np.isfinite(target2_center)
            and abs(target2_center - 4.0) < 1e-6
            and isinstance(target2_zone, list)
            and len(target2_zone) == 2
            and abs(float(target2_zone[0]) - 3.65) < 1e-6
            and abs(float(target2_zone[1]) - 4.35) < 1e-6
        )
        if old_pair_zone:
            return "CV2 auxiliary distance (atom 487-3789) (A)"
    return f"{fallback_label} (A)"


def _add_common_overlays(
    ax,
    cv1,
    cv2,
    target_center,
    target2_center,
    target_min,
    target_max,
    target2_min,
    target2_max,
    curriculum_center=None,
    curriculum_zone=None,
):
    ax.plot(cv1, cv2, color="#f39c12", linewidth=1.0, alpha=0.85, zorder=4, label="trajectory")
    ax.scatter(cv1, cv2, s=10, color="#f39c12", alpha=0.80, edgecolors="none", zorder=5)
    ax.scatter([cv1[0]], [cv2[0]], s=90, marker="^", color="#1f77b4", edgecolor="white", linewidth=0.8, zorder=6, label="start")
    ax.scatter([target_center], [target2_center], s=150, marker="*", color="#2ca02c", edgecolor="white", linewidth=0.9, zorder=6, label="final target")
    ax.add_patch(
        plt.Rectangle(
            (target_min, target2_min),
            target_max - target_min,
            target2_max - target2_min,
            fill=False,
            edgecolor="#2ca02c",
            linewidth=1.1,
            linestyle="--",
            alpha=0.9,
            zorder=6,
        )
    )
    if curriculum_center is not None and curriculum_zone is not None:
        cz_min, cz_max = float(curriculum_zone[0]), float(curriculum_zone[1])
        if abs(float(curriculum_center) - float(target_center)) > 1e-6:
            ax.axvspan(cz_min, cz_max, color="#7f8c8d", alpha=0.10, zorder=2, label="curriculum CV1 stage")
            ax.axvline(float(curriculum_center), color="#7f8c8d", linestyle=":", linewidth=1.0, zorder=3)


def make_plot(episode_num: int, temperature: float = 300.0, bins: int = 120):
    traj_csv, meta_json = _episode_paths(episode_num)
    if not os.path.exists(traj_csv):
        raise FileNotFoundError(traj_csv)
    if not os.path.exists(meta_json):
        raise FileNotFoundError(meta_json)

    time_ps, cv1, cv2 = _load_traj(traj_csv)
    meta = _load_bias_log(meta_json)
    bias_log = meta.get("bias_log", [])

    target_zone = meta.get("target_zone") or [config.TARGET_MIN, config.TARGET_MAX]
    target_min = float(target_zone[0])
    target_max = float(target_zone[1])
    target_center = float(meta.get("target_center_A", config.TARGET_CENTER))
    curriculum_zone = meta.get("curriculum_target_zone")
    curriculum_center = meta.get("curriculum_target_center_A")
    if curriculum_zone is not None:
        curriculum_zone = [float(curriculum_zone[0]), float(curriculum_zone[1])]
    if curriculum_center is not None:
        curriculum_center = float(curriculum_center)

    target2_zone = meta.get("target2_zone", [config.TARGET2_MIN, config.TARGET2_MAX])
    target2_min = float(target2_zone[0])
    target2_max = float(target2_zone[1])
    target2_center = float(meta.get("target2_center_A", config.TARGET2_CENTER))
    cv1_axis_label = _axis_label(meta, "cv1", getattr(config, "CV1_LABEL", "CV1"))
    cv2_axis_label = _axis_label(meta, "cv2", getattr(config, "CV2_LABEL", "CV2"))

    x_min, x_max, y_min, y_max = _plot_limits(cv1, cv2, target_min, target_max, target2_min, target2_max)
    frame_bias = _frame_biases(cv1, cv2, bias_log)
    x_grid, y_grid, fes, _ = _reweighted_fes(
        cv1,
        cv2,
        frame_bias,
        temperature,
        bins,
        x_limits=(x_min, x_max),
        y_limits=(y_min, y_max),
    )

    bias_surface = _grid_bias(x_grid, y_grid, bias_log) if bias_log else np.zeros((len(y_grid), len(x_grid)), dtype=np.float64)
    effective_surface = np.array(fes, copy=True)
    finite_fes = np.isfinite(effective_surface)
    effective_surface[finite_fes] = effective_surface[finite_fes] + bias_surface[finite_fes]

    transition_analysis = _transition_state_analysis(
        cv1,
        cv2,
        time_ps,
        x_grid,
        y_grid,
        fes,
        target_center,
        target2_center,
    )

    out_dir = os.path.join(config.RESULTS_DIR, "analysis_runs", f"episode_{episode_num:04d}")
    os.makedirs(out_dir, exist_ok=True)
    out_png = os.path.join(out_dir, f"episode_{episode_num:04d}_xy_diagnostics.png")
    _write_transition_outputs(out_dir, episode_num, transition_analysis)

    combined_energy = np.concatenate([
        fes[np.isfinite(fes)],
        effective_surface[np.isfinite(effective_surface)],
    ]) if np.isfinite(fes).any() or np.isfinite(effective_surface).any() else np.array([0.0, 1.0])
    vmin = float(np.nanmin(combined_energy))
    vmax = float(np.nanpercentile(combined_energy, 97.0))
    if not np.isfinite(vmax) or vmax <= vmin:
        vmax = vmin + 1.0

    bias_pos = bias_surface[np.isfinite(bias_surface)]
    bias_vmax = float(np.nanpercentile(bias_pos, 99.0)) if bias_pos.size else 1.0
    if bias_vmax <= 0.0:
        bias_vmax = 1.0

    x_edges = _grid_edges(x_grid)
    y_edges = _grid_edges(y_grid)

    fig, axes = plt.subplots(1, 4, figsize=(20.5, 5.4), constrained_layout=True)

    panel_titles = [
        f"Episode {episode_num}: Reweighted F(CV1,CV2)",
        f"Episode {episode_num}: Effective F + Vbias",
        f"Episode {episode_num}: Vbias(CV1,CV2)",
        f"Episode {episode_num}: TS crossing diagnostics",
    ]

    mesh0 = axes[0].pcolormesh(x_edges, y_edges, np.ma.masked_invalid(fes), shading="auto", cmap="viridis", vmin=vmin, vmax=vmax)
    mesh1 = axes[1].pcolormesh(x_edges, y_edges, np.ma.masked_invalid(effective_surface), shading="auto", cmap="viridis", vmin=vmin, vmax=vmax)
    mesh2 = axes[2].pcolormesh(x_edges, y_edges, np.ma.masked_invalid(bias_surface), shading="auto", cmap="viridis", vmin=0.0, vmax=bias_vmax)
    mesh3 = axes[3].pcolormesh(x_edges, y_edges, np.ma.masked_invalid(fes), shading="auto", cmap="viridis", vmin=vmin, vmax=vmax)

    for ax, title in zip(axes, panel_titles):
        ax.set_title(title)
        ax.set_xlabel(cv1_axis_label)
        ax.set_ylabel(cv2_axis_label)
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)

    overlay_args = (
        cv1,
        cv2,
        target_center,
        target2_center,
        target_min,
        target_max,
        target2_min,
        target2_max,
        curriculum_center,
        curriculum_zone,
    )
    _add_common_overlays(axes[0], *overlay_args)
    _add_common_overlays(axes[1], *overlay_args)
    _add_common_overlays(axes[3], *overlay_args)

    centers_x = [float(row[3]) for row in bias_log]
    centers_y = [float(row[4]) if _is_2d_term(row) else float(row[4] if row[4] is not None else target2_center) for row in bias_log]
    if centers_x:
        axes[2].scatter(centers_x, centers_y, marker="x", s=42, color="red", linewidths=1.2, zorder=6)
    _add_common_overlays(
        axes[2],
        cv1[:: max(1, len(cv1) // 150)],
        cv2[:: max(1, len(cv2) // 150)],
        target_center,
        target2_center,
        target_min,
        target_max,
        target2_min,
        target2_max,
        curriculum_center,
        curriculum_zone,
    )
    axes[0].legend(loc="best", fontsize=8, frameon=True)

    if transition_analysis.get("available"):
        path_points = np.asarray(transition_analysis["mfep_path_points_A"], dtype=np.float64)
        axes[3].plot(path_points[:, 0], path_points[:, 1], color="white", linewidth=1.4, linestyle="--", zorder=6)
        ts_point = transition_analysis["ts_point_A"]
        axes[3].scatter([ts_point["cv1_A"]], [ts_point["cv2_A"]], s=90, marker="D", color="black", edgecolor="white", linewidth=0.9, zorder=7)

        near_pts = np.array([[row["cv1_A"], row["cv2_A"]] for row in transition_analysis["crossings"] if row["near_ts_region"]], dtype=np.float64)
        low_pts = np.array([[row["cv1_A"], row["cv2_A"]] for row in transition_analysis["crossings"] if row["lower_energy_crossing"]], dtype=np.float64)
        other_pts = np.array([[row["cv1_A"], row["cv2_A"]] for row in transition_analysis["crossings"] if not row["near_ts_region"] and not row["lower_energy_crossing"]], dtype=np.float64)

        if near_pts.size:
            axes[3].scatter(near_pts[:, 0], near_pts[:, 1], s=48, color="#2ca02c", edgecolor="white", linewidth=0.6, zorder=8)
        if low_pts.size:
            axes[3].scatter(low_pts[:, 0], low_pts[:, 1], s=48, color="#d62728", edgecolor="white", linewidth=0.6, zorder=8)
        if other_pts.size:
            axes[3].scatter(other_pts[:, 0], other_pts[:, 1], s=42, color="#ffbf00", edgecolor="white", linewidth=0.6, zorder=8)

        summary_text = (
            f"crossings={transition_analysis['crossing_count']}\n"
            f"near TS={transition_analysis['near_ts_crossing_count']} "
            f"({100.0 * transition_analysis['near_ts_fraction']:.1f}%)\n"
            f"lower-E={transition_analysis['lower_energy_crossing_count']} "
            f"({100.0 * transition_analysis['lower_energy_fraction']:.1f}%)"
        )
        axes[3].text(
            0.02,
            0.98,
            summary_text,
            transform=axes[3].transAxes,
            va="top",
            ha="left",
            fontsize=10,
            bbox={"facecolor": "white", "alpha": 0.82, "edgecolor": "none", "pad": 4.0},
            zorder=9,
        )

    cbar0 = fig.colorbar(mesh0, ax=axes[0], fraction=0.046, pad=0.04)
    cbar0.set_label("kcal/mol")
    cbar1 = fig.colorbar(mesh1, ax=axes[1], fraction=0.046, pad=0.04)
    cbar1.set_label("kcal/mol")
    cbar2 = fig.colorbar(mesh2, ax=axes[2], fraction=0.046, pad=0.04)
    cbar2.set_label("kcal/mol")
    cbar3 = fig.colorbar(mesh3, ax=axes[3], fraction=0.046, pad=0.04)
    cbar3.set_label("kcal/mol")

    fig.savefig(out_png, dpi=220)
    plt.close(fig)
    return out_png


def main():
    parser = argparse.ArgumentParser(description="Create four-panel episode diagnostics in CV space.")
    parser.add_argument("--episode", type=int, required=True, help="Episode number, e.g. 4")
    parser.add_argument("--temperature", type=float, default=300.0)
    parser.add_argument("--bins", type=int, default=120)
    args = parser.parse_args()

    out_png = make_plot(args.episode, temperature=args.temperature, bins=args.bins)
    print(f"Saved plot: {out_png}")


if __name__ == "__main__":
    main()
