"""
Hybrid PPO exploration + Adaptive-CVgen-style restart selection.

This is a downstream analysis workflow. It consumes a TICA run generated from
PPO trajectories, ranks restart candidates, exports selected frames as PDBs,
and overlays two distinct transition pathways on a TICA-space F/kT map.
"""

import argparse
import csv
import glob
import json
import math
import os
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

try:
    import MDAnalysis as mda
except Exception as exc:  # pragma: no cover - handled at runtime
    mda = None
    _MDA_IMPORT_ERROR = exc
else:
    _MDA_IMPORT_ERROR = None

from analysis.pca_phosphate_pathway import (
    ROOT_DIR,
    _abs_path,
    _load_config,
    _resolve_default_traj_glob,
    _traj_label,
)

KB_KCAL = 0.00198720425864083


def _space_spec(space):
    if str(space).strip().lower() == "pca":
        return {
            "space": "pca",
            "label": "PCA",
            "candidate_file": "pca_adaptive_seed_candidates.csv",
            "scores_file": "phosphate_pathway_scores_all.csv",
            "run_subdir": "pca_hybrid_restart",
            "plot_name": "pca_hybrid_two_transition_pathways_f_over_kT.png",
            "x_source": "PC1",
            "y_source": "PC2",
            "x_internal": "TIC1",
            "y_internal": "TIC2",
            "x_label": "PC1",
            "y_label": "PC2",
            "distance_field": "pca_mode_distance",
        }
    return {
        "space": "tica",
        "label": "TICA",
        "candidate_file": "tica_adaptive_seed_candidates.csv",
        "scores_file": "phosphate_pathway_tica_scores_all.csv",
        "run_subdir": "hybrid_restart",
        "plot_name": "hybrid_two_transition_pathways_f_over_kT.png",
        "x_source": "TIC1",
        "y_source": "TIC2",
        "x_internal": "TIC1",
        "y_internal": "TIC2",
        "x_label": "TICA Dim 0",
        "y_label": "TICA Dim 1",
        "distance_field": "slow_mode_distance",
    }


def _float_or_nan(value):
    try:
        if value is None or value == "":
            return float("nan")
        return float(value)
    except Exception:
        return float("nan")


def _int_or_none(value):
    try:
        if value is None or value == "":
            return None
        return int(float(value))
    except Exception:
        return None


def _read_table(path):
    with open(path, "r", newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _write_table(path, rows, fieldnames):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def _episode_from_text(value):
    if not value:
        return None
    text = str(value)
    patterns = [
        r"(?:^|[_\-.])ep[_-]?0*(\d+)(?:[_\-.]|$)",
        r"episode[_-]?0*(\d+)",
        r"traj[_-]?0*(\d+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return int(match.group(1))
    return None


def _segment_from_text(value):
    if not value:
        return None
    match = re.search(r"(?:^|[_\-.])s[_-]?0*(\d+)(?:[_\-.]|$)", str(value), flags=re.IGNORECASE)
    if match:
        return int(match.group(1))
    return None


def _row_episode(row):
    for key in ("episode", "episode_index", "traj", "traj_path"):
        ep = _episode_from_text(row.get(key, ""))
        if ep is not None:
            return ep
    return None


def _row_segment(row):
    for key in ("segment", "segment_index", "traj", "traj_path"):
        seg = _segment_from_text(row.get(key, ""))
        if seg is not None:
            return seg
    return None


def _numeric_row(row):
    out = dict(row)
    for key in (
        "rank",
        "frame_idx",
        "time_ps",
        "cv1_A",
        "cv2_A",
        "TIC1",
        "TIC2",
        "cluster",
        "adaptive_score",
        "cv1_progress",
        "cv2_progress",
        "novelty",
        "slow_mode_distance",
        "PC1",
        "PC2",
        "pca_mode_distance",
    ):
        if key in out:
            if key in ("rank", "frame_idx", "cluster"):
                out[key] = _int_or_none(out[key])
            else:
                out[key] = _float_or_nan(out[key])
    out["episode"] = _row_episode(out)
    out["segment"] = _row_segment(out)
    return out


def _filter_by_max_episode(rows, max_episode):
    if max_episode is None or max_episode <= 0:
        return rows
    filtered = []
    for row in rows:
        ep = row.get("episode")
        if ep is None or int(ep) <= int(max_episode):
            filtered.append(row)
    return filtered


def _candidate_sort_key(row):
    score = row.get("adaptive_score", float("nan"))
    if score is None or not np.isfinite(score):
        score = float("-inf")
    return float(score)


def _coerce_space_columns(rows, spec):
    for row in rows:
        x_source = spec["x_source"]
        y_source = spec["y_source"]
        if spec["x_internal"] not in row and x_source in row:
            row[spec["x_internal"]] = row[x_source]
        if spec["y_internal"] not in row and y_source in row:
            row[spec["y_internal"]] = row[y_source]
        if spec["distance_field"] in row and "slow_mode_distance" not in row:
            row["slow_mode_distance"] = row[spec["distance_field"]]
    return rows


def _find_latest_tica_run(cfg, runs_root=None, spec=None):
    spec = spec or _space_spec("tica")
    root = runs_root or getattr(cfg, "RUNS_DIR", os.path.join(ROOT_DIR, "results_PPO", "analysis_runs"))
    root = _abs_path(root)
    matches = glob.glob(os.path.join(root, "*", "data", spec["candidate_file"]))
    if not matches:
        raise FileNotFoundError(
            f"No {spec['label']} candidate CSV found. Run the {spec['label']} analysis first or pass --tica-run explicitly."
        )
    matches.sort(key=lambda path: os.path.getmtime(path), reverse=True)
    return os.path.dirname(os.path.dirname(matches[0]))


def _resolve_tica_run(cfg, args):
    spec = _space_spec(args.space)
    if args.tica_run:
        run = _abs_path(args.tica_run)
    else:
        run = _find_latest_tica_run(cfg, args.runs_root, spec=spec)
    data_dir = os.path.join(run, "data")
    candidate_path = os.path.join(data_dir, spec["candidate_file"])
    scores_path = os.path.join(data_dir, spec["scores_file"])
    if not os.path.exists(candidate_path):
        raise FileNotFoundError(f"Missing {spec['label']} candidate CSV: {candidate_path}")
    if not os.path.exists(scores_path):
        raise FileNotFoundError(f"Missing {spec['label']} score CSV: {scores_path}")
    return run, candidate_path, scores_path, spec


def _build_traj_index(cfg, traj_glob):
    pattern = _abs_path(traj_glob) if traj_glob else _resolve_default_traj_glob(cfg)
    paths = sorted(glob.glob(pattern))
    index = {}
    for path in paths:
        abs_path = os.path.abspath(path)
        base = os.path.basename(path)
        stem = os.path.splitext(base)[0]
        for key in {abs_path, path, base, stem, _traj_label(path)}:
            index[str(key)] = abs_path
    return index


def _resolve_traj_path(row, traj_index):
    for key in ("traj_path", "traj"):
        value = row.get(key)
        if not value:
            continue
        if os.path.exists(str(value)):
            return os.path.abspath(str(value))
        if str(value) in traj_index:
            return traj_index[str(value)]
        stem = os.path.splitext(os.path.basename(str(value)))[0]
        if stem in traj_index:
            return traj_index[stem]
    return None


def _select_restarts(candidate_rows, top_n):
    rows = sorted(candidate_rows, key=_candidate_sort_key, reverse=True)
    if top_n and top_n > 0:
        rows = rows[: int(top_n)]
    out = []
    for rank, row in enumerate(rows, start=1):
        item = dict(row)
        item["hybrid_rank"] = rank
        out.append(item)
    return out


def _tica_distance(row_a, row_b):
    dx = float(row_a.get("TIC1", 0.0)) - float(row_b.get("TIC1", 0.0))
    dy = float(row_a.get("TIC2", 0.0)) - float(row_b.get("TIC2", 0.0))
    return math.sqrt(dx * dx + dy * dy)


def _select_two_pathway_endpoints(restarts):
    valid = [
        row for row in restarts
        if np.isfinite(row.get("TIC1", float("nan"))) and np.isfinite(row.get("TIC2", float("nan")))
    ]
    if not valid:
        return [], "no_valid_tica_coordinates"
    if len(valid) == 1:
        return [valid[0]], "single_candidate"

    first = valid[0]
    different_cluster = [
        row for row in valid[1:]
        if row.get("cluster") is not None and row.get("cluster") != first.get("cluster")
    ]
    pool = different_cluster if different_cluster else valid[1:]
    second = max(pool, key=lambda row: _tica_distance(first, row))
    mode = "different_cluster" if different_cluster else "max_tica_separation"
    return [first, second], mode


def _trajectory_rows_for_pathway(score_rows, endpoint):
    traj = endpoint.get("traj")
    endpoint_ep = endpoint.get("episode")
    endpoint_seg = endpoint.get("segment")
    frame = endpoint.get("frame_idx")

    if endpoint_ep is not None and endpoint_seg is not None and frame is not None:
        rows = [
            row for row in score_rows
            if row.get("episode") == endpoint_ep
            and row.get("segment") is not None
            and row.get("frame_idx") is not None
            and (
                int(row["segment"]) < int(endpoint_seg)
                or (int(row["segment"]) == int(endpoint_seg) and int(row["frame_idx"]) <= int(frame))
            )
            and np.isfinite(row.get("TIC1", float("nan")))
            and np.isfinite(row.get("TIC2", float("nan")))
        ]
        rows.sort(key=lambda row: (int(row.get("segment") or 0), int(row.get("frame_idx") or 0)))
        if len(rows) >= 2:
            return rows

    rows = [
        row for row in score_rows
        if row.get("traj") == traj
        and row.get("frame_idx") is not None
        and frame is not None
        and int(row["frame_idx"]) <= int(frame)
        and np.isfinite(row.get("TIC1", float("nan")))
        and np.isfinite(row.get("TIC2", float("nan")))
    ]
    rows.sort(key=lambda row: int(row["frame_idx"]))
    if len(rows) >= 2:
        return rows
    return [
        row for row in score_rows
        if row.get("traj") == traj
        and np.isfinite(row.get("TIC1", float("nan")))
        and np.isfinite(row.get("TIC2", float("nan")))
    ]


def _grid_fkT(score_rows, bins, weights=None):
    finite_rows = [
        (idx, row) for idx, row in enumerate(score_rows)
        if np.isfinite(row.get("TIC1", float("nan"))) and np.isfinite(row.get("TIC2", float("nan")))
    ]
    x = np.asarray([row["TIC1"] for _, row in finite_rows], dtype=float)
    y = np.asarray([row["TIC2"] for _, row in finite_rows], dtype=float)
    if len(x) < 2 or len(y) < 2:
        raise ValueError("At least two finite TIC1/TIC2 frames are required for the F/kT imshow plot.")
    hist_weights = None
    if weights is not None:
        weights = np.asarray(weights, dtype=np.float64)
        hist_weights = np.asarray([weights[idx] for idx, _ in finite_rows], dtype=np.float64)
    hist, xedges, yedges = np.histogram2d(x, y, bins=int(bins), weights=hist_weights)
    prob = hist / max(1.0, float(np.sum(hist)))
    with np.errstate(divide="ignore", invalid="ignore"):
        fkT = -np.log(prob)
    finite = np.isfinite(fkT)
    if np.any(finite):
        fkT[finite] -= np.nanmin(fkT[finite])
    fkT[~finite] = np.nan
    return fkT, xedges, yedges


def _draw_pathways(ax, pathways):
    colors = ["#d62828", "#1d4e89"]
    for idx, pathway in enumerate(pathways, start=1):
        rows = pathway["rows"]
        if not rows:
            continue
        x = [row["TIC1"] for row in rows]
        y = [row["TIC2"] for row in rows]
        color = colors[(idx - 1) % len(colors)]
        ax.plot(x, y, color=color, linewidth=2.4, alpha=0.9, label=f"Pathway {idx}")
        ax.scatter([x[0]], [y[0]], color=color, marker="o", s=42, edgecolor="white", linewidth=0.7)
        ax.scatter([x[-1]], [y[-1]], color=color, marker="*", s=140, edgecolor="white", linewidth=0.7)
        endpoint = pathway["endpoint"]
        ax.annotate(
            f"P{idx} seed {endpoint.get('hybrid_rank', '')}",
            (x[-1], y[-1]),
            xytext=(6, 6),
            textcoords="offset points",
            color=color,
            fontsize=9,
            weight="bold",
        )


def _imshow_fkT(ax, fkT, xedges, yedges, title, spec):
    image = ax.imshow(
        fkT.T,
        origin="lower",
        extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
        aspect="auto",
        cmap="viridis",
        interpolation="nearest",
    )
    ax.set_xlabel(spec["x_label"])
    ax.set_ylabel(spec["y_label"])
    ax.set_title(title)
    return image


def _plot_two_pathways(out_path, score_rows, pathways, mode, bins, spec, weights=None, title_prefix=None):
    fkT, xedges, yedges = _grid_fkT(score_rows, bins, weights=weights)
    fig, ax = plt.subplots(figsize=(9, 7))
    title = title_prefix or f"Hybrid restart pathways on {spec['label']} F/kT map"
    image = _imshow_fkT(ax, fkT, xedges, yedges, f"{title} ({mode})", spec)
    cbar = fig.colorbar(image, ax=ax)
    cbar.set_label("Free energy / kT")
    _draw_pathways(ax, pathways)
    ax.legend(loc="best", frameon=True)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def _plot_biased_unbiased_side_by_side(out_path, score_rows, pathways, mode, bins, spec, weights):
    biased, bx, by = _grid_fkT(score_rows, bins)
    unbiased, ux, uy = _grid_fkT(score_rows, bins, weights=weights)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.2), sharex=False, sharey=False)
    image0 = _imshow_fkT(axes[0], biased, bx, by, f"Sampled biased {spec['label']} F/kT", spec)
    _draw_pathways(axes[0], pathways)
    axes[0].legend(loc="best", frameon=True)
    image1 = _imshow_fkT(axes[1], unbiased, ux, uy, f"Bias-reweighted {spec['label']} F/kT", spec)
    _draw_pathways(axes[1], pathways)
    axes[1].legend(loc="best", frameon=True)
    cbar0 = fig.colorbar(image0, ax=axes[0], fraction=0.046, pad=0.04)
    cbar1 = fig.colorbar(image1, ax=axes[1], fraction=0.046, pad=0.04)
    cbar0.set_label("Free energy / kT")
    cbar1.set_label("Free energy / kT")
    fig.suptitle(f"Hybrid restart pathways: biased vs bias-reweighted ({mode})", y=1.02)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _bias_value_from_terms(cv1_A, cv2_A, terms):
    val = 0.0
    for row in terms:
        amp_kcal = float(row[2])
        center1_A = float(row[3])
        sigma_x_A = max(1e-6, float(row[5]))
        kind = str(row[1]) if len(row) > 1 else ""
        is_2d = kind in {"gaussian2d", "bias2d"} or (len(row) >= 7 and row[6] not in ("", None))
        if is_2d:
            center2_A = float(row[4])
            sigma_y_A = max(1e-6, float(row[6]))
            val += amp_kcal * np.exp(
                -((float(cv1_A) - center1_A) ** 2) / (2.0 * sigma_x_A * sigma_x_A)
                -((float(cv2_A) - center2_A) ** 2) / (2.0 * sigma_y_A * sigma_y_A)
            )
        else:
            val += amp_kcal * np.exp(-((float(cv1_A) - center1_A) ** 2) / (2.0 * sigma_x_A * sigma_x_A))
    return float(val)


def _episode_meta_path(cfg, episode):
    if episode is None:
        return None
    return os.path.join(getattr(cfg, "RESULTS_DIR", os.path.join(ROOT_DIR, "results_PPO")), "episode_meta", f"episode_{int(episode):04d}.json")


def _bias_reweighting_weights(score_rows, cfg, temperature):
    meta_cache = {}
    weights = []
    found_any = False
    beta = 1.0 / (KB_KCAL * float(temperature))
    for row in score_rows:
        ep = row.get("episode")
        meta = None
        if ep is not None:
            if ep not in meta_cache:
                path = _episode_meta_path(cfg, ep)
                if path and os.path.exists(path):
                    with open(path, "r", encoding="utf-8") as fh:
                        meta_cache[ep] = json.load(fh)
                else:
                    meta_cache[ep] = None
            meta = meta_cache.get(ep)
        bias_log = (meta or {}).get("bias_log", [])
        segment = row.get("segment")
        if segment is not None:
            terms = [term for term in bias_log if len(term) > 0 and int(term[0]) <= int(segment)]
        else:
            terms = bias_log
        if terms:
            found_any = True
        bias_kcal = _bias_value_from_terms(row.get("cv1_A", 0.0), row.get("cv2_A", 0.0), terms)
        weights.append(float(np.exp(np.clip(beta * bias_kcal, -100.0, 100.0))))
    if not found_any:
        return None
    return np.asarray(weights, dtype=np.float64)


def _topology_path(cfg, args):
    if args.topology:
        return _abs_path(args.topology)
    for key in ("psf_file", "pdb_file"):
        value = getattr(cfg, key, None)
        if value:
            path = _abs_path(value)
            if os.path.exists(path):
                return path
    raise FileNotFoundError("Could not resolve topology. Pass --topology or set psf_file/pdb_file in the config.")


def _export_restart_pdbs(restarts, cfg, args, traj_index, out_dir):
    if args.no_export_pdb:
        out = []
        for row in restarts:
            item = dict(row)
            item["resolved_traj_path"] = _resolve_traj_path(item, traj_index) or ""
            item["pdb_path"] = ""
            item["export_status"] = "skipped_no_export_pdb"
            out.append(item)
        return out
    if mda is None:
        raise RuntimeError(f"MDAnalysis import failed, so PDB export cannot run: {_MDA_IMPORT_ERROR}")

    topology = _topology_path(cfg, args)
    restart_dir = os.path.join(out_dir, "restart_candidates")
    os.makedirs(restart_dir, exist_ok=True)

    exported = []
    for row in restarts:
        traj_path = _resolve_traj_path(row, traj_index)
        row["resolved_traj_path"] = traj_path or ""
        if not traj_path:
            row["pdb_path"] = ""
            row["export_status"] = "missing_trajectory"
            exported.append(row)
            continue

        frame_idx = row.get("frame_idx")
        if frame_idx is None:
            row["pdb_path"] = ""
            row["export_status"] = "missing_frame_idx"
            exported.append(row)
            continue

        try:
            universe = mda.Universe(topology, traj_path)
            frame_idx = int(frame_idx)
            if frame_idx < 0 or frame_idx >= len(universe.trajectory):
                raise IndexError(f"frame {frame_idx} outside trajectory length {len(universe.trajectory)}")
            universe.trajectory[frame_idx]
            atoms = universe.select_atoms(args.atom_sel)
            safe_traj = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(row.get("traj", "traj")))
            out_name = f"seed_rank_{int(row['hybrid_rank']):03d}_{safe_traj}_frame_{frame_idx:06d}.pdb"
            pdb_path = os.path.join(restart_dir, out_name)
            atoms.write(pdb_path)
            row["pdb_path"] = os.path.abspath(pdb_path)
            row["export_status"] = "exported"
        except Exception as exc:
            row["pdb_path"] = ""
            row["export_status"] = f"export_failed: {exc}"
        exported.append(row)
    return exported


def _write_report(path, args, tica_run, restart_rows, pathways, pathway_mode, plot_path, spec, unbiased_plot_path=None, side_by_side_plot_path=None):
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(f"# Hybrid PPO + {spec['label']}-Space Restart Report\n\n")
        fh.write("This report summarizes downstream restart selection from PPO-generated trajectories.\n\n")
        fh.write("## Scope\n\n")
        fh.write("- PPO remains the online training and biasing model.\n")
        fh.write(f"- {spec['label']} and Adaptive-CVgen-style scoring are used after trajectory generation.\n")
        fh.write("- This is not the original Adaptive CVgen online CV-weight update loop.\n\n")
        fh.write("## Inputs\n\n")
        fh.write(f"- {spec['label']} run: `{tica_run}`\n")
        fh.write(f"- Max episode filter: `{args.max_episode if args.max_episode else 'none'}`\n")
        fh.write(f"- Restart candidates requested: `{args.top_restarts}`\n\n")
        fh.write("## Outputs\n\n")
        fh.write(f"- Restart candidate PDB directory: `restart_candidates/`\n")
        fh.write(f"- Two-pathway F/kT plot: `{os.path.relpath(plot_path, os.path.dirname(path))}`\n")
        if unbiased_plot_path:
            fh.write(f"- Bias-reweighted two-pathway F/kT plot: `{os.path.relpath(unbiased_plot_path, os.path.dirname(path))}`\n")
        if side_by_side_plot_path:
            fh.write(f"- Biased vs bias-reweighted comparison plot: `{os.path.relpath(side_by_side_plot_path, os.path.dirname(path))}`\n")
        if unbiased_plot_path or side_by_side_plot_path:
            fh.write("\nBias-reweighted plots are approximate unbiased estimates from saved bias metadata, not independent unbiased MD trajectories.\n")
        fh.write("\n")
        fh.write("## Pathway Selection\n\n")
        fh.write(f"- Two-pathway selection mode: `{pathway_mode}`\n")
        for idx, pathway in enumerate(pathways, start=1):
            endpoint = pathway["endpoint"]
            fh.write(
                f"- Pathway {idx}: seed rank `{endpoint.get('hybrid_rank')}`, "
                f"trajectory `{endpoint.get('traj')}`, frame `{endpoint.get('frame_idx')}`, "
                f"cluster `{endpoint.get('cluster')}`, score `{endpoint.get('adaptive_score')}`\n"
            )
        fh.write("\n## Restart Candidates\n\n")
        exported = sum(1 for row in restart_rows if row.get("export_status") == "exported")
        fh.write(f"- Restart rows written: `{len(restart_rows)}`\n")
        fh.write(f"- PDBs exported: `{exported}`\n")
        fh.write("\nUse exported PDBs for unbiased validation, weakly biased follow-up, or PPO restart/evaluation experiments.\n")


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Select Adaptive-CVgen-style restart candidates from a TICA or PCA run and plot two transition pathways."
    )
    parser.add_argument("--config-module", default="combined_2d")
    parser.add_argument("--space", choices=["tica", "pca"], default="tica", help="Analysis space to consume for restart selection.")
    parser.add_argument("--tica-run", default=None, help="Analysis run directory. Defaults to latest run with candidates for --space.")
    parser.add_argument("--runs-root", default=None, help="Root used to search for the latest analysis run when --tica-run is omitted.")
    parser.add_argument("--traj-glob", default=None, help="DCD glob used to resolve trajectory labels for PDB export.")
    parser.add_argument("--run", default=None, help="Hybrid output directory. Defaults to a space-specific subdirectory under the analysis run.")
    parser.add_argument("--top-restarts", type=int, default=10, help="Number of restart PDB candidates to export.")
    parser.add_argument("--max-episode", type=int, default=10, help="Use only candidates/scores up to this PPO episode; <=0 disables.")
    parser.add_argument("--bins", type=int, default=90, help="Analysis-space F/kT imshow bins.")
    parser.add_argument("--temperature", type=float, default=300.0, help="Temperature for bias reweighting in K.")
    parser.add_argument("--topology", default=None, help="Topology path for MDAnalysis PDB export. Defaults to config psf_file/pdb_file.")
    parser.add_argument("--atom-sel", default="all", help="MDAnalysis atom selection to write to restart PDBs.")
    parser.add_argument("--no-export-pdb", action="store_true", help="Create CSV/plot/report without writing restart PDBs.")
    return parser.parse_args()


def main():
    args = _parse_args()
    cfg = _load_config(args.config_module)
    tica_run, candidate_path, scores_path, spec = _resolve_tica_run(cfg, args)
    out_dir = _abs_path(args.run) if args.run else os.path.join(tica_run, spec["run_subdir"])
    data_dir = os.path.join(out_dir, "data")
    fig_dir = os.path.join(out_dir, "figs", "analysis")
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)

    candidates = [_numeric_row(row) for row in _read_table(candidate_path)]
    scores = [_numeric_row(row) for row in _read_table(scores_path)]
    candidates = _coerce_space_columns(candidates, spec)
    scores = _coerce_space_columns(scores, spec)
    candidates = _filter_by_max_episode(candidates, args.max_episode)
    scores = _filter_by_max_episode(scores, args.max_episode)
    if not candidates:
        raise RuntimeError("No TICA candidate rows remain after filtering. Increase --max-episode or rerun TICA.")
    if not scores:
        raise RuntimeError("No TICA score rows remain after filtering. Increase --max-episode or rerun TICA.")

    restarts = _select_restarts(candidates, args.top_restarts)
    traj_index = _build_traj_index(cfg, args.traj_glob)
    restarts = _export_restart_pdbs(restarts, cfg, args, traj_index, out_dir)

    endpoints, pathway_mode = _select_two_pathway_endpoints(restarts)
    pathways = []
    for endpoint in endpoints:
        pathways.append({
            "endpoint": endpoint,
            "rows": _trajectory_rows_for_pathway(scores, endpoint),
        })

    plot_path = os.path.join(fig_dir, spec["plot_name"])
    _plot_two_pathways(plot_path, scores, pathways, pathway_mode, args.bins, spec, title_prefix=f"Sampled biased {spec['label']} F/kT")
    unbiased_plot_path = None
    side_by_side_plot_path = None
    weights = _bias_reweighting_weights(scores, cfg, args.temperature)
    if weights is not None:
        unbiased_plot_path = os.path.join(fig_dir, spec["plot_name"].replace("_f_over_kT.png", "_unbiased_f_over_kT.png"))
        side_by_side_plot_path = os.path.join(fig_dir, spec["plot_name"].replace("_f_over_kT.png", "_biased_unbiased_side_by_side.png"))
        _plot_two_pathways(
            unbiased_plot_path,
            scores,
            pathways,
            pathway_mode,
            args.bins,
            spec,
            weights=weights,
            title_prefix=f"Bias-reweighted {spec['label']} F/kT",
        )
        _plot_biased_unbiased_side_by_side(side_by_side_plot_path, scores, pathways, pathway_mode, args.bins, spec, weights)

    candidate_fields = [
        "hybrid_rank", "rank", "episode", "segment", "traj", "traj_path", "resolved_traj_path", "frame_idx", "time_ps",
        "cv1_A", "cv2_A", "TIC1", "TIC2", "PC1", "PC2", "cluster", "adaptive_score", "cv1_progress", "cv2_progress",
        "novelty", "slow_mode_distance", "pca_mode_distance", "pdb_path", "export_status",
    ]
    _write_table(os.path.join(data_dir, "hybrid_restart_candidates.csv"), restarts, candidate_fields)

    pathway_rows = []
    for idx, pathway in enumerate(pathways, start=1):
        endpoint = pathway["endpoint"]
        for step_idx, row in enumerate(pathway["rows"], start=1):
            pathway_rows.append({
                "pathway_id": idx,
                "step": step_idx,
                "endpoint_hybrid_rank": endpoint.get("hybrid_rank"),
                "endpoint_cluster": endpoint.get("cluster"),
                "episode": row.get("episode"),
                "segment": row.get("segment"),
                "traj": row.get("traj"),
                "frame_idx": row.get("frame_idx"),
                "time_ps": row.get("time_ps"),
                "cv1_A": row.get("cv1_A"),
                "cv2_A": row.get("cv2_A"),
                "TIC1": row.get("TIC1"),
                "TIC2": row.get("TIC2"),
            })
    _write_table(
        os.path.join(data_dir, "hybrid_transition_pathways.csv"),
        pathway_rows,
        ["pathway_id", "step", "endpoint_hybrid_rank", "endpoint_cluster", "episode", "segment", "traj", "frame_idx", "time_ps", "cv1_A", "cv2_A", "TIC1", "TIC2"],
    )

    summary = {
        "tica_run": os.path.abspath(tica_run),
        "space": spec["space"],
        "hybrid_run": os.path.abspath(out_dir),
        "max_episode": args.max_episode,
        "top_restarts": args.top_restarts,
        "n_candidates_written": len(restarts),
        "n_scores_used": len(scores),
        "pathway_mode": pathway_mode,
        "plot_path": os.path.abspath(plot_path),
        "unbiased_plot_path": None if unbiased_plot_path is None else os.path.abspath(unbiased_plot_path),
        "side_by_side_plot_path": None if side_by_side_plot_path is None else os.path.abspath(side_by_side_plot_path),
    }
    with open(os.path.join(data_dir, "hybrid_restart_summary.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    report_path = os.path.join(data_dir, "hybrid_restart_report.md")
    _write_report(report_path, args, os.path.abspath(tica_run), restarts, pathways, pathway_mode, plot_path, spec, unbiased_plot_path, side_by_side_plot_path)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
