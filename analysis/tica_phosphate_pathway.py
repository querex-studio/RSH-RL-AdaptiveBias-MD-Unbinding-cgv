import argparse
import csv
import glob
import json
import os
import sys
from collections import Counter

import matplotlib
matplotlib.use("Agg")
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
import matplotlib.pyplot as plt
import numpy as np

try:
    import MDAnalysis as mda
    from MDAnalysis.lib.distances import distance_array
except Exception as exc:
    raise ImportError("MDAnalysis is required for phosphate-pathway TICA.") from exc

try:
    from scipy.linalg import eigh
except Exception as exc:
    raise ImportError("scipy is required for phosphate-pathway TICA.") from exc

try:
    from sklearn.cluster import MiniBatchKMeans
except Exception as exc:
    raise ImportError("scikit-learn is required for phosphate-pathway TICA clustering.") from exc

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from analysis import run_utils
from analysis.pca_phosphate_pathway import (
    _abs_path,
    _build_residue_groups,
    _collect_candidate_residues,
    _corrcoef_safe,
    _distance_A,
    _dt_ps_from_cfg,
    _grid_edges,
    _load_config,
    _pairwise_indices,
    _resolve_default_traj_glob,
    _residue_label,
    _select_residues_by_cutoff,
    _select_trajs,
    _set_reference_alignment,
    _snapshot_cfg,
    _traj_label,
)


def _feature_mode(mode):
    value = str(mode).strip().lower().replace("_", "-")
    aliases = {
        "pair": "residue-pair",
        "residue-pairs": "residue-pair",
        "phosphate": "phosphate-distance",
        "phosphate-distances": "phosphate-distance",
        "cv": "cv-only",
    }
    return aliases.get(value, value)


def _feature_names(residue_keys, pair_idx, feature_set):
    mode = _feature_mode(feature_set)
    names = []
    if mode in {"residue-pair", "combined", "combined-with-cv"}:
        for i, j in zip(pair_idx[0], pair_idx[1]):
            names.append(f"pair:{_residue_label(residue_keys[int(i)])}--{_residue_label(residue_keys[int(j)])}")
    if mode in {"phosphate-distance", "combined", "combined-with-cv"}:
        for key in residue_keys:
            names.append(f"phosphate_distance:{_residue_label(key)}")
    if mode in {"cv-only", "combined-with-cv"}:
        names.extend(["cv1_distance", "cv2_distance"])
    if not names:
        raise ValueError(f"Unsupported feature set: {feature_set}")
    return names


def _points_from_groups(groups, repr_mode):
    pts = []
    for ag in groups:
        if repr_mode == "ca":
            pts.append(np.asarray(ag.positions[0], dtype=np.float64))
        else:
            pts.append(np.asarray(ag.center_of_mass(), dtype=np.float64))
    return np.asarray(pts, dtype=np.float64)


def _phosphate_point(phosphate):
    if phosphate.n_atoms == 1:
        return np.asarray(phosphate.positions[0], dtype=np.float64)
    return np.asarray(phosphate.center_of_mass(), dtype=np.float64)


def _feature_vector(groups, phosphate, pair_idx, repr_mode, feature_set, cv1_A, cv2_A):
    mode = _feature_mode(feature_set)
    parts = []
    residue_points = None
    if mode in {"residue-pair", "phosphate-distance", "combined", "combined-with-cv"}:
        residue_points = _points_from_groups(groups, repr_mode)
    if mode in {"residue-pair", "combined", "combined-with-cv"}:
        dmat = distance_array(residue_points, residue_points, box=None)
        parts.append(dmat[pair_idx])
    if mode in {"phosphate-distance", "combined", "combined-with-cv"}:
        p_pt = _phosphate_point(phosphate)
        parts.append(np.linalg.norm(residue_points - p_pt[None, :], axis=1))
    if mode in {"cv-only", "combined-with-cv"}:
        parts.append(np.asarray([cv1_A, cv2_A], dtype=np.float64))
    if not parts:
        raise ValueError(f"Unsupported feature set: {feature_set}")
    return np.concatenate([np.asarray(x, dtype=np.float64).reshape(-1) for x in parts])


def _iter_features_for_traj(
    top_path,
    traj_path,
    residue_keys,
    repr_mode,
    pair_idx,
    stride,
    align_sel,
    ref_traj,
    phosphate_sel,
    atom1,
    atom2,
    atom3,
    atom4,
    feature_set,
):
    u = mda.Universe(top_path, traj_path)
    _set_reference_alignment(u, top_path, ref_traj, align_sel)
    phosphate = u.select_atoms(phosphate_sel)
    if phosphate.n_atoms == 0:
        raise ValueError(f"Phosphate selection is empty for trajectory {traj_path}: {phosphate_sel}")
    groups = _build_residue_groups(u, residue_keys, repr_mode)

    for frame_idx, _ in enumerate(u.trajectory):
        if stride > 1 and (frame_idx % stride) != 0:
            continue
        pos = u.atoms.positions.astype(np.float64)
        cv1_A = _distance_A(pos, atom1, atom2)
        cv2_A = _distance_A(pos, atom3, atom4)
        yield {
            "feature": _feature_vector(groups, phosphate, pair_idx, repr_mode, feature_set, cv1_A, cv2_A),
            "frame_idx": int(frame_idx),
            "cv1_A": cv1_A,
            "cv2_A": cv2_A,
        }


def _standardized_feature_arrays(traj_list, iter_kwargs, mean_vec, scale_vec):
    for traj_path in traj_list:
        rows = []
        meta = []
        for frame in _iter_features_for_traj(traj_path=traj_path, **iter_kwargs):
            rows.append((frame["feature"] - mean_vec) / scale_vec)
            meta.append(frame)
        if rows:
            yield traj_path, np.asarray(rows, dtype=np.float64), meta


def _tica_axis_label(index, eigenvalues, timescales_ps=None):
    if index >= len(eigenvalues):
        return f"TIC{index + 1}"
    label = f"TIC{index + 1} (lambda={float(eigenvalues[index]):.3f})"
    if timescales_ps is not None and index < len(timescales_ps) and np.isfinite(timescales_ps[index]):
        label += f", t={float(timescales_ps[index]):.2f} ps"
    return label


def _implied_timescales(eigenvalues, lag_time_ps):
    vals = []
    for eig in eigenvalues:
        lam = abs(float(eig))
        if lam <= 0.0 or lam >= 1.0:
            vals.append(float("nan"))
        else:
            vals.append(float(-float(lag_time_ps) / np.log(lam)))
    return np.asarray(vals, dtype=np.float64)


def _norm01(values):
    arr = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(arr)
    if not finite.any():
        return np.zeros_like(arr)
    lo = float(np.min(arr[finite]))
    hi = float(np.max(arr[finite]))
    if abs(hi - lo) < 1e-12:
        out = np.zeros_like(arr)
        out[finite] = 0.5
        return out
    out = (arr - lo) / (hi - lo)
    out[~finite] = 0.0
    return np.clip(out, 0.0, 1.0)


def _progress_terms(cfg, cv1, cv2):
    start1 = float(getattr(cfg, "CURRENT_DISTANCE", np.nan))
    target1 = float(getattr(cfg, "FINAL_TARGET", getattr(cfg, "TARGET_CENTER", np.nan)))
    if np.isfinite(start1) and np.isfinite(target1) and abs(target1 - start1) > 1e-12:
        p1 = np.clip((cv1 - start1) / (target1 - start1), 0.0, 1.0)
    else:
        p1 = _norm01(cv1)

    start2 = float(getattr(cfg, "CURRENT_DISTANCE2", np.nan))
    target2 = float(getattr(cfg, "FINAL_TARGET2", np.nan))
    direction2 = str(getattr(cfg, "CV2_PROGRESS_DIRECTION", "decrease")).strip().lower()
    if np.isfinite(start2) and np.isfinite(target2) and abs(start2 - target2) > 1e-12:
        if direction2 == "increase":
            p2 = np.clip((cv2 - start2) / (target2 - start2), 0.0, 1.0)
        else:
            p2 = np.clip((start2 - cv2) / (start2 - target2), 0.0, 1.0)
    else:
        p2 = _norm01(cv2) if direction2 == "increase" else 1.0 - _norm01(cv2)
    return p1, p2


def _cluster_scores(scores_2d, n_clusters, seed):
    if len(scores_2d) < 2 or n_clusters <= 1:
        labels = np.zeros(len(scores_2d), dtype=int)
        centers = np.mean(scores_2d, axis=0, keepdims=True)
        return labels, centers
    k = int(min(max(2, n_clusters), len(scores_2d)))
    model = MiniBatchKMeans(
        n_clusters=k,
        random_state=int(seed),
        batch_size=min(4096, max(128, k * 4)),
        n_init=10,
        max_iter=200,
    )
    labels = model.fit_predict(scores_2d)
    return labels.astype(int), np.asarray(model.cluster_centers_, dtype=np.float64)


def _candidate_rows(rows, scores, labels, cfg, args):
    cv1 = np.asarray([float(row["cv1_A"]) for row in rows], dtype=np.float64)
    cv2 = np.asarray([float(row["cv2_A"]) for row in rows], dtype=np.float64)
    p1, p2 = _progress_terms(cfg, cv1, cv2)
    counts = Counter(map(int, labels))
    rarity = np.asarray([1.0 / np.sqrt(max(1, counts[int(label)])) for label in labels], dtype=np.float64)
    novelty = _norm01(rarity)
    radius = np.linalg.norm(scores[:, : min(2, scores.shape[1])] - scores[0, : min(2, scores.shape[1])], axis=1)
    slow_mode = _norm01(radius)
    score = (
        float(args.progress_weight) * p1
        + float(args.cv2_weight) * p2
        + float(args.novelty_weight) * novelty
        + float(args.slow_mode_weight) * slow_mode
    )

    selected = []
    selected_per_cluster = Counter()
    for idx in np.argsort(score)[::-1]:
        label = int(labels[int(idx)])
        if args.per_cluster_candidates > 0 and selected_per_cluster[label] >= int(args.per_cluster_candidates):
            continue
        rec = dict(rows[int(idx)])
        rec.update(
            {
                "cluster": label,
                "adaptive_score": float(score[int(idx)]),
                "cv1_progress": float(p1[int(idx)]),
                "cv2_progress": float(p2[int(idx)]),
                "novelty": float(novelty[int(idx)]),
                "slow_mode_distance": float(slow_mode[int(idx)]),
            }
        )
        selected.append(rec)
        selected_per_cluster[label] += 1
        if len(selected) >= int(args.top_candidates):
            break
    return selected, score


def _grid_fes(scores_2d, bins):
    hist, xedges, yedges = np.histogram2d(scores_2d[:, 0], scores_2d[:, 1], bins=bins, density=False)
    prob = hist / max(float(np.sum(hist)), 1.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        fes = -np.log(prob)
    fes[~np.isfinite(fes)] = np.nan
    if np.isfinite(fes).any():
        fes -= np.nanmin(fes[np.isfinite(fes)])
    xcent = 0.5 * (xedges[:-1] + xedges[1:])
    ycent = 0.5 * (yedges[:-1] + yedges[1:])
    return _grid_edges(xcent), _grid_edges(ycent), fes


def _plot_fes(out_path, scores, bins, eigenvalues, timescales_ps):
    xgrid, ygrid, fes = _grid_fes(scores[:, :2], int(bins))
    fig, ax = plt.subplots(figsize=(6.8, 5.8))
    mesh = ax.pcolormesh(xgrid, ygrid, np.ma.masked_invalid(fes.T), shading="auto", cmap="YlGnBu_r")
    fig.colorbar(mesh, ax=ax).set_label("Relative FES")
    ax.set_xlabel(_tica_axis_label(0, eigenvalues, timescales_ps))
    ax.set_ylabel(_tica_axis_label(1, eigenvalues, timescales_ps))
    ax.set_title("Phosphate-Pathway TICA FES")
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def _time_colored_segments(x, y, t):
    pts = np.column_stack([x, y]).reshape(-1, 1, 2)
    return np.concatenate([pts[:-1], pts[1:]], axis=1), Normalize(vmin=float(np.min(t)), vmax=float(np.max(t)))


def _plot_per_traj_projection(out_path, scores, time_ps, title, eigenvalues, timescales_ps):
    fig, ax = plt.subplots(figsize=(6.8, 5.8))
    if len(scores) > 1:
        segs, norm = _time_colored_segments(scores[:, 0], scores[:, 1], time_ps)
        lc = LineCollection(segs, cmap="viridis", norm=norm, linewidths=1.0, alpha=0.95, zorder=3)
        lc.set_array(np.asarray(time_ps[:-1], dtype=np.float64))
        ax.add_collection(lc)
        fig.colorbar(lc, ax=ax).set_label("Time (ps)")
    else:
        sc = ax.scatter(scores[:, 0], scores[:, 1], c=time_ps, cmap="viridis", s=18)
        fig.colorbar(sc, ax=ax).set_label("Time (ps)")
    ax.scatter([scores[0, 0]], [scores[0, 1]], s=70, color="#2a9d8f", edgecolor="white", linewidth=0.8, label="Start", zorder=4)
    ax.scatter([scores[-1, 0]], [scores[-1, 1]], s=75, color="#f4a261", edgecolor="white", linewidth=0.8, label="Final", zorder=4)
    ax.set_xlabel(_tica_axis_label(0, eigenvalues, timescales_ps))
    ax.set_ylabel(_tica_axis_label(1, eigenvalues, timescales_ps))
    ax.set_title(title)
    ax.legend(loc="best", frameon=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def _plot_per_traj_timeseries(out_path, scores, time_ps, title, eigenvalues, timescales_ps):
    fig, ax = plt.subplots(figsize=(8.0, 4.6))
    ax.plot(time_ps, scores[:, 0], label=_tica_axis_label(0, eigenvalues, timescales_ps), linewidth=1.35)
    if scores.shape[1] > 1:
        ax.plot(time_ps, scores[:, 1], label=_tica_axis_label(1, eigenvalues, timescales_ps), linewidth=1.15)
    ax.set_xlabel("Time (ps)")
    ax.set_ylabel("TICA score")
    ax.set_title(title)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def _plot_all_traj_projection(out_path, traj_series, eigenvalues, timescales_ps):
    fig, ax = plt.subplots(figsize=(8.4, 6.8))
    n_traj = len(traj_series)
    cmap = plt.get_cmap("tab10" if n_traj <= 10 else "tab20" if n_traj <= 20 else "gist_ncar", n_traj)
    for idx, (traj_label, traj_scores) in enumerate(traj_series):
        color = cmap(idx)
        ax.plot(traj_scores[:, 0], traj_scores[:, 1], color=color, linewidth=1.0, alpha=0.8, label=traj_label)
        ax.scatter([traj_scores[0, 0]], [traj_scores[0, 1]], s=16, color=color, edgecolor="white", linewidth=0.4, zorder=3)
    ax.set_xlabel(_tica_axis_label(0, eigenvalues, timescales_ps))
    ax.set_ylabel(_tica_axis_label(1, eigenvalues, timescales_ps))
    ax.set_title("Phosphate-Pathway TICA | TIC1 vs TIC2")
    ax.grid(alpha=0.18, linewidth=0.5)
    if n_traj <= 24:
        ncol = 1 if n_traj <= 10 else 2
        ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=True, fontsize=8, ncol=ncol)
        fig.tight_layout(rect=(0.0, 0.0, 0.82, 1.0))
    else:
        fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def _plot_tic_vs_cv(out_path, tic_vals, cv_vals, tic_name, cv_name):
    corr = _corrcoef_safe(tic_vals, cv_vals)
    fig, ax = plt.subplots(figsize=(6.5, 5.4))
    sc = ax.scatter(tic_vals, cv_vals, c=np.arange(len(tic_vals)), cmap="viridis", s=10, alpha=0.75)
    fig.colorbar(sc, ax=ax).set_label("Frame index")
    ax.set_xlabel(tic_name)
    ax.set_ylabel(cv_name)
    title = f"{tic_name} vs {cv_name}"
    if np.isfinite(corr):
        title += f" | r={corr:.3f}"
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)
    return corr


def _write_loading_csv(path, feature_names, component):
    rows = [
        {
            "feature_index": i,
            "feature": feature_names[i],
            "loading": float(component[i]),
            "abs_loading": abs(float(component[i])),
        }
        for i in range(len(feature_names))
    ]
    rows = sorted(rows, key=lambda row: row["abs_loading"], reverse=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["feature_index", "feature", "loading", "abs_loading"])
        writer.writeheader()
        writer.writerows(rows)
    return rows


def _plot_top_loadings(out_path, component_name, rows, top_n):
    top = list(reversed(rows[: int(top_n)]))
    labels = [row["feature"] for row in top]
    values = [float(row["loading"]) for row in top]
    colors = ["#d62828" if value > 0 else "#1d4e89" for value in values]
    fig, ax = plt.subplots(figsize=(11.5, 6.8))
    ax.barh(range(len(values)), values, color=colors, alpha=0.9)
    ax.axvline(0.0, color="#444444", linewidth=1.0)
    ax.set_yticks(range(len(values)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Signed TICA loading")
    ax.set_title(f"{component_name} top feature loadings")
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def _plot_candidates(out_path, scores, labels, candidates, eigenvalues, timescales_ps):
    fig, ax = plt.subplots(figsize=(7.0, 5.9))
    sc = ax.scatter(scores[:, 0], scores[:, 1], c=labels, cmap="tab20", s=8, alpha=0.35, linewidth=0)
    fig.colorbar(sc, ax=ax).set_label("TICA cluster")
    if candidates:
        ax.scatter(
            [float(row["TIC1"]) for row in candidates],
            [float(row["TIC2"]) for row in candidates],
            s=80,
            facecolor="none",
            edgecolor="#d62828",
            linewidth=1.4,
            label="Top seed candidates",
        )
        ax.legend(loc="best")
    ax.set_xlabel(_tica_axis_label(0, eigenvalues, timescales_ps))
    ax.set_ylabel(_tica_axis_label(1, eigenvalues, timescales_ps))
    ax.set_title("TICA Clusters and Adaptive Seed Candidates")
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def _plot_score_map(out_path, scores, adaptive_score, eigenvalues, timescales_ps):
    fig, ax = plt.subplots(figsize=(7.0, 5.9))
    sc = ax.scatter(scores[:, 0], scores[:, 1], c=adaptive_score, cmap="magma", s=10, alpha=0.8, linewidth=0)
    fig.colorbar(sc, ax=ax).set_label("Adaptive seed-candidate score")
    ax.set_xlabel(_tica_axis_label(0, eigenvalues, timescales_ps))
    ax.set_ylabel(_tica_axis_label(1, eigenvalues, timescales_ps))
    ax.set_title("Adaptive Score on TICA Space")
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def _build_arg_parser():
    parser = argparse.ArgumentParser(description="TICA on phosphate-pathway residue/phosphate distance features.")
    parser.add_argument("--config-module", default=None, help="Config module to use.")
    parser.add_argument("--top", default=None, help="Topology file (defaults to config.psf_file).")
    parser.add_argument("--traj-glob", default=None, help="Glob for DCD trajectories.")
    parser.add_argument("--max-traj", type=int, default=50, help="Maximum number of trajectories to include; <=0 means all.")
    parser.add_argument("--sample", choices=["first", "random"], default="first", help="Trajectory sampling mode.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling and clustering.")
    parser.add_argument("--align-sel", default="protein and backbone", help="Alignment selection.")
    parser.add_argument("--phosphate-sel", default="segid HETC and not name H*", help="Phosphate atom selection.")
    parser.add_argument("--residue-sel", default="protein", help="Residue pool selection.")
    parser.add_argument("--cutoff", type=float, default=6.0, help="Residue-phosphate cutoff in A.")
    parser.add_argument("--contact-stride", type=int, default=1, help="Stride for residue-discovery pass.")
    parser.add_argument("--stride", type=int, default=1, help="Stride for TICA feature generation.")
    parser.add_argument("--max-residues", type=int, default=60, help="Tighten the phosphate cutoff until at most this many residues remain.")
    parser.add_argument("--residue-repr", choices=["com", "ca"], default="com", help="Residue position representation.")
    parser.add_argument("--feature-set", choices=["combined", "residue-pair", "phosphate-distance", "cv-only", "combined-with-cv"], default="combined", help="TICA feature set.")
    parser.add_argument("--lag", type=int, default=5, help="TICA lag in processed frames after --stride.")
    parser.add_argument("--dim", type=int, default=4, help="Number of TICA dimensions to save.")
    parser.add_argument("--ridge", type=float, default=1e-6, help="Diagonal regularizer for the C00 covariance matrix.")
    parser.add_argument("--bins", type=int, default=60, help="Bins for global TIC1/TIC2 FES.")
    parser.add_argument("--clusters", type=int, default=100, help="TICA clusters for adaptive seed-candidate analysis.")
    parser.add_argument("--top-candidates", type=int, default=32, help="Number of seed-candidate rows to write.")
    parser.add_argument("--per-cluster-candidates", type=int, default=1, help="Max candidates per cluster; <=0 disables the cap.")
    parser.add_argument("--top-loadings", type=int, default=12, help="Top signed feature loadings to plot per TIC.")
    parser.add_argument("--progress-weight", type=float, default=1.0, help="Weight for CV1 unbinding progress in adaptive scoring.")
    parser.add_argument("--cv2-weight", type=float, default=0.35, help="Weight for CV2 progress in adaptive scoring.")
    parser.add_argument("--novelty-weight", type=float, default=0.25, help="Weight for cluster rarity in adaptive scoring.")
    parser.add_argument("--slow-mode-weight", type=float, default=0.25, help="Weight for distance from initial TICA point in adaptive scoring.")
    parser.add_argument("--run", default=None, help="Existing analysis run directory.")
    parser.add_argument("--runs-root", default=None, help="Root folder for analysis_runs.")
    parser.add_argument("--atom1", type=int, default=None, help="CV1 atom1 override.")
    parser.add_argument("--atom2", type=int, default=None, help="CV1 atom2 override.")
    parser.add_argument("--atom3", type=int, default=None, help="CV2 atom3 override.")
    parser.add_argument("--atom4", type=int, default=None, help="CV2 atom4 override.")
    return parser


def _write_residue_and_feature_tables(data_dir, ranked, residue_meta, feature_names, selection_mode):
    with open(os.path.join(data_dir, "phosphate_pathway_tica_residues.csv"), "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["rank", "segid", "resid", "resname", "label", "contact_frames", "min_distance_A", "selection_mode"])
        for rank, (key, stats) in enumerate(ranked, start=1):
            meta = residue_meta[key]
            writer.writerow([
                rank,
                meta["segid"],
                meta["resid"],
                meta["resname"],
                meta["label"],
                int(stats["contact_frames"]),
                float(stats["min_distance_A"]),
                selection_mode,
            ])

    with open(os.path.join(data_dir, "phosphate_pathway_tica_features.csv"), "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["feature_index", "feature"])
        for idx, name in enumerate(feature_names):
            writer.writerow([idx, name])


def _write_candidate_csv(path, candidate_rows):
    fieldnames = [
        "rank", "traj", "traj_path", "frame_idx", "time_ps", "cv1_A", "cv2_A", "TIC1", "TIC2",
        "cluster", "adaptive_score", "cv1_progress", "cv2_progress", "novelty", "slow_mode_distance",
    ]
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for rank, row in enumerate(candidate_rows, start=1):
            out = {key: row.get(key, "") for key in fieldnames}
            out["rank"] = rank
            writer.writerow(out)


def _write_report(path, args, traj_list, n_frames, n_pairs, lag_time_ps, residue_keys, n_features, effective_cutoff_A, component_summaries, candidate_rows):
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("# Phosphate-Pathway TICA Report\n\n")
        fh.write(f"- Trajectories: {len(traj_list)}\n")
        fh.write(f"- Frames used for TICA: {n_frames}\n")
        fh.write(f"- Time-lagged pairs: {n_pairs}\n")
        fh.write(f"- Lag: {int(args.lag)} processed frames ({lag_time_ps:.6f} ps)\n")
        fh.write(f"- Residues selected: {len(residue_keys)}\n")
        fh.write(f"- Features: {n_features} (`{args.feature_set}`)\n")
        fh.write(f"- Phosphate selection: `{args.phosphate_sel}`\n")
        fh.write(f"- Residue representation: `{args.residue_repr}`\n")
        fh.write(f"- Effective cutoff (A): {float(effective_cutoff_A):.3f}\n")
        fh.write("\n## Slow Modes\n\n")
        for comp in component_summaries:
            fh.write(f"### {comp['component']}\n\n")
            fh.write(f"- Eigenvalue: {float(comp['eigenvalue']):.6f}\n")
            ts = comp["implied_timescale_ps"]
            fh.write(f"- Implied timescale (ps): {'not finite' if ts is None else f'{ts:.6f}'}\n")
            if comp["corr_cv1"] is not None:
                fh.write(f"- Correlation to CV1: {comp['corr_cv1']:.6f}\n")
            if comp["corr_cv2"] is not None:
                fh.write(f"- Correlation to CV2: {comp['corr_cv2']:.6f}\n")
            fh.write("\n| Feature | Loading |\n")
            fh.write("|---|---:|\n")
            for rec in comp["top_features"][:6]:
                fh.write(f"| {rec['feature']} | {rec['loading']:.6f} |\n")
            fh.write("\n")
        fh.write("## Adaptive Seed-Candidate Diagnostic\n\n")
        fh.write(
            "The candidate score combines CV1 unbinding progress, optional CV2 progress, "
            "cluster rarity, and distance from the initial TICA point. It is a practical "
            "restart-ranking diagnostic inspired by Adaptive CVgen, not the original paper's "
            "`W * Theta` reward model.\n\n"
        )
        if candidate_rows:
            fh.write("| Rank | Trajectory | Frame | CV1 A | CV2 A | Cluster | Score |\n")
            fh.write("|---:|---|---:|---:|---:|---:|---:|\n")
            for rank, row in enumerate(candidate_rows[:10], start=1):
                fh.write(
                    f"| {rank} | {row['traj']} | {int(row['frame_idx'])} | "
                    f"{float(row['cv1_A']):.3f} | {float(row['cv2_A']):.3f} | "
                    f"{int(row['cluster'])} | {float(row['adaptive_score']):.6f} |\n"
                )


def main():
    args = _build_arg_parser().parse_args()
    if args.lag < 1:
        raise ValueError("--lag must be >= 1.")

    cfg_name = args.config_module or ("combined_2d" if os.path.exists(os.path.join(ROOT_DIR, "combined_2d.py")) else "config")
    cfg = _load_config(cfg_name)
    top_path = _abs_path(args.top or getattr(cfg, "psf_file", None))
    if top_path is None or not os.path.exists(top_path):
        raise FileNotFoundError(f"Topology not found: {top_path}")

    traj_glob = args.traj_glob or _resolve_default_traj_glob(cfg)
    traj_list = sorted(glob.glob(traj_glob))
    if not traj_list:
        raise FileNotFoundError(f"No DCD files found for pattern: {traj_glob}")
    traj_list = _select_trajs(traj_list, args.max_traj, args.sample, args.seed)

    run_dir = args.run
    if run_dir is not None and not os.path.isabs(run_dir):
        run_dir = os.path.join(ROOT_DIR, run_dir)
    if run_dir is None:
        run_dir = run_utils.prepare_run_dir(run_utils.default_time_tag(), root=args.runs_root)
    for subdir in ["data", os.path.join("figs", "analysis"), os.path.join("figs", "per_trajectory")]:
        os.makedirs(os.path.join(run_dir, subdir), exist_ok=True)
    data_dir = os.path.join(run_dir, "data")
    fig_dir = os.path.join(run_dir, "figs", "analysis")
    per_traj_dir = os.path.join(run_dir, "figs", "per_trajectory")

    run_utils.write_run_metadata(
        run_dir,
        {
            "script": "analysis/tica_phosphate_pathway.py",
            "traj_glob": traj_glob,
            "traj_count": len(traj_list),
            "config_module": cfg_name,
            "phosphate_sel": args.phosphate_sel,
            "residue_sel": args.residue_sel,
            "cutoff_A": float(args.cutoff),
            "residue_repr": args.residue_repr,
            "feature_set": args.feature_set,
            "lag_frames": int(args.lag),
            "config": _snapshot_cfg(cfg),
        },
    )

    counts, residue_meta, scanned_frames = _collect_candidate_residues(
        top_path, traj_list, args.phosphate_sel, args.residue_sel, args.cutoff, args.contact_stride
    )
    if not counts:
        raise RuntimeError("No protein residues were found near the phosphate selection.")

    residue_keys, ranked, effective_cutoff_A, selection_mode = _select_residues_by_cutoff(
        counts, residue_meta, args.cutoff, args.max_residues
    )
    if len(residue_keys) < 2 and _feature_mode(args.feature_set) != "cv-only":
        raise RuntimeError("Need at least two residues for the selected TICA feature set.")

    pair_idx = _pairwise_indices(len(residue_keys))
    feature_names = _feature_names(residue_keys, pair_idx, args.feature_set)
    n_features = len(feature_names)
    _write_residue_and_feature_tables(data_dir, ranked, residue_meta, feature_names, selection_mode)

    atom1 = int(args.atom1 if args.atom1 is not None else getattr(cfg, "ATOM1_INDEX", 0))
    atom2 = int(args.atom2 if args.atom2 is not None else getattr(cfg, "ATOM2_INDEX", 0))
    atom3 = int(args.atom3 if args.atom3 is not None else getattr(cfg, "ATOM3_INDEX", 0))
    atom4 = int(args.atom4 if args.atom4 is not None else getattr(cfg, "ATOM4_INDEX", 0))
    cv1_axis_label = getattr(cfg, "CV1_AXIS_LABEL", f"{getattr(cfg, 'CV1_LABEL', 'CV1')} (A)")
    cv2_axis_label = getattr(cfg, "CV2_AXIS_LABEL", f"{getattr(cfg, 'CV2_LABEL', 'CV2')} (A)")
    dt_ps = _dt_ps_from_cfg(cfg, args.stride)
    lag_time_ps = float(dt_ps) * int(args.lag)
    ref_traj = traj_list[0]
    iter_kwargs = {
        "top_path": top_path,
        "residue_keys": residue_keys,
        "repr_mode": args.residue_repr,
        "pair_idx": pair_idx,
        "stride": args.stride,
        "align_sel": args.align_sel,
        "ref_traj": ref_traj,
        "phosphate_sel": args.phosphate_sel,
        "atom1": atom1,
        "atom2": atom2,
        "atom3": atom3,
        "atom4": atom4,
        "feature_set": args.feature_set,
    }

    feature_sum = np.zeros(n_features, dtype=np.float64)
    feature_sumsq = np.zeros(n_features, dtype=np.float64)
    n_frames = 0
    for traj_path in traj_list:
        for frame in _iter_features_for_traj(traj_path=traj_path, **iter_kwargs):
            feat = frame["feature"]
            feature_sum += feat
            feature_sumsq += feat * feat
            n_frames += 1
    if n_frames <= int(args.lag):
        raise RuntimeError(f"Need more than lag={args.lag} processed frames for TICA; got {n_frames}.")

    mean_vec = feature_sum / float(n_frames)
    scale_vec = np.sqrt(np.maximum(feature_sumsq / float(n_frames) - mean_vec * mean_vec, 1e-12))

    c00 = np.zeros((n_features, n_features), dtype=np.float64)
    ctt = np.zeros((n_features, n_features), dtype=np.float64)
    c0t = np.zeros((n_features, n_features), dtype=np.float64)
    n_pairs = 0
    for _, x, _ in _standardized_feature_arrays(traj_list, iter_kwargs, mean_vec, scale_vec):
        if len(x) <= int(args.lag):
            continue
        x0 = x[:-int(args.lag)]
        xt = x[int(args.lag):]
        c00 += x0.T @ x0
        ctt += xt.T @ xt
        c0t += x0.T @ xt
        n_pairs += int(x0.shape[0])
    if n_pairs == 0:
        raise RuntimeError("No time-lagged frame pairs were available for TICA.")

    c00 = 0.5 * (c00 + ctt) / float(n_pairs)
    c00 = 0.5 * (c00 + c00.T) + float(args.ridge) * np.eye(n_features, dtype=np.float64)
    c0t_mean = c0t / float(n_pairs)
    c0t = 0.5 * (c0t_mean + c0t_mean.T)

    evals, evecs = eigh(c0t, c00, check_finite=False)
    order = np.argsort(evals)[::-1]
    evals = np.asarray(evals[order], dtype=np.float64)
    evecs = np.asarray(evecs[:, order], dtype=np.float64)
    dim = int(min(max(1, args.dim), evecs.shape[1]))
    evals = evals[:dim]
    components = evecs[:, :dim]
    timescales_ps = _implied_timescales(evals, lag_time_ps)

    np.save(os.path.join(data_dir, "phosphate_pathway_tica_mean.npy"), mean_vec)
    np.save(os.path.join(data_dir, "phosphate_pathway_tica_scale.npy"), scale_vec)
    np.save(os.path.join(data_dir, "phosphate_pathway_tica_eigenvalues.npy"), evals)
    np.save(os.path.join(data_dir, "phosphate_pathway_tica_components.npy"), components)
    np.save(os.path.join(data_dir, "phosphate_pathway_tica_timescales_ps.npy"), timescales_ps)

    scores_all = []
    frame_rows = []
    traj_series = []
    for traj_path, x, meta in _standardized_feature_arrays(traj_list, iter_kwargs, mean_vec, scale_vec):
        scores = x @ components
        traj_label = _traj_label(traj_path)
        time_ps = np.arange(len(scores), dtype=np.float64) * float(dt_ps)
        with open(os.path.join(data_dir, f"{traj_label}_tica_scores.csv"), "w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(["traj", "frame_idx", "time_ps", "cv1_A", "cv2_A"] + [f"TIC{i+1}" for i in range(scores.shape[1])])
            for idx, (row, frame) in enumerate(zip(scores, meta)):
                values = [traj_label, frame["frame_idx"], float(time_ps[idx]), frame["cv1_A"], frame["cv2_A"], *map(float, row)]
                writer.writerow(values)
                row_dict = {
                    "traj": traj_label,
                    "traj_path": os.path.abspath(traj_path),
                    "frame_idx": int(frame["frame_idx"]),
                    "time_ps": float(time_ps[idx]),
                    "cv1_A": float(frame["cv1_A"]),
                    "cv2_A": float(frame["cv2_A"]),
                }
                for comp_i, value in enumerate(row, start=1):
                    row_dict[f"TIC{comp_i}"] = float(value)
                frame_rows.append(row_dict)
        if scores.shape[1] >= 2:
            _plot_per_traj_projection(os.path.join(per_traj_dir, f"{traj_label}_tic1_tic2_time.png"), scores[:, :2], time_ps, f"{traj_label} | TIC1 vs TIC2", evals, timescales_ps)
            traj_series.append((traj_label, scores[:, :2]))
        _plot_per_traj_timeseries(os.path.join(per_traj_dir, f"{traj_label}_tic_timeseries.png"), scores, time_ps, f"{traj_label} | TICA scores", evals, timescales_ps)
        scores_all.append(scores)

    scores = np.vstack(scores_all)
    np.save(os.path.join(data_dir, "phosphate_pathway_tica_scores.npy"), scores)
    with open(os.path.join(data_dir, "phosphate_pathway_tica_scores_all.csv"), "w", newline="", encoding="utf-8") as fh:
        fieldnames = ["traj", "traj_path", "frame_idx", "time_ps", "cv1_A", "cv2_A"] + [f"TIC{i+1}" for i in range(scores.shape[1])]
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(frame_rows)

    cv1_all = np.asarray([row["cv1_A"] for row in frame_rows], dtype=np.float64)
    cv2_all = np.asarray([row["cv2_A"] for row in frame_rows], dtype=np.float64)
    component_summaries = []
    for tic_i in range(min(2, components.shape[1])):
        name = f"TIC{tic_i + 1}"
        loading_rows = _write_loading_csv(os.path.join(data_dir, f"{name.lower()}_feature_loadings.csv"), feature_names, components[:, tic_i])
        _plot_top_loadings(os.path.join(fig_dir, f"{name.lower()}_top_feature_loadings.png"), name, loading_rows, args.top_loadings)
        corr_cv1 = _plot_tic_vs_cv(os.path.join(fig_dir, f"{name.lower()}_vs_cv1.png"), scores[:, tic_i], cv1_all, name, cv1_axis_label)
        corr_cv2 = _plot_tic_vs_cv(os.path.join(fig_dir, f"{name.lower()}_vs_cv2.png"), scores[:, tic_i], cv2_all, name, cv2_axis_label)
        component_summaries.append(
            {
                "component": name,
                "eigenvalue": float(evals[tic_i]),
                "implied_timescale_ps": None if not np.isfinite(timescales_ps[tic_i]) else float(timescales_ps[tic_i]),
                "corr_cv1": None if not np.isfinite(corr_cv1) else float(corr_cv1),
                "corr_cv2": None if not np.isfinite(corr_cv2) else float(corr_cv2),
                "top_features": loading_rows[: int(min(10, len(loading_rows)))],
            }
        )

    candidate_rows = []
    if scores.shape[1] >= 2:
        _plot_fes(os.path.join(fig_dir, "phosphate_pathway_tica_fes.png"), scores[:, :2], args.bins, evals, timescales_ps)
        _plot_all_traj_projection(os.path.join(fig_dir, "phosphate_pathway_tic1_tic2_all_trajectories.png"), traj_series, evals, timescales_ps)
        labels, centers = _cluster_scores(scores[:, :2], int(args.clusters), int(args.seed))
        candidate_rows, adaptive_score = _candidate_rows(frame_rows, scores, labels, cfg, args)
        np.save(os.path.join(data_dir, "phosphate_pathway_tica_cluster_centers.npy"), centers)
        np.save(os.path.join(data_dir, "phosphate_pathway_tica_cluster_labels.npy"), labels)
        np.save(os.path.join(data_dir, "phosphate_pathway_tica_adaptive_score.npy"), adaptive_score)
        _write_candidate_csv(os.path.join(data_dir, "tica_adaptive_seed_candidates.csv"), candidate_rows)
        _plot_candidates(os.path.join(fig_dir, "tica_clusters_seed_candidates.png"), scores[:, :2], labels, candidate_rows, evals, timescales_ps)
        _plot_score_map(os.path.join(fig_dir, "tica_adaptive_score_map.png"), scores[:, :2], adaptive_score, evals, timescales_ps)

    summary = {
        "selected_residue_count": int(len(residue_keys)),
        "feature_count": int(n_features),
        "frames_scanned_for_contacts": int(scanned_frames),
        "frames_used_for_tica": int(n_frames),
        "time_lagged_pairs": int(n_pairs),
        "trajectory_count": int(len(traj_list)),
        "input_cutoff_A": float(args.cutoff),
        "effective_cutoff_A": float(effective_cutoff_A),
        "selection_mode": selection_mode,
        "feature_set": args.feature_set,
        "lag_frames": int(args.lag),
        "lag_time_ps": float(lag_time_ps),
        "eigenvalues": [float(x) for x in evals],
        "implied_timescales_ps": [None if not np.isfinite(x) else float(x) for x in timescales_ps],
        "selected_residues": [residue_meta[key] | {"contact_frames": int(counts[key])} for key in residue_keys],
        "component_summaries": component_summaries,
        "adaptive_seed_candidates": candidate_rows,
        "adaptive_score_note": "Adaptive-CVgen-inspired diagnostic only; not the paper's W*Theta reward model.",
    }
    with open(os.path.join(data_dir, "phosphate_pathway_tica_summary.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    _write_report(
        os.path.join(data_dir, "phosphate_pathway_tica_report.md"),
        args,
        traj_list,
        n_frames,
        n_pairs,
        lag_time_ps,
        residue_keys,
        n_features,
        effective_cutoff_A,
        component_summaries,
        candidate_rows,
    )

    run_utils.cleanup_empty_dirs(run_dir)
    print(f"Saved phosphate-pathway TICA run: {run_dir}")


if __name__ == "__main__":
    main()
