import argparse
import csv
import glob
import importlib
import json
import os
import random
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
    from MDAnalysis.transformations import fit_rot_trans
except Exception as exc:
    raise ImportError("MDAnalysis is required for phosphate-pathway PCA.") from exc

try:
    from sklearn.decomposition import IncrementalPCA
except Exception as exc:
    raise ImportError("scikit-learn is required for phosphate-pathway PCA.") from exc

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from analysis import run_utils


def _abs_path(path):
    if path is None:
        return None
    if os.path.isabs(path):
        return path
    return os.path.join(ROOT_DIR, path)


def _load_config(module_name):
    try:
        return importlib.import_module(module_name)
    except Exception:
        return importlib.import_module("config")


def _resolve_default_traj_glob(cfg):
    base = getattr(cfg, "RESULTS_TRAJ_DIR", os.path.join(ROOT_DIR, "results_PPO", "dcd_trajs"))
    if not os.path.isabs(base):
        base = os.path.join(ROOT_DIR, base)
    return os.path.join(base, "*.dcd")


def _select_trajs(traj_list, max_traj, sample_mode, seed):
    if max_traj is None or max_traj <= 0 or len(traj_list) <= max_traj:
        return traj_list
    if sample_mode == "random":
        rng = random.Random(seed)
        return sorted(rng.sample(traj_list, max_traj))
    return traj_list[:max_traj]


def _snapshot_cfg(cfg):
    keys = [
        "ATOM1_INDEX", "ATOM2_INDEX", "ATOM3_INDEX", "ATOM4_INDEX",
        "CURRENT_DISTANCE", "FINAL_TARGET", "CURRENT_DISTANCE2", "FINAL_TARGET2",
        "TARGET_MIN", "TARGET_MAX", "TARGET2_MIN", "TARGET2_MAX",
        "stepsize", "dcdfreq_mfpt", "DCD_REPORT_INTERVAL",
        "RESULTS_DIR", "RESULTS_TRAJ_DIR",
    ]
    unit_mod = getattr(cfg, "u", None) or getattr(cfg, "unit", None)
    snap = {}
    for key in keys:
        if not hasattr(cfg, key):
            continue
        val = getattr(cfg, key)
        try:
            if hasattr(val, "value_in_unit") and unit_mod is not None:
                val = float(val.value_in_unit(unit_mod.picoseconds))
        except Exception:
            pass
        snap[key] = val
    return snap


def _pairwise_indices(n_items):
    return np.triu_indices(n_items, k=1)


def _grid_edges(centers):
    centers = np.asarray(centers, dtype=np.float64)
    if centers.size == 1:
        return np.array([centers[0] - 0.5, centers[0] + 0.5], dtype=np.float64)
    mids = 0.5 * (centers[:-1] + centers[1:])
    left = centers[0] - 0.5 * (centers[1] - centers[0])
    right = centers[-1] + 0.5 * (centers[-1] - centers[-2])
    return np.concatenate([[left], mids, [right]])


def _traj_label(path):
    return os.path.splitext(os.path.basename(path))[0]


def _residue_key(residue):
    return (str(residue.segid), int(residue.resid), str(residue.resname))


def _residue_label(key):
    segid, resid, resname = key
    if segid:
        return f"{segid}:{resname}{resid}"
    return f"{resname}{resid}"


def _distance_A(pos, i, j):
    return float(np.linalg.norm(pos[int(i)] - pos[int(j)]))


def _dt_ps_from_cfg(cfg, stride):
    unit_mod = getattr(cfg, "u", None) or getattr(cfg, "unit", None)
    if unit_mod is None or not hasattr(cfg, "stepsize"):
        return float(max(1, stride))
    report_interval = int(getattr(cfg, "DCD_REPORT_INTERVAL", getattr(cfg, "dcdfreq_mfpt", 1)))
    return float(cfg.stepsize.value_in_unit(unit_mod.picoseconds)) * report_interval * max(1, int(stride))


def _set_reference_alignment(u, top_path, ref_path, align_sel):
    if not align_sel:
        return
    ref = mda.Universe(top_path, ref_path)
    ref.trajectory[0]
    ref_atoms = ref.select_atoms(align_sel)
    mob_atoms = u.select_atoms(align_sel)
    if ref_atoms.n_atoms == 0 or mob_atoms.n_atoms == 0:
        raise ValueError(f"Alignment selection is empty: {align_sel}")
    u.trajectory.add_transformations(fit_rot_trans(mob_atoms, ref_atoms))


def _collect_candidate_residues(top_path, traj_list, phosphate_sel, residue_sel, cutoff_A, stride):
    counts = Counter()
    residue_meta = {}
    total_frames = 0

    for traj_path in traj_list:
        u = mda.Universe(top_path, traj_path)
        phosphate = u.select_atoms(phosphate_sel)
        if phosphate.n_atoms == 0:
            raise ValueError(f"Phosphate selection is empty for trajectory {traj_path}: {phosphate_sel}")
        near_atoms = u.select_atoms(f"({residue_sel}) and around {float(cutoff_A)} ({phosphate_sel})", updating=True)
        for frame_idx, _ in enumerate(u.trajectory):
            if stride > 1 and (frame_idx % stride) != 0:
                continue
            total_frames += 1
            seen = set()
            for residue in near_atoms.residues:
                key = _residue_key(residue)
                if key in seen:
                    continue
                seen.add(key)
                counts[key] += 1
                residue_meta[key] = {
                    "segid": key[0],
                    "resid": key[1],
                    "resname": key[2],
                    "label": _residue_label(key),
                }

    return counts, residue_meta, total_frames


def _rank_residues(counts, residue_meta, max_residues):
    ranked = sorted(
        counts.items(),
        key=lambda kv: (-kv[1], kv[0][0], kv[0][1], kv[0][2]),
    )
    if max_residues is not None and max_residues > 0:
        ranked = ranked[:int(max_residues)]
    residue_keys = [key for key, _ in ranked]
    return residue_keys, ranked


def _build_residue_groups(u, residue_keys, repr_mode):
    residue_map = {_residue_key(res): res for res in u.residues}
    groups = []
    for key in residue_keys:
        residue = residue_map.get(key)
        if residue is None:
            raise KeyError(f"Residue {key} not found in topology.")
        if repr_mode == "ca":
            ag = residue.atoms.select_atoms("name CA")
            if ag.n_atoms == 0:
                ag = residue.atoms.select_atoms("not name H*")
        else:
            ag = residue.atoms.select_atoms("not name H*")
        if ag.n_atoms == 0:
            ag = residue.atoms
        groups.append(ag)
    return groups


def _residue_points(groups, repr_mode):
    pts = []
    for ag in groups:
        if repr_mode == "ca":
            pts.append(np.asarray(ag.positions[0], dtype=np.float64))
        else:
            pts.append(np.asarray(ag.center_of_mass(), dtype=np.float64))
    return np.asarray(pts, dtype=np.float64)


def _feature_vector(groups, pair_idx, repr_mode):
    pts = _residue_points(groups, repr_mode)
    dmat = distance_array(pts, pts, box=None)
    return dmat[pair_idx]


def _iter_features_for_traj(
    top_path,
    traj_path,
    residue_keys,
    repr_mode,
    pair_idx,
    stride,
    align_sel,
    ref_traj,
    atom1,
    atom2,
    atom3,
    atom4,
):
    u = mda.Universe(top_path, traj_path)
    _set_reference_alignment(u, top_path, ref_traj, align_sel)
    groups = _build_residue_groups(u, residue_keys, repr_mode)

    for frame_idx, _ in enumerate(u.trajectory):
        if stride > 1 and (frame_idx % stride) != 0:
            continue
        pos = u.atoms.positions.astype(np.float64)
        yield {
            "feature": _feature_vector(groups, pair_idx, repr_mode),
            "frame_idx": int(frame_idx),
            "cv1_A": _distance_A(pos, atom1, atom2),
            "cv2_A": _distance_A(pos, atom3, atom4),
        }


def _plot_scree(out_path, explained):
    plt.figure(figsize=(7.0, 4.6))
    x = np.arange(1, len(explained) + 1)
    plt.plot(x, explained, marker="o")
    plt.xlabel("Component")
    plt.ylabel("Explained variance ratio")
    plt.title("Phosphate-Pathway PCA Scree")
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


def _plot_fes(out_path, scores, bins):
    x = scores[:, 0]
    y = scores[:, 1]
    hist, xedges, yedges = np.histogram2d(x, y, bins=bins, density=False)
    prob = hist / max(float(np.sum(hist)), 1.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        fes = -np.log(prob)
    fes[~np.isfinite(fes)] = np.nan
    if np.isfinite(fes).any():
        fes -= np.nanmin(fes[np.isfinite(fes)])

    xcent = 0.5 * (xedges[:-1] + xedges[1:])
    ycent = 0.5 * (yedges[:-1] + yedges[1:])
    xgrid = _grid_edges(xcent)
    ygrid = _grid_edges(ycent)

    fig, ax = plt.subplots(figsize=(6.7, 5.7))
    mesh = ax.pcolormesh(xgrid, ygrid, np.ma.masked_invalid(fes.T), shading="auto", cmap="YlGnBu_r")
    cbar = fig.colorbar(mesh, ax=ax)
    cbar.set_label("Relative FES")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("Phosphate-Pathway PCA FES")
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def _time_colored_segments(x, y, t):
    pts = np.column_stack([x, y]).reshape(-1, 1, 2)
    return np.concatenate([pts[:-1], pts[1:]], axis=1), Normalize(vmin=float(np.min(t)), vmax=float(np.max(t)))


def _plot_per_traj_projection(out_path, scores, time_ps, title):
    fig, ax = plt.subplots(figsize=(6.8, 5.8))
    if len(scores) > 1:
        segs, norm = _time_colored_segments(scores[:, 0], scores[:, 1], time_ps)
        lc = LineCollection(segs, cmap="viridis", norm=norm, linewidths=1.0, alpha=0.95, zorder=3)
        lc.set_array(np.asarray(time_ps[:-1], dtype=np.float64))
        ax.add_collection(lc)
        cbar = fig.colorbar(lc, ax=ax)
        cbar.set_label("Time (ps)")
    else:
        sc = ax.scatter(scores[:, 0], scores[:, 1], c=time_ps, cmap="viridis", s=16)
        cbar = fig.colorbar(sc, ax=ax)
        cbar.set_label("Time (ps)")

    ax.scatter([scores[0, 0]], [scores[0, 1]], s=70, color="#2a9d8f", edgecolor="white", linewidth=0.8, label="Start", zorder=4)
    ax.scatter([scores[-1, 0]], [scores[-1, 1]], s=75, color="#f4a261", edgecolor="white", linewidth=0.8, label="Final", zorder=4)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title(title)
    ax.legend(loc="best", frameon=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def _plot_per_traj_timeseries(out_path, scores, time_ps, title):
    fig, ax = plt.subplots(figsize=(8.0, 4.6))
    ax.plot(time_ps, scores[:, 0], label="PC1", linewidth=1.4)
    if scores.shape[1] > 1:
        ax.plot(time_ps, scores[:, 1], label="PC2", linewidth=1.2)
    ax.set_xlabel("Time (ps)")
    ax.set_ylabel("Score")
    ax.set_title(title)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def _plot_all_traj_projection(out_path, traj_series, title):
    fig, ax = plt.subplots(figsize=(8.4, 6.8))
    n_traj = len(traj_series)
    if n_traj <= 10:
        cmap = plt.get_cmap("tab10", n_traj)
    elif n_traj <= 20:
        cmap = plt.get_cmap("tab20", n_traj)
    else:
        cmap = plt.get_cmap("gist_ncar", n_traj)

    for idx, (traj_label, traj_scores) in enumerate(traj_series):
        color = cmap(idx)
        x = traj_scores[:, 0]
        y = traj_scores[:, 1]
        ax.plot(x, y, color=color, linewidth=1.0, alpha=0.8, label=traj_label, zorder=2)
        ax.scatter([x[0]], [y[0]], s=16, color=color, edgecolor="white", linewidth=0.4, alpha=0.95, zorder=3)

    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title(title)
    ax.grid(alpha=0.18, linewidth=0.5)
    if n_traj <= 24:
        ncol = 1 if n_traj <= 10 else 2
        ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=True, fontsize=8, ncol=ncol)
        fig.tight_layout(rect=(0.0, 0.0, 0.82, 1.0))
    else:
        fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def _pair_records_for_component(component, residue_keys, pair_idx):
    records = []
    for feat_idx, (i, j) in enumerate(zip(pair_idx[0], pair_idx[1])):
        loading = float(component[int(feat_idx)])
        label_i = _residue_label(residue_keys[int(i)])
        label_j = _residue_label(residue_keys[int(j)])
        records.append(
            {
                "pair_index": int(feat_idx),
                "residue_i": label_i,
                "residue_j": label_j,
                "pair_label": f"{label_i} -- {label_j}",
                "loading": loading,
                "abs_loading": abs(loading),
            }
        )
    return records


def _write_component_loading_csv(out_path, records):
    with open(out_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["pair_index", "residue_i", "residue_j", "pair_label", "loading", "abs_loading"])
        for rec in records:
            writer.writerow([
                rec["pair_index"],
                rec["residue_i"],
                rec["residue_j"],
                rec["pair_label"],
                rec["loading"],
                rec["abs_loading"],
            ])


def _plot_component_loadings(out_path, component_name, records, top_n):
    top = sorted(records, key=lambda rec: rec["abs_loading"], reverse=True)[:int(top_n)]
    top = list(reversed(top))
    labels = [rec["pair_label"] for rec in top]
    values = [rec["loading"] for rec in top]
    colors = ["#d62828" if val > 0 else "#1d4e89" for val in values]

    fig, ax = plt.subplots(figsize=(11.5, 6.5))
    ax.barh(range(len(values)), values, color=colors, alpha=0.9)
    ax.axvline(0.0, color="#444444", linewidth=1.0)
    ax.set_yticks(range(len(values)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Signed loading")
    ax.set_title(f"{component_name} top residue-pair loadings")
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def _corrcoef_safe(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.size < 2 or y.size < 2:
        return float("nan")
    if np.allclose(np.std(x), 0.0) or np.allclose(np.std(y), 0.0):
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _plot_pc_vs_cv(out_path, pc_vals, cv_vals, pc_name, cv_name):
    corr = _corrcoef_safe(pc_vals, cv_vals)
    fig, ax = plt.subplots(figsize=(6.5, 5.4))
    sc = ax.scatter(pc_vals, cv_vals, c=np.arange(len(pc_vals)), cmap="viridis", s=10, alpha=0.75)
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("Frame index")
    ax.set_xlabel(pc_name)
    ax.set_ylabel(cv_name)
    title = f"{pc_name} vs {cv_name}"
    if np.isfinite(corr):
        title += f" | r={corr:.3f}"
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)
    return corr


def _component_definition_text(component_name, records, corr_cv1, corr_cv2, top_n=3):
    top = sorted(records, key=lambda rec: rec["abs_loading"], reverse=True)[:int(top_n)]
    pair_text = ", ".join(rec["pair_label"] for rec in top)
    parts = [f"{component_name} is dominated by residue-pair distance changes involving {pair_text}."]
    if np.isfinite(corr_cv1) or np.isfinite(corr_cv2):
        corr_bits = []
        if np.isfinite(corr_cv1):
            corr_bits.append(f"corr({component_name}, CV1)={corr_cv1:.3f}")
        if np.isfinite(corr_cv2):
            corr_bits.append(f"corr({component_name}, CV2)={corr_cv2:.3f}")
        parts.append(" ".join(corr_bits) + ".")
    return " ".join(parts)


def main():
    parser = argparse.ArgumentParser(description="PCA on phosphate-pathway residue-pair distances.")
    parser.add_argument("--config-module", default=None, help="Config module to use.")
    parser.add_argument("--top", default=None, help="Topology file (defaults to config.psf_file).")
    parser.add_argument("--traj-glob", default=None, help="Glob for DCD trajectories.")
    parser.add_argument("--max-traj", type=int, default=50, help="Maximum number of trajectories to include.")
    parser.add_argument("--sample", choices=["first", "random"], default="first", help="Trajectory sampling mode.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling.")
    parser.add_argument("--align-sel", default="protein and backbone", help="Alignment selection.")
    parser.add_argument("--phosphate-sel", default="segid HETC and not name H*", help="Phosphate selection.")
    parser.add_argument("--residue-sel", default="protein", help="Residue pool selection.")
    parser.add_argument("--cutoff", type=float, default=6.0, help="Residue-phosphate cutoff in Å.")
    parser.add_argument("--contact-stride", type=int, default=1, help="Stride for residue-discovery pass.")
    parser.add_argument("--stride", type=int, default=1, help="Stride for PCA passes.")
    parser.add_argument("--max-residues", type=int, default=60, help="Keep the top-contact residues if exceeded.")
    parser.add_argument("--residue-repr", choices=["com", "ca"], default="com", help="Residue position representation.")
    parser.add_argument("--n-components", type=int, default=6, help="Number of PCA components.")
    parser.add_argument("--batch-size", type=int, default=2000, help="IncrementalPCA batch size.")
    parser.add_argument("--bins", type=int, default=60, help="Bins for global PC1/PC2 FES.")
    parser.add_argument("--top-loadings", type=int, default=12, help="Top signed residue-pair loadings to plot per component.")
    parser.add_argument("--run", default=None, help="Existing analysis run directory.")
    parser.add_argument("--runs-root", default=None, help="Root folder for analysis_runs.")
    parser.add_argument("--atom1", type=int, default=None, help="CV1 atom1 override.")
    parser.add_argument("--atom2", type=int, default=None, help="CV1 atom2 override.")
    parser.add_argument("--atom3", type=int, default=None, help="CV2 atom3 override.")
    parser.add_argument("--atom4", type=int, default=None, help="CV2 atom4 override.")
    args = parser.parse_args()

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

    run_utils.write_run_metadata(
        run_dir,
        {
            "script": "analysis/pca_phosphate_pathway.py",
            "traj_glob": traj_glob,
            "traj_count": len(traj_list),
            "config_module": cfg_name,
            "phosphate_sel": args.phosphate_sel,
            "residue_sel": args.residue_sel,
            "cutoff_A": float(args.cutoff),
            "residue_repr": args.residue_repr,
            "max_residues": args.max_residues,
            "config": _snapshot_cfg(cfg),
        },
    )

    counts, residue_meta, scanned_frames = _collect_candidate_residues(
        top_path,
        traj_list,
        args.phosphate_sel,
        args.residue_sel,
        args.cutoff,
        args.contact_stride,
    )
    if not counts:
        raise RuntimeError("No protein residues were found near the phosphate selection.")

    residue_keys, ranked = _rank_residues(counts, residue_meta, args.max_residues)
    if len(residue_keys) < 2:
        raise RuntimeError("Need at least two residues for pair-distance PCA.")

    pair_idx = _pairwise_indices(len(residue_keys))
    n_features = len(pair_idx[0])
    atom1 = int(args.atom1 if args.atom1 is not None else getattr(cfg, "ATOM1_INDEX", 0))
    atom2 = int(args.atom2 if args.atom2 is not None else getattr(cfg, "ATOM2_INDEX", 0))
    atom3 = int(args.atom3 if args.atom3 is not None else getattr(cfg, "ATOM3_INDEX", 0))
    atom4 = int(args.atom4 if args.atom4 is not None else getattr(cfg, "ATOM4_INDEX", 0))
    dt_ps = _dt_ps_from_cfg(cfg, args.stride)

    data_dir = os.path.join(run_dir, "data")
    fig_dir = os.path.join(run_dir, "figs", "analysis")
    per_traj_dir = os.path.join(run_dir, "figs", "per_trajectory")

    residue_csv = os.path.join(data_dir, "phosphate_pathway_residues.csv")
    with open(residue_csv, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["rank", "segid", "resid", "resname", "label", "contact_frames"])
        for rank, (key, count) in enumerate(ranked, start=1):
            meta = residue_meta[key]
            writer.writerow([rank, meta["segid"], meta["resid"], meta["resname"], meta["label"], int(count)])

    pair_csv = os.path.join(data_dir, "phosphate_pathway_pairs.csv")
    with open(pair_csv, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["pair_index", "residue_i", "residue_j"])
        for idx, (i, j) in enumerate(zip(pair_idx[0], pair_idx[1])):
            writer.writerow([idx, _residue_label(residue_keys[int(i)]), _residue_label(residue_keys[int(j)])])

    mean_vec = np.zeros(n_features, dtype=np.float64)
    n_frames = 0
    ref_traj = traj_list[0]
    for traj_path in traj_list:
        for frame in _iter_features_for_traj(
            top_path, traj_path, residue_keys, args.residue_repr, pair_idx, args.stride,
            args.align_sel, ref_traj, atom1, atom2, atom3, atom4,
        ):
            mean_vec += frame["feature"]
            n_frames += 1
    if n_frames == 0:
        raise RuntimeError("No frames were processed for PCA.")
    mean_vec /= float(n_frames)

    pca = IncrementalPCA(n_components=min(int(args.n_components), int(n_features)), batch_size=int(args.batch_size))
    for traj_path in traj_list:
        batch = []
        for frame in _iter_features_for_traj(
            top_path, traj_path, residue_keys, args.residue_repr, pair_idx, args.stride,
            args.align_sel, ref_traj, atom1, atom2, atom3, atom4,
        ):
            batch.append(frame["feature"] - mean_vec)
            if len(batch) >= args.batch_size:
                pca.partial_fit(np.asarray(batch, dtype=np.float64))
                batch = []
        if batch:
            pca.partial_fit(np.asarray(batch, dtype=np.float64))

    scores_all = []
    traj_series = []
    frame_rows = []
    for traj_path in traj_list:
        traj_features = []
        frame_info = []
        for frame in _iter_features_for_traj(
            top_path, traj_path, residue_keys, args.residue_repr, pair_idx, args.stride,
            args.align_sel, ref_traj, atom1, atom2, atom3, atom4,
        ):
            traj_features.append(frame["feature"] - mean_vec)
            frame_info.append(frame)
        if not traj_features:
            continue

        traj_features = np.asarray(traj_features, dtype=np.float64)
        traj_scores = pca.transform(traj_features)
        traj_label = _traj_label(traj_path)
        time_ps = np.arange(len(traj_scores), dtype=np.float64) * float(dt_ps)

        per_csv = os.path.join(data_dir, f"{traj_label}_pc_scores.csv")
        with open(per_csv, "w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            header = ["traj", "frame_idx", "time_ps", "cv1_A", "cv2_A"] + [f"PC{i+1}" for i in range(traj_scores.shape[1])]
            writer.writerow(header)
            for idx, (row, meta) in enumerate(zip(traj_scores, frame_info)):
                writer.writerow([traj_label, meta["frame_idx"], float(time_ps[idx]), meta["cv1_A"], meta["cv2_A"], *map(float, row)])
                frame_rows.append([traj_label, meta["frame_idx"], float(time_ps[idx]), meta["cv1_A"], meta["cv2_A"], *map(float, row)])

        _plot_per_traj_projection(
            os.path.join(per_traj_dir, f"{traj_label}_pc1_pc2_time.png"),
            traj_scores[:, :2],
            time_ps,
            title=f"{traj_label} | PC1 vs PC2",
        )
        _plot_per_traj_timeseries(
            os.path.join(per_traj_dir, f"{traj_label}_pc_timeseries.png"),
            traj_scores,
            time_ps,
            title=f"{traj_label} | PCA scores",
        )
        scores_all.append(traj_scores)
        traj_series.append((traj_label, traj_scores[:, :2]))

    if not scores_all:
        raise RuntimeError("No trajectory scores were produced.")

    scores = np.vstack(scores_all)
    frame_rows_arr = np.asarray(frame_rows, dtype=object)
    cv1_all = np.asarray(frame_rows_arr[:, 3], dtype=np.float64)
    cv2_all = np.asarray(frame_rows_arr[:, 4], dtype=np.float64)
    np.save(os.path.join(data_dir, "phosphate_pathway_mean.npy"), mean_vec)
    np.save(os.path.join(data_dir, "phosphate_pathway_components.npy"), pca.components_)
    np.save(os.path.join(data_dir, "phosphate_pathway_explained.npy"), pca.explained_variance_ratio_)
    np.save(os.path.join(data_dir, "phosphate_pathway_scores.npy"), scores)

    pooled_csv = os.path.join(data_dir, "phosphate_pathway_scores_all.csv")
    with open(pooled_csv, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        header = ["traj", "frame_idx", "time_ps", "cv1_A", "cv2_A"] + [f"PC{i+1}" for i in range(scores.shape[1])]
        writer.writerow(header)
        writer.writerows(frame_rows)

    _plot_scree(os.path.join(fig_dir, "phosphate_pathway_pca_scree.png"), pca.explained_variance_ratio_)
    if scores.shape[1] >= 2:
        _plot_fes(os.path.join(fig_dir, "phosphate_pathway_pca_fes.png"), scores[:, :2], bins=int(args.bins))
        _plot_all_traj_projection(
            os.path.join(fig_dir, "phosphate_pathway_pc1_pc2_all_trajectories.png"),
            traj_series,
            title="Phosphate-Pathway PCA | PC1 vs PC2 (all trajectories)",
        )

    component_summaries = []
    for pc_i in range(min(2, pca.components_.shape[0])):
        component_name = f"PC{pc_i + 1}"
        records = _pair_records_for_component(pca.components_[pc_i], residue_keys, pair_idx)
        records_sorted = sorted(records, key=lambda rec: rec["abs_loading"], reverse=True)
        _write_component_loading_csv(
            os.path.join(data_dir, f"{component_name.lower()}_pair_loadings.csv"),
            records_sorted,
        )
        _plot_component_loadings(
            os.path.join(fig_dir, f"{component_name.lower()}_top_pair_loadings.png"),
            component_name,
            records_sorted,
            top_n=int(args.top_loadings),
        )

        corr_cv1 = _plot_pc_vs_cv(
            os.path.join(fig_dir, f"{component_name.lower()}_vs_cv1.png"),
            scores[:, pc_i],
            cv1_all,
            component_name,
            "CV1 (Å)",
        )
        corr_cv2 = _plot_pc_vs_cv(
            os.path.join(fig_dir, f"{component_name.lower()}_vs_cv2.png"),
            scores[:, pc_i],
            cv2_all,
            component_name,
            "CV2 (Å)",
        )
        component_summaries.append(
            {
                "component": component_name,
                "corr_cv1": None if not np.isfinite(corr_cv1) else float(corr_cv1),
                "corr_cv2": None if not np.isfinite(corr_cv2) else float(corr_cv2),
                "top_pairs": records_sorted[: int(min(10, len(records_sorted)))],
                "definition": _component_definition_text(component_name, records_sorted, corr_cv1, corr_cv2),
            }
        )

    summary = {
        "selected_residue_count": int(len(residue_keys)),
        "selected_pair_count": int(n_features),
        "frames_scanned_for_contacts": int(scanned_frames),
        "frames_used_for_pca": int(n_frames),
        "trajectory_count": int(len(traj_list)),
        "phosphate_sel": args.phosphate_sel,
        "residue_sel": args.residue_sel,
        "cutoff_A": float(args.cutoff),
        "residue_repr": args.residue_repr,
        "explained_variance_ratio": [float(x) for x in pca.explained_variance_ratio_],
        "selected_residues": [residue_meta[key] | {"contact_frames": int(counts[key])} for key in residue_keys],
        "component_summaries": component_summaries,
    }
    with open(os.path.join(data_dir, "phosphate_pathway_summary.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    report_md = os.path.join(data_dir, "phosphate_pathway_report.md")
    with open(report_md, "w", encoding="utf-8") as fh:
        fh.write("# Phosphate-Pathway PCA Report\n\n")
        fh.write(f"- Trajectories: {len(traj_list)}\n")
        fh.write(f"- Frames used for PCA: {n_frames}\n")
        fh.write(f"- Residues selected: {len(residue_keys)}\n")
        fh.write(f"- Residue-pair features: {n_features}\n")
        fh.write(f"- Phosphate selection: `{args.phosphate_sel}`\n")
        fh.write(f"- Residue representation: `{args.residue_repr}`\n")
        fh.write("\n## Explained Variance\n\n")
        for i, value in enumerate(pca.explained_variance_ratio_, start=1):
            fh.write(f"- PC{i}: {float(value):.6f}\n")
        fh.write("\n## PC Definitions\n\n")
        for comp in component_summaries:
            fh.write(f"### {comp['component']}\n\n")
            fh.write(comp["definition"] + "\n\n")
            fh.write("| Pair | Loading |\n")
            fh.write("|---|---:|\n")
            for rec in comp["top_pairs"][:6]:
                fh.write(f"| {rec['pair_label']} | {rec['loading']:.6f} |\n")
            fh.write("\n")

    run_utils.cleanup_empty_dirs(run_dir)
    print(f"Saved phosphate-pathway PCA run: {run_dir}")


if __name__ == "__main__":
    main()
