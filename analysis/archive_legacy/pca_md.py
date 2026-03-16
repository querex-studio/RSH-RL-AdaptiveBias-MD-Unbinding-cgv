import argparse
import csv
import glob
import os
import random
import sys
import importlib
import json
import re

import numpy as np
import matplotlib.pyplot as plt

try:
    import MDAnalysis as mda
    from MDAnalysis.transformations import fit_rot_trans
    from MDAnalysis.lib.distances import distance_array
except Exception as exc:
    raise ImportError("MDAnalysis is required for PCA on MD trajectories.") from exc

try:
    from sklearn.decomposition import IncrementalPCA
except Exception as exc:
    raise ImportError("scikit-learn is required for IncrementalPCA.") from exc

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from analysis import run_utils

KB_KCAL = 0.00198720425864083  # kcal/mol/K


_DCD_STEP_RE = re.compile(r"ep(\d+)_s(\d+)", re.IGNORECASE)


def _abs_path(path):
    if path is None:
        return None
    if os.path.isabs(path):
        return path
    return os.path.join(ROOT_DIR, path)


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


def _pairwise_indices(n_atoms):
    return np.triu_indices(n_atoms, k=1)


def _feature_dim(n_atoms, mode):
    if mode == "pairwise":
        return int(n_atoms * (n_atoms - 1) / 2)
    return int(n_atoms * 3)


def _pairwise_features(pos, box=None, pair_idx=None):
    if pair_idx is None:
        pair_idx = _pairwise_indices(pos.shape[0])
    dmat = distance_array(pos, pos, box=box)
    return dmat[pair_idx]


def _feature_vector(u, sel, mode, pair_idx=None):
    if mode == "pairwise":
        return _pairwise_features(sel.positions.astype(np.float64), box=u.dimensions, pair_idx=pair_idx)
    return sel.positions.astype(np.float64).ravel()


def _compute_mean(u, sel, stride, mode, pair_idx=None):
    n_feat = _feature_dim(sel.n_atoms, mode)
    mean_vec = np.zeros(n_feat, dtype=np.float64)
    count = 0
    for i, ts in enumerate(u.trajectory):
        if stride > 1 and (i % stride) != 0:
            continue
        mean_vec += _feature_vector(u, sel, mode, pair_idx=pair_idx)
        count += 1
    if count == 0:
        raise RuntimeError("No frames found for mean calculation.")
    mean_vec /= float(count)
    return mean_vec, count


def _iter_batches(u, sel, mean_vec, stride, batch_size, mode, pair_idx=None):
    batch = []
    for i, ts in enumerate(u.trajectory):
        if stride > 1 and (i % stride) != 0:
            continue
        x = _feature_vector(u, sel, mode, pair_idx=pair_idx) - mean_vec
        batch.append(x)
        if len(batch) >= batch_size:
            yield np.asarray(batch)
            batch = []
    if batch:
        yield np.asarray(batch)


def _distance_A(pos, i, j):
    return float(np.linalg.norm(pos[i] - pos[j]))


def _compute_time_axis(n_frames, dt_ps):
    return np.arange(n_frames, dtype=np.float32) * (dt_ps / 1000.0)


def _plot_scatter_time(out_path, x, y, time_ps=None):
    plt.figure()
    if time_ps is None:
        time_ps = np.arange(len(x), dtype=np.float32)
    plt.scatter(x, y, c=time_ps, s=4, cmap="viridis")
    plt.colorbar(label="Time (ps)")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("PC1 vs PC2 (time colored)")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def _plot_fes_2d(out_path, fes, xedges, yedges, title):
    plt.figure()
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    plt.imshow(fes.T, origin="lower", extent=extent, cmap="coolwarm")
    plt.colorbar(label="F (relative)")
    plt.xlabel("PC1 (largest variance)")
    plt.ylabel("PC2 (2nd largest variance)")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def _plot_fes_with_traj(out_path, fes, xedges, yedges, x, y, title):
    plt.figure()
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    plt.imshow(fes.T, origin="lower", extent=extent, cmap="coolwarm")
    plt.colorbar(label="F (relative)")
    plt.scatter(x, y, s=3, alpha=0.35, c="black")
    plt.xlabel("PC1 (largest variance)")
    plt.ylabel("PC2 (2nd largest variance)")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def _compute_fes_2d(x, y, bins=60, kT=1.0, weights=None):
    hist, xedges, yedges = np.histogram2d(x, y, bins=bins, density=False, weights=weights)
    prob = hist / np.sum(hist)
    with np.errstate(divide="ignore", invalid="ignore"):
        fes = -kT * np.log(prob)
    fes[~np.isfinite(fes)] = np.nan
    if np.any(np.isfinite(fes)):
        fes = fes - np.nanmin(fes)
    return fes, hist, xedges, yedges


def _compute_fes_1d(x, bins=60, kT=1.0, weights=None):
    hist, edges = np.histogram(x, bins=bins, density=False, weights=weights)
    prob = hist / np.sum(hist)
    with np.errstate(divide="ignore", invalid="ignore"):
        fes = -kT * np.log(prob)
    fes[~np.isfinite(fes)] = np.nan
    if np.any(np.isfinite(fes)):
        fes = fes - np.nanmin(fes)
    centers = 0.5 * (edges[:-1] + edges[1:])
    return centers, fes, hist, edges


def _plot_fes_1d(out_path, centers, fes, title, xlabel):
    plt.figure()
    plt.plot(centers, fes)
    plt.xlabel(xlabel)
    plt.ylabel("F (relative)")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def _plot_pc_loadings(out_path, pca_sel, pc_vec, top_n=10, title="PC loadings"):
    load = pc_vec.reshape((-1, 3))
    mag = np.linalg.norm(load, axis=1)
    top_idx = np.argsort(mag)[-top_n:]
    labels = [
        f"{pca_sel.atoms[i].resname}{pca_sel.atoms[i].resid}-{pca_sel.atoms[i].name}"
        for i in top_idx
    ]
    vals = mag[top_idx]
    plt.figure(figsize=(10, 6))
    plt.barh(range(len(vals)), vals, align="center")
    plt.yticks(range(len(vals)), labels)
    plt.xlabel("Loading magnitude")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def _atom_label(atom):
    return f"{atom.resname}{atom.resid}-{atom.name}"


def _plot_pc_pair_loadings(out_path, pca_sel, pair_idx, pc_vec, top_n=10, title="PC pair loadings"):
    mag = np.abs(pc_vec)
    top_idx = np.argsort(mag)[-top_n:]
    labels = []
    vals = mag[top_idx]
    i_idx, j_idx = pair_idx
    for k in top_idx:
        i = int(i_idx[k])
        j = int(j_idx[k])
        labels.append(f"{_atom_label(pca_sel.atoms[i])} -- {_atom_label(pca_sel.atoms[j])}")
    plt.figure(figsize=(10, 6))
    plt.barh(range(len(vals)), vals, align="center")
    plt.yticks(range(len(vals)), labels)
    plt.xlabel("Loading magnitude")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def _downsample_atomgroup(sel, max_atoms, sample_mode, seed):
    if max_atoms is None or max_atoms <= 0 or sel.n_atoms <= max_atoms:
        return sel
    if sample_mode == "random":
        rng = random.Random(seed)
        idx = sorted(rng.sample(range(sel.n_atoms), max_atoms))
    else:
        idx = list(range(max_atoms))
    return sel[idx]


def _select_evenly(indices, max_count):
    if max_count is None or max_count <= 0 or len(indices) <= max_count:
        return indices
    pick = np.linspace(0, len(indices) - 1, max_count, dtype=int)
    return [indices[i] for i in pick]


def _load_config(module_name):
    try:
        return importlib.import_module(module_name)
    except Exception:
        return importlib.import_module("config")


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


def main():
    parser = argparse.ArgumentParser(description="PCA on PPO DCD trajectories (streaming).")
    parser.add_argument("--config-module", default=None, help="Config module to use (default: combined_2d or config).")
    parser.add_argument("--top", default=None, help="Topology file (PSF/PDB). Defaults to config.psf_file.")
    parser.add_argument("--traj-glob", default=None, help="Glob for DCD trajectories.")
    parser.add_argument("--max-traj", type=int, default=50, help="Max trajectories to include (default: 50).")
    parser.add_argument("--sample", choices=["first", "random"], default="first", help="Trajectory sampling mode.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling.")
    parser.add_argument("--stride", type=int, default=1, help="Frame stride for PCA (default: 1).")
    parser.add_argument("--align-sel", default="protein and backbone", help="MDAnalysis selection for alignment.")
    parser.add_argument("--pca-sel", default=None, help="MDAnalysis selection for PCA atoms.")
    parser.add_argument("--pca-radius", type=float, default=20.0, help="Radius for default PCA selection.")
    parser.add_argument("--feature-mode", choices=["xyz", "pairwise"], default="pairwise",
                        help="PCA feature mode: pairwise (default) or xyz coordinates.")
    parser.add_argument("--pairwise-sel", default="protein and name CA",
                        help="MDAnalysis selection for pairwise features (default: protein CA).")
    parser.add_argument("--pairwise-max-atoms", type=int, default=120,
                        help="Max atoms for pairwise mode (downsample if exceeded).")
    parser.add_argument("--pairwise-sample", choices=["first", "random"], default="first",
                        help="Downsample mode when pairwise selection exceeds max atoms.")
    parser.add_argument("--pairwise-seed", type=int, default=42, help="Seed for random pairwise downsampling.")
    parser.add_argument("--n-components", type=int, default=10, help="Number of PCA components.")
    parser.add_argument("--batch-size", type=int, default=2000, help="IncrementalPCA batch size.")
    parser.add_argument("--bins", type=int, default=60, help="Bins for PCA FES.")
    parser.add_argument("--kT", type=float, default=1.0, help="kT for PCA FES (relative units).")
    parser.add_argument("--run", default=None, help="Existing analysis run directory.")
    parser.add_argument("--runs-root", default=None, help="Root folder for analysis_runs.")
    parser.add_argument("--write-structures", action="store_true", help="Write PDBs for TS/endpoints/PC2 extremes.")
    parser.add_argument("--max-structures", type=int, default=20, help="Max PDBs per category.")
    parser.add_argument("--ts-center", type=float, default=None, help="Transition-state center (CV1, Å).")
    parser.add_argument("--ts-width", type=float, default=0.5, help="Transition-state half-width (Å).")
    parser.add_argument("--bound-max", type=float, default=None, help="Bound state max CV1 (Å).")
    parser.add_argument("--unbound-min", type=float, default=None, help="Unbound state min CV1 (Å).")
    parser.add_argument("--pc2-percentile", type=float, default=95.0, help="Percentile for PC2 high/low split.")
    parser.add_argument("--atom1", type=int, default=None, help="CV1 atom1 index override.")
    parser.add_argument("--atom2", type=int, default=None, help="CV1 atom2 index override.")
    parser.add_argument("--atom3", type=int, default=None, help="CV2 atom3 index override.")
    parser.add_argument("--atom4", type=int, default=None, help="CV2 atom4 index override.")
    parser.add_argument("--bias-reweight", action="store_true", help="Compute per-frame bias and weights for unbiasing.")
    parser.add_argument("--bias-meta-dir", default=None, help="Directory with episode_XXXX.json bias logs.")
    parser.add_argument("--reweight-temperature", type=float, default=None, help="Temperature in K for reweighting.")
    parser.add_argument("--reweight-shift", choices=["max", "none"], default="max",
                        help="Shift bias by max before exp(beta*Vbias) to improve numerical stability.")
    args = parser.parse_args()

    cfg_name = args.config_module or ("combined_2d" if os.path.exists(os.path.join(ROOT_DIR, "combined_2d.py")) else "config")
    cfg = _load_config(cfg_name)

    # Bias reweighting setup (optional)
    bias_meta_dir = args.bias_meta_dir
    if bias_meta_dir is None:
        bias_meta_dir = os.path.join(ROOT_DIR, "results_PPO", "episode_meta")
    if not os.path.isabs(bias_meta_dir):
        bias_meta_dir = os.path.join(ROOT_DIR, bias_meta_dir)

    def _parse_ep_step(path):
        if not path:
            return None
        name = os.path.basename(path)
        m = _DCD_STEP_RE.search(name)
        if not m:
            return None
        return int(m.group(1)), int(m.group(2))

    def _load_bias_log(ep_idx):
        meta_path = os.path.join(bias_meta_dir, f"episode_{ep_idx:04d}.json")
        if not os.path.exists(meta_path):
            return None
        with open(meta_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        raw = payload.get("bias_log", [])
        entries = {"2d": [], "cv1": [], "cv2": []}
        for row in raw:
            if len(row) >= 7:
                step, kind, amp_kcal, c1_A, c2_A, sx_A, sy_A = row[:7]
                if str(kind) in ("gaussian2d", "bias2d"):
                    entries["2d"].append(
                        (int(step), float(amp_kcal), float(c1_A), float(c2_A), float(sx_A), float(sy_A))
                    )
                continue
            if len(row) < 6:
                continue
            step, cv_id, kind, amp_kcal, center_A, width_A = row[:6]
            if kind not in ("gaussian", "bias", "bias1"):
                continue
            if int(cv_id) == 1:
                entries["cv1"].append((int(step), float(amp_kcal), float(center_A), float(width_A)))
            elif int(cv_id) == 2:
                entries["cv2"].append((int(step), float(amp_kcal), float(center_A), float(width_A)))
        return entries

    bias_log_cache = {}
    bias_terms_cache = {}

    def _get_bias_terms(ep_idx, step_idx):
        key = (ep_idx, step_idx)
        if key in bias_terms_cache:
            return bias_terms_cache[key]
        if ep_idx not in bias_log_cache:
            bias_log_cache[ep_idx] = _load_bias_log(ep_idx)
        entries = bias_log_cache.get(ep_idx)
        if not entries:
            bias_terms_cache[key] = ([], [], [])
            return bias_terms_cache[key]
        terms_2d = []
        terms_cv1 = []
        terms_cv2 = []
        for st, amp_kcal, c1_A, c2_A, sx_A, sy_A in entries.get("2d", []):
            if st > step_idx:
                continue
            terms_2d.append((amp_kcal, c1_A, c2_A, sx_A, sy_A))
        for st, amp_kcal, center_A, width_A in entries.get("cv1", []):
            if st > step_idx:
                continue
            terms_cv1.append((amp_kcal, center_A, width_A))
        for st, amp_kcal, center_A, width_A in entries.get("cv2", []):
            if st > step_idx:
                continue
            terms_cv2.append((amp_kcal, center_A, width_A))
        bias_terms_cache[key] = (terms_2d, terms_cv1, terms_cv2)
        return bias_terms_cache[key]

    def _bias_value(cv1_A, cv2_A, ep_idx, step_idx):
        terms_2d, terms_cv1, terms_cv2 = _get_bias_terms(ep_idx, step_idx)
        vb = 0.0
        if terms_2d:
            for amp, c1, c2, sx, sy in terms_2d:
                sx = max(1e-6, float(sx))
                sy = max(1e-6, float(sy))
                vb += float(amp) * np.exp(-((float(cv1_A) - float(c1)) ** 2) / (2.0 * sx * sx)
                                          -((float(cv2_A) - float(c2)) ** 2) / (2.0 * sy * sy))
        if terms_cv1:
            for amp, center, width in terms_cv1:
                w = max(1e-6, float(width))
                vb += float(amp) * np.exp(-((float(cv1_A) - float(center)) ** 2) / (2.0 * w * w))
        if terms_cv2:
            for amp, center, width in terms_cv2:
                w = max(1e-6, float(width))
                vb += float(amp) * np.exp(-((float(cv2_A) - float(center)) ** 2) / (2.0 * w * w))
        return float(vb)

    feature_mode = args.feature_mode
    if feature_mode == "pairwise":
        pca_sel_str = args.pairwise_sel
    else:
        if args.pca_sel:
            pca_sel_str = args.pca_sel
        else:
            mg_idx = int(args.atom2 if args.atom2 is not None else getattr(cfg, "ATOM2_INDEX", 0))
            pca_sel_str = f"protein and not name H* and around {args.pca_radius} index {mg_idx}"

    top = _abs_path(args.top or getattr(cfg, "psf_file", None))
    if top is None or not os.path.exists(top):
        raise FileNotFoundError(f"Topology not found: {top}")

    traj_glob = args.traj_glob or _resolve_default_traj_glob(cfg)
    traj_list = sorted(glob.glob(traj_glob))
    if not traj_list:
        raise FileNotFoundError(f"No DCD files found for pattern: {traj_glob}")

    traj_list = _select_trajs(traj_list, args.max_traj, args.sample, args.seed)

    # Resolve analysis run directory
    run_dir = args.run
    if run_dir is not None and not os.path.isabs(run_dir):
        run_dir = os.path.join(ROOT_DIR, run_dir)
    if run_dir is None:
        time_tag = run_utils.default_time_tag()
        run_dir = run_utils.prepare_run_dir(time_tag, root=args.runs_root)
        run_utils.write_run_metadata(
            run_dir,
            {
                "script": "analysis/pca_md.py",
                "traj_glob": traj_glob,
                "traj_count": len(traj_list),
                "config_module": cfg_name,
                "feature_mode": feature_mode,
                "pca_sel": pca_sel_str,
                "pairwise_max_atoms": args.pairwise_max_atoms if feature_mode == "pairwise" else None,
                "config": _snapshot_cfg(cfg),
            },
        )
    else:
        for sub in ["data", os.path.join("figs", "analysis"), "structures"]:
            os.makedirs(os.path.join(run_dir, sub), exist_ok=True)

    # Open universe
    u = mda.Universe(top, traj_list)

    # Selections
    align_sel = u.select_atoms(args.align_sel) if args.align_sel else None
    pca_sel = u.select_atoms(pca_sel_str)
    if feature_mode == "pairwise":
        pca_sel = _downsample_atomgroup(pca_sel, args.pairwise_max_atoms, args.pairwise_sample, args.pairwise_seed)

    if align_sel is not None and align_sel.n_atoms == 0:
        raise ValueError("Alignment selection is empty.")
    if pca_sel.n_atoms == 0:
        raise ValueError("PCA selection is empty.")
    if feature_mode == "pairwise" and pca_sel.n_atoms < 2:
        raise ValueError("Pairwise feature mode requires at least two atoms in selection.")

    # Apply alignment transformation
    if align_sel is not None:
        ref = mda.Universe(top, traj_list)
        ref.trajectory[0]
        ref_atoms = ref.select_atoms(args.align_sel)
        mob_atoms = u.select_atoms(args.align_sel)
        u.trajectory.add_transformations(fit_rot_trans(mob_atoms, ref_atoms))

    pair_idx = None
    if feature_mode == "pairwise":
        pair_idx = _pairwise_indices(pca_sel.n_atoms)

    # Mean vector
    mean_vec, n_frames = _compute_mean(u, pca_sel, args.stride, feature_mode, pair_idx=pair_idx)

    # Fit incremental PCA
    pca = IncrementalPCA(n_components=args.n_components, batch_size=args.batch_size)
    u.trajectory.rewind()
    for batch in _iter_batches(u, pca_sel, mean_vec, args.stride, args.batch_size, feature_mode, pair_idx=pair_idx):
        pca.partial_fit(batch)

    explained = pca.explained_variance_ratio_

    # Transform pass: collect scores and CV distances
    atom1 = int(args.atom1 if args.atom1 is not None else getattr(cfg, "ATOM1_INDEX", 0))
    atom2 = int(args.atom2 if args.atom2 is not None else getattr(cfg, "ATOM2_INDEX", 0))
    atom3 = int(args.atom3 if args.atom3 is not None else getattr(cfg, "ATOM3_INDEX", 0))
    atom4 = int(args.atom4 if args.atom4 is not None else getattr(cfg, "ATOM4_INDEX", 0))

    scores_list = []
    cv1_list = []
    cv2_list = []
    frame_meta = []
    bias_list = []
    bias_ep_list = []
    bias_step_list = []
    bias_missing_list = []

    u.trajectory.rewind()
    batch = []
    batch_meta = []
    batch_cv1 = []
    batch_cv2 = []
    for i, ts in enumerate(u.trajectory):
        if args.stride > 1 and (i % args.stride) != 0:
            continue
        x = _feature_vector(u, pca_sel, feature_mode, pair_idx=pair_idx) - mean_vec
        batch.append(x)
        batch_meta.append(i)
        pos = u.atoms.positions.astype(np.float64)
        d1 = _distance_A(pos, atom1, atom2)
        d2 = _distance_A(pos, atom3, atom4)
        batch_cv1.append(d1)
        batch_cv2.append(d2)

        if args.bias_reweight:
            ts_file = getattr(ts, "filename", None) or getattr(getattr(u, "trajectory", None), "filename", None)
            ep_step = _parse_ep_step(ts_file)
            if ep_step is None:
                bias_list.append(0.0)
                bias_ep_list.append(-1)
                bias_step_list.append(-1)
                bias_missing_list.append(True)
            else:
                ep_idx, step_idx = ep_step
                if ep_idx not in bias_log_cache:
                    bias_log_cache[ep_idx] = _load_bias_log(ep_idx)
                missing = bias_log_cache.get(ep_idx) is None
                vb = _bias_value(d1, d2, ep_idx, step_idx)
                bias_list.append(vb)
                bias_ep_list.append(ep_idx)
                bias_step_list.append(step_idx)
                bias_missing_list.append(missing)

        if len(batch) >= args.batch_size:
            Yb = pca.transform(np.asarray(batch))
            for row, fi, d1, d2 in zip(Yb, batch_meta, batch_cv1, batch_cv2):
                scores_list.append(row)
                cv1_list.append(d1)
                cv2_list.append(d2)
                frame_meta.append(fi)
            batch = []
            batch_meta = []
            batch_cv1 = []
            batch_cv2 = []

    if batch:
        Yb = pca.transform(np.asarray(batch))
        for row, fi, d1, d2 in zip(Yb, batch_meta, batch_cv1, batch_cv2):
            scores_list.append(row)
            cv1_list.append(d1)
            cv2_list.append(d2)
            frame_meta.append(fi)

    scores = np.asarray(scores_list, dtype=np.float32)
    cv1 = np.asarray(cv1_list, dtype=np.float32)
    cv2 = np.asarray(cv2_list, dtype=np.float32)
    frame_meta = np.asarray(frame_meta, dtype=np.int64)

    # Save raw arrays
    data_dir = os.path.join(run_dir, "data")
    os.makedirs(data_dir, exist_ok=True)
    np.save(os.path.join(data_dir, "pca_mean.npy"), mean_vec)
    np.save(os.path.join(data_dir, "pca_evals.npy"), pca.explained_variance_)
    np.save(os.path.join(data_dir, "pca_evecs.npy"), pca.components_.T)
    np.save(os.path.join(data_dir, "pca_scores.npy"), scores)
    np.save(os.path.join(data_dir, "pca_explained.npy"), explained)
    np.save(os.path.join(data_dir, "cv1.npy"), cv1)
    np.save(os.path.join(data_dir, "cv2.npy"), cv2)
    np.save(os.path.join(data_dir, "frame_indices.npy"), frame_meta)

    # Bias reweighting outputs (optional)
    weights = None
    if args.bias_reweight:
        if len(bias_list) == scores.shape[0]:
            bias_kcal = np.asarray(bias_list, dtype=np.float32)
            bias_ep = np.asarray(bias_ep_list, dtype=np.int32)
            bias_step = np.asarray(bias_step_list, dtype=np.int32)
            bias_missing = np.asarray(bias_missing_list, dtype=bool)

            # Determine temperature
            temp_K = None
            unit_mod = None
            if hasattr(cfg, "unit"):
                unit_mod = cfg.unit
            elif hasattr(cfg, "u"):
                unit_mod = cfg.u
            if args.reweight_temperature is not None:
                temp_K = float(args.reweight_temperature)
            elif hasattr(cfg, "T"):
                try:
                    if unit_mod is not None and hasattr(cfg.T, "value_in_unit"):
                        temp_K = float(cfg.T.value_in_unit(unit_mod.kelvin))
                    else:
                        temp_K = float(cfg.T)
                except Exception:
                    temp_K = None
            if temp_K is None:
                temp_K = 300.0

            beta = 1.0 / (KB_KCAL * float(temp_K))
            if args.reweight_shift == "max":
                shift = float(np.nanmax(bias_kcal)) if bias_kcal.size > 0 else 0.0
            else:
                shift = 0.0
            weights = np.exp(beta * (bias_kcal - shift))

            np.save(os.path.join(data_dir, "bias_kcal.npy"), bias_kcal)
            np.save(os.path.join(data_dir, "bias_weights.npy"), weights)
            np.save(os.path.join(data_dir, "bias_episode.npy"), bias_ep)
            np.save(os.path.join(data_dir, "bias_step.npy"), bias_step)
            np.save(os.path.join(data_dir, "bias_missing.npy"), bias_missing)

            meta = {
                "bias_meta_dir": bias_meta_dir,
                "temp_K": float(temp_K),
                "beta_kcal": float(beta),
                "bias_shift_kcal": float(shift),
                "missing_frames": int(np.sum(bias_missing)),
            }
            with open(os.path.join(data_dir, "bias_weights_meta.json"), "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2)
            if meta["missing_frames"] > 0:
                print(f"[bias_reweight] Missing bias logs for {meta['missing_frames']} frames; weights defaulted to exp(beta*0).")
        else:
            print("[bias_reweight] Bias list length does not match frames; skipping weights.")

    # Summary CSV of scores
    scores_csv = os.path.join(data_dir, "pca_scores.csv")
    with open(scores_csv, "w", newline="") as f:
        w = csv.writer(f)
        header = ["frame_idx", "pc1", "pc2", "cv1_A", "cv2_A"]
        w.writerow(header)
        for i in range(scores.shape[0]):
            w.writerow([int(frame_meta[i]), float(scores[i, 0]), float(scores[i, 1]), float(cv1[i]), float(cv2[i])])

    # PCA plots
    fig_dir = os.path.join(run_dir, "figs", "analysis")
    os.makedirs(fig_dir, exist_ok=True)

    plt.figure()
    plt.plot(np.arange(1, len(explained) + 1), explained, marker="o")
    plt.xlabel("Component")
    plt.ylabel("Explained variance ratio")
    plt.title("PCA scree plot")
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "pca_scree.png"))
    plt.close()

    plt.figure()
    plt.plot(np.arange(1, len(explained) + 1), np.cumsum(explained), marker="o")
    plt.xlabel("Component")
    plt.ylabel("Cumulative explained variance")
    plt.title("PCA cumulative variance")
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "pca_cumulative.png"))
    plt.close()

    cv1_label = getattr(cfg, "CV1_LABEL", "CV1 distance")
    cv2_label = getattr(cfg, "CV2_LABEL", "CV2 distance")

    if scores.shape[1] >= 2:
        plt.figure()
        plt.scatter(scores[:, 0], scores[:, 1], s=4, alpha=0.6)
        plt.xlabel("PC1 (largest variance)")
        plt.ylabel("PC2 (2nd largest variance)")
        plt.title("PC1 vs PC2")
        plt.tight_layout()
        plt.savefig(os.path.join(fig_dir, "pca_pc1_pc2.png"))
        plt.close()
        time_ps = None
        unit_mod = None
        if hasattr(cfg, "unit"):
            unit_mod = cfg.unit
        elif hasattr(cfg, "u"):
            unit_mod = cfg.u
        if unit_mod is not None and hasattr(cfg, "stepsize"):
            report_interval = int(getattr(cfg, "DCD_REPORT_INTERVAL", getattr(cfg, "dcdfreq_mfpt", 1)))
            dt_ps = float(cfg.stepsize.value_in_unit(unit_mod.picoseconds)) * report_interval * args.stride
            time_ps = np.arange(scores.shape[0], dtype=np.float32) * dt_ps

        _plot_scatter_time(
            os.path.join(fig_dir, "pca_pc1_pc2_time.png"),
            scores[:, 0],
            scores[:, 1],
            time_ps=time_ps,
        )

    # Correlations
    corr_cv1_pc1 = float(np.corrcoef(cv1, scores[:, 0])[0, 1]) if scores.shape[1] >= 1 else 0.0
    corr_cv1_pc2 = float(np.corrcoef(cv1, scores[:, 1])[0, 1]) if scores.shape[1] >= 2 else 0.0
    corr_cv2_pc1 = float(np.corrcoef(cv2, scores[:, 0])[0, 1]) if scores.shape[1] >= 1 else 0.0
    corr_cv2_pc2 = float(np.corrcoef(cv2, scores[:, 1])[0, 1]) if scores.shape[1] >= 2 else 0.0

    plt.figure()
    plt.scatter(cv1, scores[:, 0], s=6, alpha=0.4)
    plt.xlabel(f"{cv1_label} (Å)")
    plt.ylabel("PC1 (largest variance)")
    plt.title(f"CV1 vs PC1 (corr={corr_cv1_pc1:.3f})")
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "corr_cv1_pc1.png"))
    plt.close()

    if scores.shape[1] >= 2:
        plt.figure()
        plt.scatter(cv1, scores[:, 1], s=6, alpha=0.4)
        plt.xlabel(f"{cv1_label} (Å)")
        plt.ylabel("PC2 (2nd largest variance)")
        plt.title(f"CV1 vs PC2 (corr={corr_cv1_pc2:.3f})")
        plt.tight_layout()
        plt.savefig(os.path.join(fig_dir, "corr_cv1_pc2.png"))
        plt.close()

        plt.figure()
        plt.scatter(cv2, scores[:, 1], s=6, alpha=0.4)
        plt.xlabel(f"{cv2_label} (Å)")
        plt.ylabel("PC2 (2nd largest variance)")
        plt.title(f"CV2 vs PC2 (corr={corr_cv2_pc2:.3f})")
        plt.tight_layout()
        plt.savefig(os.path.join(fig_dir, "corr_cv2_pc2.png"))
        plt.close()

    # PCA FES
    if scores.shape[1] >= 2:
        fes2d, hist2d, xedges, yedges = _compute_fes_2d(scores[:, 0], scores[:, 1], bins=args.bins, kT=args.kT)
        np.save(os.path.join(data_dir, "pca_fes.npy"), fes2d)
        np.save(os.path.join(data_dir, "pca_fes_hist.npy"), hist2d)
        np.save(os.path.join(data_dir, "pca_fes_xedges.npy"), xedges)
        np.save(os.path.join(data_dir, "pca_fes_yedges.npy"), yedges)
        _plot_fes_2d(os.path.join(fig_dir, "pca_fes.png"), fes2d, xedges, yedges, "PCA FES (PC1-PC2)")
        _plot_fes_with_traj(
            os.path.join(fig_dir, "pca_fes_with_traj.png"),
            fes2d,
            xedges,
            yedges,
            scores[:, 0],
            scores[:, 1],
            "PCA FES with trajectory",
        )

        pc1_centers, pc1_fes, pc1_hist, pc1_edges = _compute_fes_1d(scores[:, 0], bins=args.bins, kT=args.kT)
        pc2_centers, pc2_fes, pc2_hist, pc2_edges = _compute_fes_1d(scores[:, 1], bins=args.bins, kT=args.kT)
        np.save(os.path.join(data_dir, "pca_pc1_centers.npy"), pc1_centers)
        np.save(os.path.join(data_dir, "pca_pc2_centers.npy"), pc2_centers)
        np.save(os.path.join(data_dir, "pca_pc1_fes.npy"), pc1_fes)
        np.save(os.path.join(data_dir, "pca_pc2_fes.npy"), pc2_fes)
        _plot_fes_1d(os.path.join(fig_dir, "pca_pc1_fes.png"), pc1_centers, pc1_fes, "PC1 free energy", "PC1")
        _plot_fes_1d(os.path.join(fig_dir, "pca_pc2_fes.png"), pc2_centers, pc2_fes, "PC2 free energy", "PC2")

        # Unbiased PCA FES (if weights were computed)
        if weights is not None and len(weights) == scores.shape[0]:
            fes2d_w, hist2d_w, xedges_w, yedges_w = _compute_fes_2d(
                scores[:, 0], scores[:, 1], bins=args.bins, kT=args.kT, weights=weights
            )
            np.save(os.path.join(data_dir, "pca_fes_unbiased.npy"), fes2d_w)
            np.save(os.path.join(data_dir, "pca_fes_unbiased_hist.npy"), hist2d_w)
            np.save(os.path.join(data_dir, "pca_fes_unbiased_xedges.npy"), xedges_w)
            np.save(os.path.join(data_dir, "pca_fes_unbiased_yedges.npy"), yedges_w)
            _plot_fes_2d(
                os.path.join(fig_dir, "pca_fes_unbiased.png"),
                fes2d_w,
                xedges_w,
                yedges_w,
                "PCA FES (unbiased, PC1-PC2)",
            )
            _plot_fes_with_traj(
                os.path.join(fig_dir, "pca_fes_unbiased_with_traj.png"),
                fes2d_w,
                xedges_w,
                yedges_w,
                scores[:, 0],
                scores[:, 1],
                "PCA FES (unbiased) with trajectory",
            )

            pc1_centers_w, pc1_fes_w, pc1_hist_w, pc1_edges_w = _compute_fes_1d(
                scores[:, 0], bins=args.bins, kT=args.kT, weights=weights
            )
            pc2_centers_w, pc2_fes_w, pc2_hist_w, pc2_edges_w = _compute_fes_1d(
                scores[:, 1], bins=args.bins, kT=args.kT, weights=weights
            )
            np.save(os.path.join(data_dir, "pca_pc1_centers_unbiased.npy"), pc1_centers_w)
            np.save(os.path.join(data_dir, "pca_pc2_centers_unbiased.npy"), pc2_centers_w)
            np.save(os.path.join(data_dir, "pca_pc1_fes_unbiased.npy"), pc1_fes_w)
            np.save(os.path.join(data_dir, "pca_pc2_fes_unbiased.npy"), pc2_fes_w)
            _plot_fes_1d(
                os.path.join(fig_dir, "pca_pc1_fes_unbiased.png"),
                pc1_centers_w,
                pc1_fes_w,
                "PC1 free energy (unbiased)",
                "PC1",
            )
            _plot_fes_1d(
                os.path.join(fig_dir, "pca_pc2_fes_unbiased.png"),
                pc2_centers_w,
                pc2_fes_w,
                "PC2 free energy (unbiased)",
                "PC2",
            )

    # Loadings plots
    if scores.shape[1] >= 2:
        if feature_mode == "pairwise":
            _plot_pc_pair_loadings(
                os.path.join(fig_dir, "pc1_pair_loadings.png"),
                pca_sel,
                pair_idx,
                pca.components_[0],
                top_n=10,
                title="Top PC1 pair loadings",
            )
            _plot_pc_pair_loadings(
                os.path.join(fig_dir, "pc2_pair_loadings.png"),
                pca_sel,
                pair_idx,
                pca.components_[1],
                top_n=10,
                title="Top PC2 pair loadings",
            )
        else:
            _plot_pc_loadings(os.path.join(fig_dir, "pc1_loadings.png"), pca_sel, pca.components_[0], top_n=10, title="Top PC1 loadings")
            _plot_pc_loadings(os.path.join(fig_dir, "pc2_loadings.png"), pca_sel, pca.components_[1], top_n=10, title="Top PC2 loadings")

    # Structure selection defaults (used for report + optional PDB output)
    ts_center_val = args.ts_center
    if ts_center_val is None:
        try:
            ts_center_val = 0.5 * (float(cfg.CURRENT_DISTANCE) + float(cfg.FINAL_TARGET))
        except Exception:
            ts_center_val = float(np.median(cv1))
    bound_max_val = args.bound_max
    if bound_max_val is None:
        try:
            bound_max_val = float(cfg.CURRENT_DISTANCE) + 0.5
        except Exception:
            bound_max_val = float(np.percentile(cv1, 20))
    unbound_min_val = args.unbound_min
    if unbound_min_val is None:
        try:
            unbound_min_val = float(cfg.FINAL_TARGET) - 0.5
        except Exception:
            unbound_min_val = float(np.percentile(cv1, 80))

    # Interpretation report
    def _strength_label(val):
        aval = abs(val)
        if aval >= 0.7:
            return "strong"
        if aval >= 0.4:
            return "moderate"
        if aval >= 0.2:
            return "weak"
        return "very weak"

    pc1_rep = "PC1"
    pc2_rep = "PC2"
    if abs(corr_cv1_pc1) >= abs(corr_cv1_pc2):
        pc1_rep = f"PC1 likely aligns with CV1 (corr={corr_cv1_pc1:.3f}, {_strength_label(corr_cv1_pc1)})"
        pc2_rep = f"PC2 may encode alternate pathways / orthogonal motion (corr with CV1={corr_cv1_pc2:.3f})"
    else:
        pc1_rep = f"PC1 likely aligns with CV2 (corr={corr_cv2_pc1:.3f}, {_strength_label(corr_cv2_pc1)})"
        pc2_rep = f"PC2 may align with CV1 (corr={corr_cv1_pc2:.3f}, {_strength_label(corr_cv1_pc2)})"

    report_path = os.path.join(data_dir, "pca_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# PCA Report (MD trajectories)\n\n")
        f.write(f"- Topology: `{top}`\n")
        f.write(f"- DCD count: {len(traj_list)}\n")
        f.write(f"- Frames used (after stride): {scores.shape[0]}\n")
        f.write(f"- Feature mode: {feature_mode}\n")
        f.write(f"- PCA selection: `{pca_sel_str}`\n")
        f.write(f"- PCA selection atoms: {int(pca_sel.n_atoms)}\n")
        f.write(f"- CV1: {cv1_label} (atoms {atom1}, {atom2})\n")
        f.write(f"- CV2: {cv2_label} (atoms {atom3}, {atom4})\n")
        f.write("\n## Explained variance\n\n")
        for i, val in enumerate(explained, start=1):
            f.write(f"- PC{i}: {val:.6f}\n")
        f.write("\n## Correlations\n\n")
        f.write(f"- corr(CV1, PC1): {corr_cv1_pc1:.6f}\n")
        f.write(f"- corr(CV1, PC2): {corr_cv1_pc2:.6f}\n")
        f.write(f"- corr(CV2, PC1): {corr_cv2_pc1:.6f}\n")
        f.write(f"- corr(CV2, PC2): {corr_cv2_pc2:.6f}\n")
        f.write("\n## Interpretation\n\n")
        f.write(f"- {pc1_rep}\n")
        f.write(f"- {pc2_rep}\n")
        f.write("\n## Structure selection defaults\n\n")
        f.write(f"- TS center (CV1): {ts_center_val:.3f}\n")
        f.write(f"- TS half-width (CV1): {args.ts_width}\n")
        f.write(f"- Bound max (CV1): {bound_max_val:.3f}\n")
        f.write(f"- Unbound min (CV1): {unbound_min_val:.3f}\n")
        f.write(f"- PC2 percentile split: {args.pc2_percentile}\n")

    # Write top-loading atoms/pairs CSV
    if feature_mode == "pairwise":
        load_csv = os.path.join(data_dir, "pca_top_pairs.csv")
        with open(load_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["pc", "rank", "pair_idx", "atom_i_idx", "atom_j_idx", "atom_i", "atom_j", "loading_mag"])
            i_idx, j_idx = pair_idx
            for pc_i in [0, 1]:
                if pc_i >= pca.components_.shape[0]:
                    continue
                mag = np.abs(pca.components_[pc_i])
                top_idx = np.argsort(mag)[-20:][::-1]
                for rank, idx in enumerate(top_idx, 1):
                    ai = int(i_idx[idx])
                    aj = int(j_idx[idx])
                    a_i = pca_sel.atoms[ai]
                    a_j = pca_sel.atoms[aj]
                    w.writerow([
                        pc_i + 1,
                        rank,
                        int(idx),
                        int(a_i.index),
                        int(a_j.index),
                        _atom_label(a_i),
                        _atom_label(a_j),
                        float(mag[idx]),
                    ])
    else:
        load_csv = os.path.join(data_dir, "pca_top_atoms.csv")
        with open(load_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["pc", "rank", "pca_sel_idx", "global_atom_index", "resid", "resname", "name", "loading_mag"])
            for pc_i in [0, 1]:
                if pc_i >= pca.components_.shape[0]:
                    continue
                load = pca.components_[pc_i].reshape((-1, 3))
                mag = np.linalg.norm(load, axis=1)
                top_idx = np.argsort(mag)[-20:][::-1]
                for rank, idx in enumerate(top_idx, 1):
                    a = pca_sel.atoms[idx]
                    w.writerow([pc_i + 1, rank, int(idx), int(a.index), int(a.resid), a.resname, a.name, float(mag[idx])])

    # Structures: transition state + endpoints + PC2 extremes
    if args.write_structures and scores.shape[1] >= 2:
        ts_min = ts_center_val - float(args.ts_width)
        ts_max = ts_center_val + float(args.ts_width)

        ts_idx = np.where((cv1 >= ts_min) & (cv1 <= ts_max))[0].tolist()
        bound_idx = np.where(cv1 <= bound_max_val)[0].tolist()
        unbound_idx = np.where(cv1 >= unbound_min_val)[0].tolist()

        perc = float(args.pc2_percentile)
        hi = np.percentile(scores[:, 1], perc)
        lo = np.percentile(scores[:, 1], 100.0 - perc)
        pc2_hi_idx = np.where(scores[:, 1] >= hi)[0].tolist()
        pc2_lo_idx = np.where(scores[:, 1] <= lo)[0].tolist()

        ts_idx = _select_evenly(ts_idx, args.max_structures)
        bound_idx = _select_evenly(bound_idx, args.max_structures)
        unbound_idx = _select_evenly(unbound_idx, args.max_structures)
        pc2_hi_idx = _select_evenly(pc2_hi_idx, args.max_structures)
        pc2_lo_idx = _select_evenly(pc2_lo_idx, args.max_structures)

        select_map = {
            "transition_state": set(ts_idx),
            "bound": set(bound_idx),
            "unbound": set(unbound_idx),
            "pc2_high": set(pc2_hi_idx),
            "pc2_low": set(pc2_lo_idx),
        }

        out_root = os.path.join(run_dir, "structures")
        for key in select_map:
            os.makedirs(os.path.join(out_root, key), exist_ok=True)

        # Pass through trajectory to write PDBs
        u.trajectory.rewind()
        processed_index = -1
        written = {k: 0 for k in select_map}
        for i, ts in enumerate(u.trajectory):
            if args.stride > 1 and (i % args.stride) != 0:
                continue
            processed_index += 1
            for key, idx_set in select_map.items():
                if processed_index in idx_set:
                    out_path = os.path.join(out_root, key, f"{key}_frame_{processed_index:06d}.pdb")
                    u.atoms.write(out_path)
                    written[key] += 1
            if all(written[k] >= len(select_map[k]) for k in select_map):
                break

        # Summary CSV
        struct_csv = os.path.join(out_root, "structure_selection.csv")
        with open(struct_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["category", "frame_idx", "cv1_A", "cv2_A", "pc1", "pc2"])
            for key, idx_set in select_map.items():
                for idx in idx_set:
                    w.writerow([
                        key,
                        int(idx),
                        float(cv1[idx]),
                        float(cv2[idx]),
                        float(scores[idx, 0]),
                        float(scores[idx, 1]),
                    ])

    run_utils.cleanup_empty_dirs(run_dir)


if __name__ == "__main__":
    main()
