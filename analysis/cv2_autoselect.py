import argparse
import csv
import glob
import importlib
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

try:
    import MDAnalysis as mda
except Exception as exc:
    raise ImportError("MDAnalysis is required for CV2 auto-selection.") from exc

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from analysis import run_utils
from analysis.pca_phosphate_pathway import _abs_path, _corrcoef_safe, _distance_A, _resolve_default_traj_glob, _select_trajs


def _load_config(module_name):
    try:
        return importlib.import_module(module_name)
    except Exception:
        return importlib.import_module("combined_2d")


def _atom_label(atom):
    return f"{atom.ix}:{atom.segid}:{atom.resname}{atom.resid}:{atom.name}"


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


def _autocorr(values, lag):
    arr = np.asarray(values, dtype=np.float64)
    if len(arr) <= lag or np.allclose(np.std(arr), 0.0):
        return 0.0
    return abs(_corrcoef_safe(arr[:-lag], arr[lag:]))


def _coordination(mg_pos, oxy_pos, r0_A, power):
    d = np.linalg.norm(oxy_pos - mg_pos[None, :], axis=1)
    return float(np.sum(1.0 / (1.0 + np.power(np.maximum(d, 1e-6) / float(r0_A), int(power)))))


def _direction(start, final):
    return "increase" if float(final) > float(start) else "decrease"


def _find_o2(phosphate_oxygens):
    for atom in phosphate_oxygens:
        if atom.name.strip().upper() == "O2":
            return atom
    return None


def _candidate_definitions(u, args):
    mg = u.select_atoms(args.mg_sel)
    oxygens = u.select_atoms(args.phosphate_o_sel)
    if mg.n_atoms == 0:
        raise ValueError(f"Mg selection is empty: {args.mg_sel}")
    if oxygens.n_atoms == 0:
        raise ValueError(f"Phosphate oxygen selection is empty: {args.phosphate_o_sel}")
    if mg.n_atoms > 1:
        print(f"[warn] Mg selection returned {mg.n_atoms} atoms; using the first one: {_atom_label(mg[0])}")
    mg_atom = mg[0]

    candidates = []
    o2_atom = _find_o2(oxygens)
    if o2_atom is not None:
        candidates.append(
            {
                "name": "mg_phosphate_o2_distance",
                "kind": "distance",
                "priority": 1.0,
                "atom_i": int(mg_atom.ix),
                "atom_j": int(o2_atom.ix),
                "description": f"Mg to phosphate O2 distance ({_atom_label(mg_atom)} -- {_atom_label(o2_atom)})",
            }
        )
    for atom in oxygens:
        if o2_atom is not None and int(atom.ix) == int(o2_atom.ix):
            continue
        candidates.append(
            {
                "name": f"mg_phosphate_{atom.name.lower()}_distance",
                "kind": "distance",
                "priority": 0.90,
                "atom_i": int(mg_atom.ix),
                "atom_j": int(atom.ix),
                "description": f"Mg to phosphate oxygen distance ({_atom_label(mg_atom)} -- {_atom_label(atom)})",
            }
        )
    candidates.append(
        {
            "name": "mg_phosphate_oxygen_coordination",
            "kind": "coordination",
            "priority": 0.98,
            "atom_i": int(mg_atom.ix),
            "atom_group": [int(atom.ix) for atom in oxygens],
            "description": "Smooth Mg coordination number to phosphate oxygens",
        }
    )

    ligand = u.select_atoms(args.phosphate_sel)
    protein_atoms = u.select_atoms(args.pathway_atom_sel)
    if ligand.n_atoms and protein_atoms.n_atoms:
        ligand_center = ligand.center_of_mass()
        dists = np.linalg.norm(protein_atoms.positions - ligand_center[None, :], axis=1)
        order = np.argsort(dists)[: int(args.max_pathway_atoms)]
        p_atom = u.select_atoms(args.phosphate_p_sel)
        p_idx = int(p_atom[0].ix) if p_atom.n_atoms else None
        for local_idx in order:
            atom = protein_atoms[int(local_idx)]
            candidates.append(
                {
                    "name": f"mg_pathway_{atom.resname}{atom.resid}_{atom.name}_distance",
                    "kind": "distance",
                    "priority": 0.55,
                    "atom_i": int(mg_atom.ix),
                    "atom_j": int(atom.ix),
                    "description": f"Mg to nearby pathway atom ({_atom_label(mg_atom)} -- {_atom_label(atom)})",
                }
            )
            if p_idx is not None:
                candidates.append(
                    {
                        "name": f"phosphate_pathway_{atom.resname}{atom.resid}_{atom.name}_distance",
                        "kind": "distance",
                        "priority": 0.50,
                        "atom_i": p_idx,
                        "atom_j": int(atom.ix),
                        "description": f"Phosphate P to nearby pathway atom ({_atom_label(u.atoms[p_idx])} -- {_atom_label(atom)})",
                    }
                )
    return candidates


def _candidate_value(candidate, positions, args):
    if candidate["kind"] == "distance":
        return _distance_A(positions, candidate["atom_i"], candidate["atom_j"])
    if candidate["kind"] == "coordination":
        mg_pos = positions[int(candidate["atom_i"])]
        oxy_pos = positions[np.asarray(candidate["atom_group"], dtype=int)]
        return _coordination(mg_pos, oxy_pos, args.coord_r0, args.coord_power)
    raise ValueError(f"Unknown candidate kind: {candidate['kind']}")


def _collect_series(top_path, traj_list, candidates, cfg, args):
    series = {cand["name"]: [] for cand in candidates}
    cv1 = []
    frame_rows = []
    atom1 = int(getattr(cfg, "ATOM1_INDEX", 7799))
    atom2 = int(getattr(cfg, "ATOM2_INDEX", 7840))
    for traj_path in traj_list:
        u = mda.Universe(top_path, traj_path)
        traj_label = os.path.splitext(os.path.basename(traj_path))[0]
        for frame_idx, _ in enumerate(u.trajectory):
            if args.stride > 1 and (frame_idx % args.stride) != 0:
                continue
            positions = u.atoms.positions.astype(np.float64)
            cv1_val = _distance_A(positions, atom1, atom2)
            cv1.append(cv1_val)
            frame_rows.append({"traj": traj_label, "frame_idx": int(frame_idx), "cv1_A": float(cv1_val)})
            for cand in candidates:
                series[cand["name"]].append(_candidate_value(cand, positions, args))
    return series, np.asarray(cv1, dtype=np.float64), frame_rows


def _rank_candidates(candidates, series, cv1, args):
    stds = np.asarray([float(np.std(series[c["name"]])) for c in candidates], dtype=np.float64)
    net_changes = np.asarray([abs(float(series[c["name"]][-1]) - float(series[c["name"]][0])) for c in candidates], dtype=np.float64)
    std_score = _norm01(stds)
    net_score = _norm01(net_changes)

    rows = []
    for idx, cand in enumerate(candidates):
        values = np.asarray(series[cand["name"]], dtype=np.float64)
        corr_cv1 = _corrcoef_safe(values, cv1)
        autocorr = _autocorr(values, int(args.lag))
        redundancy_penalty = 1.0 - 0.25 * min(1.0, abs(corr_cv1) if np.isfinite(corr_cv1) else 0.0)
        score = (
            0.30 * std_score[idx]
            + 0.30 * autocorr
            + 0.20 * net_score[idx]
            + 0.20 * float(cand["priority"])
        ) * redundancy_penalty
        start = float(values[0])
        final = float(np.mean(values[-max(1, min(len(values), args.final_window)) :]))
        row = dict(cand)
        row.update(
            {
                "score": float(score),
                "start_value": start,
                "suggested_final_value": final,
                "direction": _direction(start, final),
                "std": float(stds[idx]),
                "net_change": float(net_changes[idx]),
                "autocorr_lag": float(autocorr),
                "corr_cv1": None if not np.isfinite(corr_cv1) else float(corr_cv1),
                "config_ready": cand["kind"] == "distance",
            }
        )
        rows.append(row)
    return sorted(rows, key=lambda row: row["score"], reverse=True)


def _write_outputs(out_dir, ranked, series, args):
    os.makedirs(os.path.join(out_dir, "data"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "figs", "analysis"), exist_ok=True)

    csv_path = os.path.join(out_dir, "data", "cv2_auto_selection_candidates.csv")
    fieldnames = [
        "rank", "score", "name", "kind", "config_ready", "atom_i", "atom_j",
        "atom_group", "start_value", "suggested_final_value", "direction",
        "std", "net_change", "autocorr_lag", "corr_cv1", "description",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for rank, row in enumerate(ranked, start=1):
            out = {key: row.get(key, "") for key in fieldnames}
            out["rank"] = rank
            if isinstance(out.get("atom_group"), list):
                out["atom_group"] = " ".join(map(str, out["atom_group"]))
            writer.writerow(out)

    json_path = os.path.join(out_dir, "data", "cv2_auto_selection_summary.json")
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump({"top_candidates": ranked[: int(args.top_candidates)], "all_candidates": ranked}, fh, indent=2)

    report_path = os.path.join(out_dir, "data", "cv2_auto_selection_report.md")
    best = ranked[0] if ranked else None
    with open(report_path, "w", encoding="utf-8") as fh:
        fh.write("# CV2 Auto-Selection Report\n\n")
        fh.write("This report ranks CV2 candidates from trajectory dynamics and chemical priors.\n\n")
        if best:
            fh.write("## Recommended Candidate\n\n")
            fh.write(f"- Name: `{best['name']}`\n")
            fh.write(f"- Type: `{best['kind']}`\n")
            fh.write(f"- Description: {best['description']}\n")
            fh.write(f"- Score: {float(best['score']):.6f}\n")
            fh.write(f"- Start value: {float(best['start_value']):.6f}\n")
            fh.write(f"- Suggested final value: {float(best['suggested_final_value']):.6f}\n")
            fh.write(f"- Direction: `{best['direction']}`\n\n")
            if best["kind"] == "distance":
                fh.write("Config snippet for current distance-based PPO CV2:\n\n")
                fh.write("```python\n")
                fh.write(f"ATOM3_INDEX = {int(best['atom_i'])}\n")
                fh.write(f"ATOM4_INDEX = {int(best['atom_j'])}\n")
                fh.write(f"CURRENT_DISTANCE2 = {float(best['start_value']):.3f}\n")
                fh.write(f"FINAL_TARGET2 = {float(best['suggested_final_value']):.3f}\n")
                fh.write(f'CV2_PROGRESS_DIRECTION = "{best["direction"]}"\n')
                fh.write("```\n\n")
            else:
                fh.write("This candidate is chemically strong but is not directly compatible with the current distance-only CV2 bias force.\n\n")
        fh.write("## Top Candidates\n\n")
        fh.write("| Rank | Score | Name | Type | Direction | Corr CV1 | Description |\n")
        fh.write("|---:|---:|---|---|---|---:|---|\n")
        for rank, row in enumerate(ranked[: int(args.top_candidates)], start=1):
            corr = "" if row["corr_cv1"] is None else f"{float(row['corr_cv1']):.3f}"
            fh.write(f"| {rank} | {float(row['score']):.4f} | `{row['name']}` | {row['kind']} | {row['direction']} | {corr} | {row['description']} |\n")

    fig_path = os.path.join(out_dir, "figs", "analysis", "cv2_auto_selection_top_timeseries.png")
    fig, ax = plt.subplots(figsize=(10.0, 5.4))
    max_len = 0
    for row in ranked[: min(5, len(ranked))]:
        vals = np.asarray(series[row["name"]], dtype=np.float64)
        max_len = max(max_len, int(vals.size))
        ax.plot(np.arange(vals.size), vals, marker="o", markersize=3.0, linewidth=1.2, label=row["name"])
    ax.set_xlabel("Processed frame")
    ax.set_ylabel("Candidate value (A for distance CVs)")
    ax.set_title("Top CV2 Candidate Time Series")
    if max_len <= 1:
        ax.text(
            0.5,
            0.5,
            "Only one processed frame was available.\nUse DCD trajectories for a true time series.",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=11,
            bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "#999999", "pad": 6.0},
        )
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(fig_path, dpi=220)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Auto-rank CV2 candidates for Mg-phosphate unbinding.")
    parser.add_argument("--config-module", default="combined_2d", help="Config module to use.")
    parser.add_argument("--top", default=None, help="Topology file; defaults to config.psf_file.")
    parser.add_argument("--traj-glob", default=None, help="DCD glob; defaults to config.RESULTS_TRAJ_DIR/*.dcd.")
    parser.add_argument("--max-traj", type=int, default=50, help="Maximum trajectories; <=0 means all.")
    parser.add_argument("--sample", choices=["first", "random"], default="first", help="Trajectory sampling mode.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for trajectory sampling.")
    parser.add_argument("--stride", type=int, default=1, help="Frame stride.")
    parser.add_argument("--lag", type=int, default=5, help="Autocorrelation lag in processed frames.")
    parser.add_argument("--mg-sel", default="resname MG or name MG", help="Mg atom selection.")
    parser.add_argument("--phosphate-sel", default="segid HETC and not name H*", help="Phosphate selection.")
    parser.add_argument("--phosphate-o-sel", default="segid HETC and name O*", help="Phosphate oxygen selection.")
    parser.add_argument("--phosphate-p-sel", default="segid HETC and name P", help="Phosphate P atom selection.")
    parser.add_argument("--pathway-atom-sel", default="protein and (name O* or name N*)", help="Nearby protein atom pool.")
    parser.add_argument("--max-pathway-atoms", type=int, default=20, help="Nearby protein atoms to include.")
    parser.add_argument("--coord-r0", type=float, default=2.4, help="Coordination switching length in A.")
    parser.add_argument("--coord-power", type=int, default=6, help="Coordination switching exponent.")
    parser.add_argument("--final-window", type=int, default=25, help="Frames used for suggested final value.")
    parser.add_argument("--top-candidates", type=int, default=12, help="Number of candidates to emphasize in reports.")
    parser.add_argument("--run", default=None, help="Existing output run directory.")
    parser.add_argument("--runs-root", default=None, help="Root folder for analysis_runs.")
    args = parser.parse_args()

    cfg = _load_config(args.config_module)
    top_path = _abs_path(args.top or getattr(cfg, "psf_file", None))
    if top_path is None or not os.path.exists(top_path):
        raise FileNotFoundError(f"Topology not found: {top_path}")
    traj_glob = args.traj_glob or _resolve_default_traj_glob(cfg)
    traj_list = sorted(glob.glob(traj_glob))
    if not traj_list:
        raise FileNotFoundError(f"No DCD files found for pattern: {traj_glob}")
    traj_list = _select_trajs(traj_list, args.max_traj, args.sample, args.seed)

    out_dir = args.run
    if out_dir is not None and not os.path.isabs(out_dir):
        out_dir = os.path.join(ROOT_DIR, out_dir)
    if out_dir is None:
        out_dir = run_utils.prepare_run_dir(run_utils.default_time_tag(), root=args.runs_root)
    os.makedirs(out_dir, exist_ok=True)

    ref = mda.Universe(top_path, traj_list[0])
    candidates = _candidate_definitions(ref, args)
    series, cv1, frame_rows = _collect_series(top_path, traj_list, candidates, cfg, args)
    if not frame_rows:
        raise RuntimeError("No trajectory frames were processed.")
    ranked = _rank_candidates(candidates, series, cv1, args)

    run_utils.write_run_metadata(
        out_dir,
        {
            "script": "analysis/cv2_autoselect.py",
            "config_module": args.config_module,
            "traj_glob": traj_glob,
            "traj_count": len(traj_list),
            "frame_count": len(frame_rows),
            "candidate_count": len(candidates),
            "mg_sel": args.mg_sel,
            "phosphate_o_sel": args.phosphate_o_sel,
        },
    )
    _write_outputs(out_dir, ranked, series, args)
    print(f"Saved CV2 auto-selection run: {out_dir}")


if __name__ == "__main__":
    main()
