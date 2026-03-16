"""
combined2.py (2D CV version)

Drop-in replacement for combined.py when you want *two* collective variables (CVs)
for protein RL training, while preserving the overall structure/flow:

- PPOAgent (actor/critic, same PPO update logic)
- ProteinEnvironmentRedesigned (OpenMM rollout, milestone/zone logic, per-step DCD)
- save_checkpoint / load via torch
- plot_distance_trajectory (now plots CV1 by default; also saves CV2 if present)

Key change vs 1D:
- Environment tracks two distances:
    CV1: distance(atom1, atom2)   (main progress CV)
    CV2: distance(atom3, atom4)   (auxiliary CV)
- Action maps to a single 2D Gaussian bias in (CV1, CV2).
- Bias is applied as the *sum of 2D Gaussians* accumulated across steps.

This keeps method signatures, agent logic, and main-loop usage unchanged.
"""

# ========================= Imports =========================

import os
import csv
import time
import uuid
import json
import random
import sys as _sys
import numpy as np
from datetime import datetime
from collections import deque, Counter
from tqdm import tqdm
from typing import Any, Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import openmm
from openmm import unit as u
import openmm.unit as omm_unit
from openmm.app import CharmmPsfFile, PDBFile, CharmmParameterSet
from openmm.app import Simulation, DCDReporter
from openmm.app import PME, HBonds, CutoffNonPeriodic
from openmm.app import NoCutoff

# ========================= Config (in-file) =========================

SEED = 42

# ---- OpenMM platform ----
class SliceableDeque(deque):
    """deque + supports list-style slicing (e.g. d[1:]) by materializing to list."""
    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return list(self)[idx]
        return super().__getitem__(idx)

def get_best_platform(verbose: bool = True):
    names = [
        openmm.Platform.getPlatform(i).getName()
        for i in range(openmm.Platform.getNumPlatforms())
    ]
    if verbose:
        print(f"Available OpenMM platforms: {names}")
    if "CUDA" in names:
        if verbose: print("Using CUDA platform (GPU)")
        return openmm.Platform.getPlatformByName("CUDA")
    if "OpenCL" in names:
        if verbose: print("Using OpenCL platform")
        return openmm.Platform.getPlatformByName("OpenCL")
    if verbose: print("Using CPU platform")
    return openmm.Platform.getPlatformByName("CPU")

# ---- Files ----
psf_file = "step3_input.psf"
pdb_file = "traj_0.restart.pdb"
toppar_file = "toppar.str"

# ---- 2D CV definitions (MUST set both pairs for true 2D) ----
# CV1 (primary progress CV)
ATOM1_INDEX = 7799
ATOM2_INDEX = 7840
CV1_LABEL = f"CV1 (atom {ATOM1_INDEX} - atom {ATOM2_INDEX} distance)"

# CV2 (auxiliary CV)  <-- YOU MUST SET THESE
ATOM3_INDEX = 487
ATOM4_INDEX = 3789
CV2_LABEL = f"CV2 (atom {ATOM3_INDEX} - atom {ATOM4_INDEX} distance)"

# Optional: override CV pairs via list (first two are used)
# e.g. ATOM_PAIRS = [(7799,7840), (123,456)]
ATOM_PAIRS = []

# ---- Targets (CV1) ----
CURRENT_DISTANCE = 3.3
FINAL_TARGET = 7.5
TARGET_CENTER = FINAL_TARGET
TARGET_ZONE_HALF_WIDTH = 0.35
TARGET_MIN = TARGET_CENTER - TARGET_ZONE_HALF_WIDTH
TARGET_MAX = TARGET_CENTER + TARGET_ZONE_HALF_WIDTH

# ---- Targets (CV2) ----
# If you do not know a good CV2 target yet:
#   - leave CURRENT_DISTANCE2 = None to auto-detect from the starting structure
#   - set TARGET2_ZONE_HALF_WIDTH to a reasonable corridor width (Å)
CURRENT_DISTANCE2 = 8.5
FINAL_TARGET2 = 4.0            # optional; if None, uses CURRENT_DISTANCE2
TARGET2_CENTER = FINAL_TARGET2           # optional; if None, uses FINAL_TARGET2 or CURRENT_DISTANCE2
TARGET2_ZONE_HALF_WIDTH = 0.35
TARGET2_MIN = TARGET2_CENTER - TARGET2_ZONE_HALF_WIDTH
TARGET2_MAX = TARGET2_CENTER + TARGET2_ZONE_HALF_WIDTH

# ---- Milestones ----
DISTANCE_INCREMENTS = [3.5, 3.8, 4.2, 4.6, 5.0, 5.4, 5.8, 6.2, 6.6, 7.0, 7.3]
DISTANCE2_INCREMENTS = [8.4, 7.6, 6.8, 6.0, 5.2, 4.4]
CURRICULUM_SUCCESS_WINDOW = 4
CURRICULUM_PROMOTION_THRESHOLD = 0.75
CURRICULUM_ZONE_SCALE = 0.40
CURRICULUM_MIN_HALF_WIDTH = 0.08

# ---- Locks / confinement (CV1) ----
ENABLE_MILESTONE_LOCKS = False
LOCK_MARGIN = 0.15
BACKSTOP_K = 3.0e4
PERSIST_LOCKS_ACROSS_EPISODES = False
CARRY_STATE_ACROSS_EPISODES = False
FREEZE_EXPLORATION_AT_ZONE = False
TRAIN_FRESH_START_EVERY_EPISODE = True

# ---- Zone confinement (CV1) ----
ZONE_CONFINEMENT = True
ZONE_K = 8.0e4
ZONE_MARGIN_LOW = 0.05
ZONE_MARGIN_HIGH = 0.05

# ---- Zone confinement (CV2) ----
CV2_ZONE_CONFINEMENT = True
CV2_ZONE_K = 8.0e4
CV2_ZONE_MARGIN_LOW = 0.05
CV2_ZONE_MARGIN_HIGH = 0.05
CV2_AMP_FRACTION = 0.5
CV2_CENTER_RESTRAINT = False
CV2_CENTER_K = 5.0e3   # start here; tune 2e4–2e5

SEED_ZONE_CAP_IF_BEST_IN_ZONE = True

# ---- Observation/action ----
# 2D state adds 2 features (cv2 normalized + cv2 trend), so default is 10.
STATE_SIZE = 11

AMP_BINS = [0.0, 4.0, 8.0, 12.0, 16.0]
# 2D Gaussian center offsets (A): PPO selects dx, dy relative to current CVs
DX_BINS = [-1.0, -0.5, 0.0, 0.5, 1.0]
DY_BINS = [-1.0, -0.5, 0.0, 0.5, 1.0]
# Shared width (A) for the 2D Gaussian
SIGMA_BINS = [0.35, 0.6, 0.9]
ACTION_SIZE = len(AMP_BINS) * len(DX_BINS) * len(DY_BINS) * len(SIGMA_BINS)
ACTION_MODE = "discrete"  # future: add continuous mode

MIN_AMP, MAX_AMP = 0.0, 32.0
APPLIED_MAX_AMP = 20.0
MIN_WIDTH, MAX_WIDTH = 0.1, 2.5
MAX_ESCALATION_FACTOR = 1.60
IN_ZONE_MAX_AMP = 1e9
OBS_CLIP = 8.0

# ---- PPO ----
N_STEPS = 8
BATCH_SIZE = 4
N_EPOCHS = 4
GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_RANGE = 0.2
ENT_COEF = 0.01
VF_COEF = 0.5
MAX_GRAD_NORM = 0.3
LR = 1e-4
PPO_TARGET_KL = 0.03

# ---- Episode & MD ----
MAX_ACTIONS_PER_EPISODE = 10
MAX_GAUSSIANS_PER_ACTION = 3
MAX_BIAS_TERMS_PER_EPISODE = MAX_ACTIONS_PER_EPISODE * MAX_GAUSSIANS_PER_ACTION
RIBBON_CV1_NEAR_OFFSET_A = 0.22
RIBBON_CV1_NEAR_OFFSET_SCALE = 0.18
RIBBON_CV1_SPAN_A = 0.55
RIBBON_CV1_SPAN_SCALE = 0.45
RIBBON_CV1_STALL_SPAN_SCALE = 0.12
RIBBON_CV1_BACK_OFFSET_MAX_A = 1.35
RIBBON_CV2_SPREAD_A = 0.60
RIBBON_CV2_SPREAD_SCALE = 0.35
RIBBON_CV2_SHIFT_SCALE = 0.30
RIBBON_SECONDARY_AMP_FRACTION = 0.82
RIBBON_TERTIARY_AMP_FRACTION = 0.68
RIBBON_SIGMA_Y_SCALE = 1.60
RIBBON_SECONDARY_SIGMA_X_SCALE = 1.10
RIBBON_TERTIARY_SIGMA_X_SCALE = 1.22
STRONG_BIAS_STALL_TRIGGER = 2
STRONG_BIAS_PLATEAU_MIN_A = 4.8
STRONG_BIAS_PLATEAU_MAX_A = 6.3
STRONG_BIAS_REMAINING_MIN_A = 1.0
STRONG_BIAS_GAIN = 0.18
PRIMARY_OFFSET_SIGMA_SCALE = 0.78
PRIMARY_OFFSET_SIGMA_SCALE_PLATEAU = 0.98
BIAS_FRONTIER_STEP_A = 0.12
BIAS_FRONTIER_STALL_STEP_A = 0.05
BIAS_FRONTIER_PLATEAU_STEP_A = 0.08
CV2_PROGRESS_REWARD_SCALE = 0.0
stepsize = 0.001 * u.picoseconds
fricCoef = 2.0 / u.picoseconds
T = 300 * u.kelvin

propagation_step = 4000
dcdfreq_mfpt = 40

# NaN recovery
MAX_INTEGRATOR_RETRIES = 2
MIN_STEPSIZE = 0.0005 * u.picoseconds
MD_FAILURE_PENALTY = -250.0

# ---- Rewards ----
PROGRESS_REWARD = 120.0
MILESTONE_REWARD = 200.0
BACKTRACK_PENALTY = -15.0
VELOCITY_BONUS = 10.0
STEP_PENALTY = -0.5

PHASE2_TOL = 0.08
CENTER_GAIN = 400.0
STABILITY_STEPS = 6
CONSISTENCY_BONUS = 50.0

# CV2 shaping (keeps auxiliary CV in corridor without dominating training)
CV2_DEVIATION_PENALTY = 0.0   # per Å outside CV2 zone

# ---- Curriculum / Eval ----
PROB_FRESH_START = 0.5
EVAL_EVERY = 5
N_EVAL_EPISODES = 3
SAVE_CHECKPOINT_EVERY = 5

ROOT_DIR = os.path.dirname(__file__)
RESULTS_DIR = os.path.join(ROOT_DIR, "results_PPO")
PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")
RUNS_DIR = os.path.join(RESULTS_DIR, "analysis_runs")
METRICS_CSV = f"{RESULTS_DIR}/training_metrics.csv"

EVAL_GREEDY = True

# ---- OpenMM system options ----
nonbondedCutoff = 1.0 * u.nanometer
backbone_constraint_strength = 100

# ---- DCD output ----
DCD_SAVE = True
RESULTS_TRAJ_DIR = os.path.join(RESULTS_DIR, "dcd_trajs")
DCD_REPORT_INTERVAL = dcdfreq_mfpt
RUN_NAME_PREFIX = "ep"
RUNS_TXT = os.path.join(RESULTS_TRAJ_DIR, "runs.txt")

# ---- Episode outputs ----
SAVE_EPISODE_PDB = True
EPISODE_PDB_EVERY = 1
EPISODE_PDB_DIR = os.path.join(RESULTS_DIR, "episode_pdbs")

SAVE_BIAS_PROFILE = True
BIAS_PROFILE_EVERY = 1
BIAS_PROFILE_DIR = os.path.join(RESULTS_DIR, "bias_profiles")
BIAS_PROFILE_BINS = 250
BIAS_PROFILE_PAD_SIGMA = 3.0
AUTO_REWEIGHTED_FES_PLOT = True
AUTO_REWEIGHTED_FES_EVERY = 1
AUTO_REWEIGHTED_FES_BINS = 120
AUTO_REWEIGHTED_FES_TEMPERATURE = 300.0


def curriculum_cv1_targets():
    vals = [float(x) for x in DISTANCE_INCREMENTS]
    if not vals or abs(vals[-1] - float(FINAL_TARGET)) > 1e-6:
        vals.append(float(FINAL_TARGET))
    out = []
    for v in vals:
        if not out or abs(v - out[-1]) > 1e-6:
            out.append(v)
    return out


def curriculum_half_width_for_target(target_center_A: float) -> float:
    gap = max(0.0, float(target_center_A) - float(CURRENT_DISTANCE))
    scaled = float(gap) * float(CURRICULUM_ZONE_SCALE)
    return float(np.clip(scaled, float(CURRICULUM_MIN_HALF_WIDTH), float(TARGET_ZONE_HALF_WIDTH)))

# --------------------- module self-alias (for parity with combined.py) ---------------------
config = _sys.modules[__name__]

# ========================= Small utilities =========================
def load_charmm_params(filename: str):
    """
    Load CHARMM parameters the same way as combined.py:
    - Read `toppar.str` line-by-line
    - Strip comments after '!'
    - Treat each non-empty line as a referenced parameter/topology file
    """
    par_files = []
    with open(filename, "r") as f:
        for line in f:
            line = line.split("!")[0].strip()
            if line:
                par_files.append(line)
    return CharmmParameterSet(*tuple(par_files))

def add_backbone_posres(system: openmm.System,
                        psf: CharmmPsfFile,
                        pdb: PDBFile,
                        strength: float,
                        skip_indices=None):
    """
    Same as combined.py: restrain backbone atoms (N, CA, C) to initial coordinates
    using a CustomExternalForce. Skip indices in skip_indices (e.g. CV atoms).
    """
    if skip_indices is None:
        skip_indices = set()
    else:
        skip_indices = set(skip_indices)

    force = openmm.CustomExternalForce("k*((x-x0)^2 + (y-y0)^2 + (z-z0)^2)")
    force.addGlobalParameter("k", float(strength))
    force.addPerParticleParameter("x0")
    force.addPerParticleParameter("y0")
    force.addPerParticleParameter("z0")

    for i, pos in enumerate(pdb.positions):
        if i in skip_indices:
            continue
        # CharmmPsfFile exposes atom_list in OpenMM app
        if psf.atom_list[i].name in ("N", "CA", "C"):
            xyz = pos.value_in_unit(u.nanometer)
            force.addParticle(i, xyz)

    system.addForce(force)
    return force

def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)
    return path

def _ensure(path: str):
    os.makedirs(path, exist_ok=True)
    return path

# ========================= Plotting / checkpoints =========================
# This section brings combined.py parity utilities into the 2D-CV module.
def _ensure(path: str) -> str:
    """Create directory (if needed) and return the path."""
    _ensure_dir(path)
    return path

# ======================== EPISODE EXPORT =====================
def export_episode_metadata(
    episode_num: int,
    bias_log,
    backstops_A,
    backstop_events,
    curriculum_target_center_A=None,
    curriculum_target_zone=None,
    target_stage=None,
) -> None:
    meta_dir = _ensure(f"{config.RESULTS_DIR}/episode_meta/")
    bias_cols = ["step", "kind", "amp_kcal", "center1_A", "center2_A", "sigma_x_A", "sigma_y_A"]
    meta = {
        "episode": int(episode_num),
        "bias_log_columns": bias_cols,
        "bias_log": [list(x) for x in bias_log],
        "backstops_A": list(map(float, backstops_A or [])),
        "backstop_events": [list(map(float, x)) for x in (backstop_events or [])],
        "start_A": float(config.CURRENT_DISTANCE),
        "target_center_A": float(config.TARGET_CENTER),
        "target_zone": [float(config.TARGET_MIN), float(config.TARGET_MAX)],
        "curriculum_target_center_A": float(
            config.TARGET_CENTER if curriculum_target_center_A is None else curriculum_target_center_A
        ),
        "curriculum_target_zone": [
            float(config.TARGET_MIN if curriculum_target_zone is None else curriculum_target_zone[0]),
            float(config.TARGET_MAX if curriculum_target_zone is None else curriculum_target_zone[1]),
        ],
        "target_stage": None if target_stage is None else int(target_stage),
    }
    with open(os.path.join(meta_dir, f"episode_{episode_num:04d}.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Saved episode metadata JSON for episode {episode_num}.")

# ---------------------------------------------------------------------
# End-of-episode PDB writer (optional helper)
# ---------------------------------------------------------------------
def _now_tag() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")

def _find_attr(obj, names):
    for n in names:
        if hasattr(obj, n):
            val = getattr(obj, n)
            if val is not None:
                return val
    return None

def _get_topology(env, simulation=None):
    if simulation is not None:
        topo = getattr(simulation, "topology", None)
        if topo is not None:
            return topo
    psf = getattr(env, "psf", None)
    if psf is not None and hasattr(psf, "topology"):
        return psf.topology
    topo = getattr(env, "_last_topology", None)
    if topo is not None:
        return topo
    raise RuntimeError("No Topology found (simulation.topology / env.psf.topology / env._last_topology).")

def _get_positions_from_simulation(simulation):
    state = simulation.context.getState(getPositions=True)
    pos = state.getPositions(asNumpy=True)
    # Must be a Quantity with units
    if not hasattr(pos, "unit"):
        raise RuntimeError("Simulation positions lack units.")
    return pos

def _get_positions_from_context(context):
    state = context.getState(getPositions=True)
    pos = state.getPositions(asNumpy=True)
    if not hasattr(pos, "unit"):
        raise RuntimeError("Context positions lack units.")
    return pos

def _coerce_positions_quantity(pos_array):
    # Wrap (N,3) array as Quantity in nm
    arr = np.asarray(pos_array)
    if arr.ndim == 2 and arr.shape[1] == 3:
        return arr * u.nanometer
    raise RuntimeError("Cached positions are not a valid (N,3) array.")

def write_episode_pdb(env: "ProteinEnvironmentRedesigned", out_dir: str, episode_idx: int) -> str:
    _ensure_dir(out_dir)

    simulation = _find_attr(env, ("simulation", "sim", "_simulation", "_last_simulation"))
    context = None
    topo = None
    pos = None

    if simulation is not None:
        topo = _get_topology(env, simulation=simulation)
        try:
            pos = _get_positions_from_simulation(simulation)
        except Exception:
            pos = None

    if pos is None:
        context = _find_attr(env, ("context", "_context", "sim_context", "_last_context"))
        if context is not None and topo is None:
            topo = _get_topology(env, simulation=None)
        if context is not None and pos is None:
            try:
                pos = _get_positions_from_context(context)
            except Exception:
                pos = None

    if pos is None:
        cached = _find_attr(env, ("_last_positions", "current_positions", "positions_cache"))
        if cached is not None:
            pos = _coerce_positions_quantity(cached)

    if topo is None or pos is None:
        raise RuntimeError(
            "Could not assemble Topology+Positions. Checked Simulation/Context and cached positions. "
            "Ensure env caches _last_topology and _last_positions/current_positions."
        )

    tag = _now_tag()
    fname = os.path.join(out_dir, f"{tag}_episode_{episode_idx:04d}.pdb")
    with open(fname, "w") as fh:
        PDBFile.writeFile(topo, pos, fh, keepIds=True)

    print(f"[episode_pdb_writer] Saved end-of-episode PDB: {fname}")
    return fname

# ---------------------------------------------------------------------
# Metrics CSV helper
# ---------------------------------------------------------------------
def append_metrics_row(path: str, row_dict: Dict[str, Any]) -> None:
    _ensure(os.path.dirname(path))
    write_header = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=sorted(row_dict.keys()))
        if write_header:
            w.writeheader()
        w.writerow(row_dict)

# ---------------------------------------------------------------------
# Plotting utilities
# ---------------------------------------------------------------------
def plot_distance_trajectory(
    episode_trajectories: List[List[float]],
    episode_num: int,
    distance_history: Optional[List[float]] = None,
    cv2_history: Optional[List[float]] = None,
    episode_trajectories_cv2: Optional[List[List[float]]] = None,
) -> None:
    """
    Plot full CV1 trajectory with the same time-axis logic as combined.py.
    If cv2_history is provided, also save CV2 plot/CSV.
    """
    plot_dir = _ensure(f"{config.RESULTS_DIR}/full_trajectories/")

    # Fallback to distance_history if no segments were captured
    if ((not episode_trajectories or len(episode_trajectories) == 0)
        and distance_history is not None and len(distance_history) > 1):
        episode_trajectories = [list(map(float, distance_history[1:]))]

    if not episode_trajectories:
        print(f"[Warning] No trajectory data recorded for episode {episode_num}. Nothing to plot.")
        return

    segs = [np.asarray(seg, dtype=np.float32) for seg in episode_trajectories if len(seg) > 0]
    full_traj = np.concatenate(segs) if len(segs) > 1 else segs[0]
    if full_traj.ndim == 0:
        full_traj = full_traj.reshape(1)

    dt_frame_ps = config.dcdfreq_mfpt * float(config.stepsize.value_in_unit(omm_unit.picoseconds))
    time_axis_ps = np.arange(len(full_traj), dtype=np.float32) * dt_frame_ps

    plt.figure(figsize=(9, 4.5))
    plt.plot(time_axis_ps, full_traj, linewidth=1.8)
    plt.axhspan(config.TARGET_MIN, config.TARGET_MAX, alpha=0.18, label="Target zone (CV1)")
    plt.axhline(config.TARGET_CENTER, linestyle="--", linewidth=1.0, label="Target center")
    plt.xlabel("Time (ps)")
    plt.ylabel(f"{config.CV1_LABEL} (Å)")
    plt.title(f"Episode {episode_num:04d} CV1 trajectory")
    plt.legend()
    out_png = os.path.join(plot_dir, f"progressive_traj_ep_{episode_num:04d}_cv1.png")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()
    print(f"Saved trajectory plot: {out_png}")

    out_csv = os.path.join(plot_dir, f"progressive_traj_ep_{episode_num:04d}_cv1.csv")
    np.savetxt(out_csv, np.c_[time_axis_ps, full_traj],
               delimiter=",", header="time_ps,cv1_distance_A", comments="")
    print(f"Saved trajectory CSV: {out_csv}")

    # ---- CV2 plotted exactly like CV1 (from per-chunk segments) ----
    if episode_trajectories_cv2:
        segs2 = [np.asarray(seg, dtype=np.float32) for seg in episode_trajectories_cv2 if len(seg) > 0]
        cv2_full = np.concatenate(segs2) if len(segs2) > 1 else segs2[0]
        if cv2_full.ndim == 0:
            cv2_full = cv2_full.reshape(1)

        # ---- 2D phase-space plot: CV1 vs CV2 colored by time ----
        n = int(min(len(full_traj), len(cv2_full)))
        if n > 1:
            x_cv1 = np.asarray(full_traj[:n], dtype=np.float32)
            y_cv2 = np.asarray(cv2_full[:n], dtype=np.float32)
            t_ps  = np.asarray(time_axis_ps[:n], dtype=np.float32)

            plt.figure(figsize=(6.5, 5.5))
            sc = plt.scatter(x_cv1, y_cv2, c=t_ps, s=10)
            cbar = plt.colorbar(sc)
            cbar.set_label("Time (ps)")

            plt.axvspan(config.TARGET_MIN, config.TARGET_MAX, alpha=0.15)
            plt.axhspan(config.TARGET2_MIN, config.TARGET2_MAX, alpha=0.12)

            plt.xlabel(f"{config.CV1_LABEL} (Å)")
            plt.ylabel(f"{config.CV2_LABEL} (Å)")
            plt.title(f"Episode {episode_num:04d} CV1 vs CV2 (colored by time)")
            out2d_png = os.path.join(plot_dir, f"progressive_traj_ep_{episode_num:04d}_cv1_cv2_2d.png")
            plt.tight_layout()
            plt.savefig(out2d_png, dpi=220)
            plt.close()
            print(f"Saved 2D CV plot: {out2d_png}")

            out2d_csv = os.path.join(plot_dir, f"progressive_traj_ep_{episode_num:04d}_cv1_cv2_2d.csv")
            np.savetxt(
                out2d_csv,
                np.c_[t_ps, x_cv1, y_cv2],
                delimiter=",",
                header="time_ps,cv1_pi_mg_A,cv2_atom3_atom4_A",
                comments=""
            )
            print(f"Saved 2D CV CSV: {out2d_csv}")

        # ---- regular CV2 time-series plot/CSV (always runs when CV2 exists) ----
        dt_frame_ps = config.dcdfreq_mfpt * float(config.stepsize.value_in_unit(omm_unit.picoseconds))
        time_axis2_ps = np.arange(len(cv2_full), dtype=np.float32) * dt_frame_ps

        plt.figure(figsize=(9, 4.5))
        plt.plot(time_axis2_ps, cv2_full, linewidth=1.8)
        plt.axhspan(config.TARGET2_MIN, config.TARGET2_MAX, alpha=0.15, label="Target zone (CV2)")
        plt.axhline(config.TARGET2_CENTER, linestyle="--", linewidth=1.0, label="CV2 center")
        plt.xlabel("Time (ps)")
        plt.ylabel(f"{config.CV2_LABEL} (Å)")
        plt.title(f"Episode {episode_num:04d} CV2 trajectory")
        plt.legend()
        out2_png = os.path.join(plot_dir, f"progressive_traj_ep_{episode_num:04d}_cv2.png")
        plt.tight_layout()
        plt.savefig(out2_png, dpi=200)
        plt.close()
        print(f"Saved CV2 plot: {out2_png}")

        out2_csv = os.path.join(plot_dir, f"progressive_traj_ep_{episode_num:04d}_cv2.csv")
        np.savetxt(out2_csv, np.c_[time_axis2_ps, cv2_full],
                   delimiter=",", header="time_ps,cv2_distance_A", comments="")
        print(f"Saved CV2 CSV: {out2_csv}")

def save_episode_bias_profiles(all_biases, episode_num: int):
    """Save total 2D bias surface (sum of 2D Gaussians)."""
    if not all_biases:
        return

    out_dir = _ensure(config.BIAS_PROFILE_DIR)

    centers1 = np.array([b[1] for b in all_biases], dtype=float)
    centers2 = np.array([b[2] for b in all_biases], dtype=float)
    widths1 = np.array([max(1e-6, b[3]) for b in all_biases], dtype=float)
    widths2 = np.array([max(1e-6, b[4]) for b in all_biases], dtype=float)

    pad = float(config.BIAS_PROFILE_PAD_SIGMA)
    lo1 = float(np.min(centers1 - pad * widths1)) if centers1.size else max(0.5, float(config.CURRENT_DISTANCE) - 1.0)
    hi1 = float(np.max(centers1 + pad * widths1)) if centers1.size else float(config.FINAL_TARGET) + 1.0
    lo2 = float(np.min(centers2 - pad * widths2)) if centers2.size else max(0.5, float(config.FINAL_TARGET2) - 1.0)
    hi2 = float(np.max(centers2 + pad * widths2)) if centers2.size else (
        float(config.CURRENT_DISTANCE2) + 1.0 if config.CURRENT_DISTANCE2 is not None else 10.0
    )

    x = np.linspace(min(lo1, float(config.CURRENT_DISTANCE) - 1.0), max(hi1, float(config.FINAL_TARGET) + 1.0),
                    int(config.BIAS_PROFILE_BINS))
    y = np.linspace(min(lo2, float(config.FINAL_TARGET2) - 1.0), max(hi2, float(config.CURRENT_DISTANCE2) + 1.0),
                    int(config.BIAS_PROFILE_BINS))
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X, dtype=np.float64)
    for amp, c1, c2, sx, sy in all_biases:
        sx = max(1e-6, float(sx))
        sy = max(1e-6, float(sy))
        Z += float(amp) * np.exp(-((X - float(c1)) ** 2) / (2.0 * sx ** 2)
                                 -((Y - float(c2)) ** 2) / (2.0 * sy ** 2))

    png = os.path.join(out_dir, f"ep_{episode_num:04d}_bias2d.png")
    npy = os.path.join(out_dir, f"ep_{episode_num:04d}_bias2d.npy")
    np.save(npy, Z)

    plt.figure(figsize=(6.5, 5.5))
    plt.imshow(Z, origin="lower", aspect="auto", cmap="coolwarm",
               extent=[x.min(), x.max(), y.min(), y.max()])
    plt.colorbar(label="Bias (kcal/mol)")
    plt.xlabel(f"{config.CV1_LABEL} (A)")
    plt.ylabel(f"{config.CV2_LABEL} (A)")
    plt.title(f"Episode {episode_num:04d} 2D Bias Surface")
    plt.tight_layout()
    plt.savefig(png, dpi=220)
    plt.close()


def compute_coverage(dist_segments, bin_edges):
    if not dist_segments:
        return np.zeros(len(bin_edges) - 1, dtype=float)
    arrs = [np.asarray(seg, dtype=float) for seg in dist_segments if len(seg) > 0]
    if not arrs:
        return np.zeros(len(bin_edges) - 1, dtype=float)
    data = np.concatenate(arrs)
    hist, _ = np.histogram(data, bins=bin_edges)
    cov = hist.astype(float)
    if cov.sum() > 0:
        cov = cov / cov.sum()
    return cov

def plot_coverage_histogram(dist_segments, episode_num: int, bin_size=0.25):
    out_dir = _ensure(f"{config.RESULTS_DIR}/coverage/")
    lo = max(0.5, config.CURRENT_DISTANCE - 1.0)
    hi = config.FINAL_TARGET + 1.5
    bins = np.arange(lo, hi + bin_size, bin_size)
    cov = compute_coverage(dist_segments, bins)

    centers = 0.5 * (bins[1:] + bins[:-1])
    plt.figure(figsize=(10, 4))
    plt.bar(centers, cov, width=bin_size * 0.9, align='center')
    plt.axvspan(config.TARGET_MIN, config.TARGET_MAX, alpha=0.18, label='Target Zone')
    for m in config.DISTANCE_INCREMENTS:
        plt.axvline(x=m, linestyle=':', alpha=0.4)
    plt.xlabel('Distance (Å)')
    plt.ylabel('Visit fraction')
    plt.title(f'Distance Coverage — Episode {episode_num:04d}')
    plt.legend(loc='upper right')
    out = os.path.join(out_dir, f"coverage_ep_{episode_num:04d}.png")
    plt.tight_layout()
    plt.savefig(out, dpi=250)
    plt.close()
    print(f"Saved coverage histogram: {out}")

def _bias_energy_components(bias_log, backstops_A, xA_grid):
    """
    Parity with combined.py:
      - returns per-bias energy curves and total energy curve (kJ/mol)
      - includes backstop walls if provided
    Note: 2D Gaussian biases are not represented here (1D projection only).
    """
    r_nm = xA_grid / 10.0
    per_bias = []
    total = np.zeros_like(xA_grid, dtype=float)

    for entry in bias_log:
        if len(entry) == 7:
            # 2D Gaussian bias (skip for 1D projection)
            continue
        if len(entry) == 6:
            _, cv_id, kind, amp_kcal, center_A, width_A = entry
        else:
            continue

        # Only decompose CV1 Gaussian terms
        if int(cv_id) != 1:
            continue
        if kind not in ("gaussian", "bias", "bias1"):
            continue

        A_kJ = float(amp_kcal) * 4.184
        mu_nm = float(center_A) / 10.0
        sig_nm = max(1e-6, float(width_A) / 10.0)
        e = A_kJ * np.exp(-((r_nm - mu_nm) ** 2) / (2.0 * sig_nm ** 2))
        per_bias.append(e)
        total += e

    if backstops_A:
        for m_eff_A in backstops_A:
            m_nm = float(m_eff_A) / 10.0
            mask = r_nm < m_nm
            e = np.zeros_like(xA_grid, dtype=float)
            e[mask] = float(config.BACKSTOP_K) * (m_nm - r_nm[mask]) ** 2
            total += e
            per_bias.append(e)

    return per_bias, total

def plot_bias_components_and_sum(bias_log, backstops_A, episode_num):
    if not bias_log and not backstops_A:
        return
    plot_dir = _ensure(f"{config.RESULTS_DIR}/bias_profiles/")
    lo = max(0.5, config.CURRENT_DISTANCE - 2.0)
    hi = config.FINAL_TARGET + 2.0
    xA = np.linspace(lo, hi, 1000)
    per_bias, total = _bias_energy_components(bias_log, backstops_A, xA)

    plt.figure(figsize=(12, 7))
    for i, e in enumerate(per_bias):
        label = (f"Bias {i+1}" if i < len(bias_log) else f"Backstop {i - len(bias_log) + 1}")
        plt.plot(xA, e, linewidth=1.5, alpha=0.9, label=label)
    plt.plot(xA, total, linewidth=2.5, alpha=1.0, label='Cumulative Bias')

    plt.axvline(x=config.CURRENT_DISTANCE, linestyle='--', linewidth=2, label='Start')
    plt.axvspan(config.TARGET_MIN, config.TARGET_MAX, alpha=0.18, label='Target Zone')
    plt.axvline(x=config.TARGET_CENTER, linestyle='--', linewidth=3, label='Target Center')

    plt.xlabel('Position (Å)')
    plt.ylabel('Bias Energy (kJ/mol)')
    plt.title(f'Bias Potentials in Episode {episode_num}')
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.04, 1), loc='upper left')
    plt.tight_layout()
    out = os.path.join(plot_dir, f"bias_components_ep_{episode_num:04d}.png")
    plt.savefig(out, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved bias components plot: {out}")

def plot_bias_sum_only(bias_log, backstops_A, episode_num):
    if not bias_log and not backstops_A:
        return
    plot_dir = _ensure(f"{config.RESULTS_DIR}/bias_profiles/")
    lo = max(0.5, config.CURRENT_DISTANCE - 2.0)
    hi = config.FINAL_TARGET + 2.0
    xA = np.linspace(lo, hi, 1000)
    _, total = _bias_energy_components(bias_log, backstops_A, xA)

    plt.figure(figsize=(12, 7))
    plt.plot(xA, total, linewidth=2.5)
    plt.axvline(x=config.CURRENT_DISTANCE, linestyle='--', linewidth=2, label='Start')
    plt.axvspan(config.TARGET_MIN, config.TARGET_MAX, alpha=0.18, label='Target Zone')
    plt.axvline(x=config.TARGET_CENTER, linestyle='--', linewidth=3, label='Target Center')

    idx_min = np.argmin(total)
    plt.scatter([xA[idx_min]], [total[idx_min]], s=60, label=f"Min: {xA[idx_min]:.2f} Å")

    plt.xlabel('Position (Å)')
    plt.ylabel('Bias Energy (kJ/mol)')
    plt.title(f'Cumulative Bias Landscape — Episode {episode_num:04d}')
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.04, 1), loc='upper left')
    plt.tight_layout()
    out = os.path.join(plot_dir, f"bias_sum_ep_{episode_num:04d}.png")
    plt.savefig(out, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved cumulative bias plot: {out}")

def plot_bias_timeline(bias_log, backstop_events, episode_num):
    plot_dir = _ensure(f"{config.RESULTS_DIR}/bias_profiles/")
    if not bias_log:
        return
    steps = [int(e[0]) for e in bias_log]
    amps = []
    for e in bias_log:
        if len(e) >= 3:
            amps.append(float(e[2]))
        else:
            amps.append(0.0)
    plt.figure(figsize=(9, 4.0))
    plt.plot(steps, amps, linewidth=1.8)
    if backstop_events:
        for (s, _) in backstop_events:
            plt.axvline(int(s), linestyle="--", linewidth=1.0, alpha=0.5)
    plt.xlabel("RL step")
    plt.ylabel("Bias amplitude (kcal/mol)")
    plt.title(f"Episode {episode_num:04d} bias amplitude timeline")
    out = os.path.join(plot_dir, f"bias_timeline_ep_{episode_num:04d}.png")
    plt.tight_layout()
    plt.savefig(out, dpi=200)
    plt.close()

def plot_lock_snapshot(backstops_A, episode_num):
    if not backstops_A:
        return
    lock_dir = _ensure(f"{config.RESULTS_DIR}/locks/")
    plt.figure(figsize=(12, 3))
    for i, m in enumerate(backstops_A):
        plt.axvline(x=m, linestyle="--", label=f"Lock {i+1} @ {m:.2f} Å")
    plt.axvspan(config.TARGET_MIN, config.TARGET_MAX, alpha=0.18, label="Target Zone")
    plt.axvline(x=config.TARGET_CENTER, linestyle="--", linewidth=2, label="Target Center")
    plt.xlim(max(0.5, config.CURRENT_DISTANCE - 1.5), config.FINAL_TARGET + 1.5)
    plt.ylim(0, 1)
    plt.yticks([])
    plt.xlabel("Position (Å)")
    plt.title(f"Milestone Locks Snapshot — Episode {episode_num}")
    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    plt.tight_layout()
    out = os.path.join(lock_dir, f"locks_ep_{episode_num:04d}.png")
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved locks snapshot: {out}")

# ---------------------------------------------------------------------
# Checkpoints
# ---------------------------------------------------------------------
def _checkpoint_config_snapshot():
    """
    Keep checkpoint metadata lightweight and serialization-safe.

    The previous implementation tried to store every uppercase symbol from the
    config module, which can include OpenMM/unit objects and other values that
    are not guaranteed to round-trip through torch.save().
    """
    keys = [
        "SEED",
        "STATE_SIZE",
        "ACTION_SIZE",
        "AMP_BINS",
        "DX_BINS",
        "DY_BINS",
        "SIGMA_BINS",
        "CURRENT_DISTANCE",
        "FINAL_TARGET",
        "CURRENT_DISTANCE2",
        "FINAL_TARGET2",
        "TARGET_CENTER",
        "TARGET_ZONE_HALF_WIDTH",
        "TARGET2_CENTER",
        "TARGET2_ZONE_HALF_WIDTH",
        "MAX_ACTIONS_PER_EPISODE",
        "MAX_GAUSSIANS_PER_ACTION",
        "LR",
        "GAMMA",
        "GAE_LAMBDA",
        "CLIP_RANGE",
        "ENT_COEF",
        "VF_COEF",
        "PPO_TARGET_KL",
    ]
    snap = {}
    for key in keys:
        if hasattr(config, key):
            val = getattr(config, key)
            if isinstance(val, (list, tuple)):
                snap[key] = list(val)
            elif isinstance(val, (str, int, float, bool)):
                snap[key] = val
    return snap


def save_checkpoint(
    agent: "PPOAgent",
    env: "ProteinEnvironmentRedesigned",
    ckpt_dir: str,
    episode: int,
    training_meta: Optional[Dict[str, Any]] = None,
) -> None:
    _ensure(ckpt_dir)
    path = os.path.join(ckpt_dir, f"ckpt_ep_{episode:04d}.pt")
    tmp_path = path + ".tmp"
    payload = {
        "episode": int(episode),
        "actor": agent.actor.state_dict(),
        "critic": agent.critic.state_dict(),
        "optimizer": agent.optimizer.state_dict(),
        "scheduler": agent.scheduler.state_dict(),
        "obs_norm": agent.obs_norm_state(),
        "config": _checkpoint_config_snapshot(),
        "env_meta": {
            "best_distance_ever": float(getattr(env, "best_distance_ever", 0.0)),
            "phase": int(getattr(env, "phase", 1)),
        },
        "training_meta": dict(training_meta or {}),
    }
    try:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        torch.save(payload, tmp_path)
        # Verify the just-written archive before publishing it as the official checkpoint.
        _ = torch.load(tmp_path, map_location="cpu", weights_only=False)
        os.replace(tmp_path, path)
        print(f"Saved checkpoint: {path}")
    except Exception as e:
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass
        print(f"[warn] failed to save checkpoint: {e}")

# ======================== Monitoring helpers =================
def _safe_import_mdanalysis():
    try:
        import MDAnalysis
        from MDAnalysis.analysis import distances
        return MDAnalysis, distances
    except Exception:
        print("[monitor] MDAnalysis not available; skipping trajectory analysis.")
        return None, None

def analyze_mg_coordination_for_dcd(psf_file, dcd_file, out_dir, run_name):
    MDAnalysis, distances = _safe_import_mdanalysis()
    if MDAnalysis is None:
        return None

    _ensure(out_dir)
    out_path = os.path.join(out_dir, f"{run_name}.txt")

    u = MDAnalysis.Universe(psf_file, dcd_file)
    mg_sel = u.select_atoms("resname MG")
    coordinating_atoms_updating = u.select_atoms("name O* and around 5 resname MG", updating=True)

    coord_counts = []
    site_counts = Counter()

    with open(out_path, "w") as output:
        for ts in u.trajectory:
            if len(coordinating_atoms_updating) == 0 or len(mg_sel) == 0:
                continue

            dist_arr = distances.distance_array(mg_sel, coordinating_atoms_updating)
            d_sorted = np.sort(dist_arr)[0]
            if len(d_sorted) == 0:
                continue
            if len(d_sorted) >= 7:
                cutoff = d_sorted[5] + (d_sorted[6] - d_sorted[5]) / 2.0
            elif len(d_sorted) >= 6:
                cutoff = d_sorted[5]
            else:
                cutoff = d_sorted[-1]

            temp_sele = u.select_atoms(f"name O* and around {cutoff} resname MG", updating=True)
            n = min(6, len(temp_sele))
            coord_counts.append(n)

            for ik in range(n):
                atom = temp_sele[ik]
                key = (atom.resid, atom.resname, atom.name)
                site_counts[key] += 1
                output.write(f"{atom.resid}_{atom.resname}_{atom.name}")
                output.write("," if ik < n - 1 else "\n")

    metrics = {}
    if coord_counts:
        arr = np.asarray(coord_counts, dtype=float)
        metrics["mg_coordination_mean"] = float(arr.mean())
        metrics["mg_coordination_std"] = float(arr.std())
        metrics["mg_unique_sites"] = int(len(site_counts))
    print(f"[monitor] Wrote Mg coordination file: {out_path}")
    return metrics


def analyze_pi_path_for_dcd(psf_file, dcd_file, out_dir, run_name):
    MDAnalysis, _ = _safe_import_mdanalysis()
    if MDAnalysis is None:
        return None

    _ensure(out_dir)
    out_path = os.path.join(out_dir, f"Pi-{run_name}.txt")

    u = MDAnalysis.Universe(psf_file, dcd_file)
    coord1 = u.select_atoms(
        "not segid HETC and around 2 (segid HETC and (name O2 or name O3))",
        updating=True
    )
    coord2 = u.select_atoms(
        "not segid HETC and around 2 (segid HETC and (name H1 or name H2))",
        updating=True
    )

    site_counts = Counter()
    frame_counts = []

    with open(out_path, "w") as output:
        for ts in u.trajectory:
            coordinating_atoms_updating = coord1 + coord2
            n_tot = len(coordinating_atoms_updating)
            frame_counts.append(n_tot)

            for ik in range(n_tot):
                atom = coordinating_atoms_updating[ik]
                key = (atom.resid, atom.resname, atom.name)
                site_counts[key] += 1
                output.write(f"{atom.resid}_{atom.resname}_{atom.name}")
                output.write("," if ik < n_tot - 1 else "\n")

    metrics = {}
    if frame_counts:
        arr = np.asarray(frame_counts, dtype=float)
        metrics["pi_contacts_mean"] = float(arr.mean())
        metrics["pi_contacts_std"] = float(arr.std())
        metrics["pi_unique_sites"] = int(len(site_counts))
    print(f"[monitor] Wrote Pi-path file: {out_path}")
    return metrics


def run_mdanalysis_monitoring(run_name, psf_file, dcd_file):
    if not os.path.isfile(dcd_file):
        print(f"[monitor] DCD file not found for run {run_name}: {dcd_file}")
        return None

    mg_metrics = analyze_mg_coordination_for_dcd(psf_file, dcd_file, config.MG_MONITOR_DIR, run_name)
    pi_metrics = analyze_pi_path_for_dcd(psf_file, dcd_file, config.PI_MONITOR_DIR, run_name)

    return {"mg": mg_metrics, "pi": pi_metrics}

# --------------------- public exports ---------------------
__all__ = [
    "config",
    "PPOAgent", "RunningNorm", "Actor", "Critic",
    "ProteinEnvironmentRedesigned",
    "write_episode_pdb",
    "_ensure", "append_metrics_row", "plot_distance_trajectory",
    "plot_coverage_histogram", "plot_bias_sum_only",
    "plot_bias_components_and_sum", "plot_bias_timeline",
    "plot_lock_snapshot", "export_episode_metadata",
    "save_checkpoint", "run_mdanalysis_monitoring",
]
class RunningNorm:
    def __init__(self):
        self.mean = None
        self.var = None
        self.count = 1e-8

    def update(self, x: np.ndarray):
        x = np.asarray(x, dtype=np.float64)
        if x.ndim == 1:
            x = x[None, :]
        batch_mean = x.mean(axis=0)
        batch_var = x.var(axis=0)
        batch_count = x.shape[0]

        if self.mean is None:
            self.mean = batch_mean
            self.var = batch_var
            self.count = batch_count
            return

        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + (delta ** 2) * self.count * batch_count / tot_count
        new_var = M2 / tot_count

        self.mean = new_mean
        self.var = new_var
        self.count = tot_count

    def normalize(self, x):
        if self.mean is None or self.var is None:
            return np.asarray(x, dtype=np.float32)
        x = np.asarray(x, dtype=np.float32)
        norm = (x - self.mean.astype(np.float32)) / (
            np.sqrt(self.var.astype(np.float32)) + 1e-8
        )
        return np.clip(norm, -float(OBS_CLIP), float(OBS_CLIP))

    def state_dict(self):
        return {
            "mean": None if self.mean is None else self.mean.tolist(),
            "var": None if self.var is None else self.var.tolist(),
            "count": float(self.count),
        }

    def load_state_dict(self, d):
        if not d:
            return
        self.mean = None if d.get("mean") is None else np.asarray(d["mean"], dtype=np.float64)
        self.var = None if d.get("var") is None else np.asarray(d["var"], dtype=np.float64)
        self.count = float(d.get("count", 1e-8))

# ========================= PPO networks / agent =========================
class Actor(nn.Module):
    def __init__(self, state_size, action_size, seed, fc1=256, fc2=128, fc3=64):
        super().__init__()
        torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1)
        self.fc2 = nn.Linear(fc1, fc2)
        self.fc3 = nn.Linear(fc2, fc3)
        self.fc4 = nn.Linear(fc3, action_size)

    def forward_logits(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return torch.clamp(self.fc4(x), -20, 20)

    def forward(self, state):
        return F.softmax(self.forward_logits(state), dim=-1)


class Critic(nn.Module):
    def __init__(self, state_size, seed, fc1=256, fc2=128, fc3=64):
        super().__init__()
        torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1)
        self.fc2 = nn.Linear(fc1, fc2)
        self.fc3 = nn.Linear(fc2, fc3)
        self.fc4 = nn.Linear(fc3, 1)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.fc4(x)

class PPOAgent:
    def __init__(self, state_size, action_size, seed):
        self.state_size = state_size; self.action_size = action_size; self.seed = seed
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

        self.action_map = [(A, dx, dy, s) for A in config.AMP_BINS
                                         for dx in config.DX_BINS
                                         for dy in config.DY_BINS
                                         for s in config.SIGMA_BINS]

        self.actor = Actor(state_size, action_size, seed)
        self.critic = Critic(state_size, seed)
        self.optimizer = optim.Adam(
            list(self.actor.parameters()) + list(self.critic.parameters()),
            lr=config.LR, eps=1e-5
        )
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=100, gamma=0.95)

        self.memory = []
        self.gamma = config.GAMMA; self.gae_lambda = config.GAE_LAMBDA
        self.clip_range = config.CLIP_RANGE; self.n_epochs = config.N_EPOCHS
        self.batch_size = config.BATCH_SIZE; self.ent_coef = config.ENT_COEF
        self.vf_coef = config.VF_COEF; self.max_grad_norm = config.MAX_GRAD_NORM
        self.target_kl = config.PPO_TARGET_KL

        self.exploration_noise = 0.1; self.min_exploration_noise = 0.01
        self.exploration_decay = 0.995

        self.obs_rms = RunningNorm()
        self.episode_count = 0

    def _mask_logits_if_needed(self, logits, raw_state_batch):
        if logits.ndim == 1:
            logits = logits.unsqueeze(0)
            raw_state_batch = raw_state_batch.unsqueeze(0)
        mask_vals = []
        for s in raw_state_batch.detach().cpu().numpy():
            cv1_in_zone = (s[5] >= 0.5)    # CV1 in-zone flag
            cv2_in_zone = (s[10] >= 0.5)   # CV2 in-zone flag (NEW)
            in_zone = (cv1_in_zone and cv2_in_zone) 
            if not in_zone:
                mask_vals.append(np.zeros(self.action_size, dtype=np.float32))
            else:
                m = np.zeros(self.action_size, dtype=np.float32)
                for idx, (A, _, _, _) in enumerate(self.action_map):
                    if A > config.IN_ZONE_MAX_AMP:
                        m[idx] = -1e9
                mask_vals.append(m)
        mask = torch.tensor(np.stack(mask_vals), dtype=logits.dtype, device=logits.device)
        return logits + mask

    def act(self, state, training=True):
        s_np = np.asarray(state, dtype=np.float32)
        if s_np.ndim == 1: s_np = s_np[None, :]
        if self.obs_rms.mean is None: self.obs_rms.update(s_np)
        norm_np = self.obs_rms.normalize(s_np).astype(np.float32)
        norm_state = torch.from_numpy(norm_np)
        raw_state = torch.from_numpy(s_np.astype(np.float32))

        with torch.no_grad():
            logits = self.actor.forward_logits(norm_state)
            logits = self._mask_logits_if_needed(logits, raw_state)
            probs = F.softmax(logits, dim=-1)
            value = self.critic(norm_state).item()

        if torch.isnan(probs).any() or torch.isinf(probs).any():
            probs = torch.ones_like(probs) / self.action_size

        if not training and config.EVAL_GREEDY:
            action = torch.argmax(probs, dim=-1)
            log_prob = torch.log(probs.gather(1, action.view(-1, 1)).squeeze(1) + 1e-8)
        else:
            if training and self.exploration_noise > self.min_exploration_noise and not config.FREEZE_EXPLORATION_AT_ZONE:
                noise = torch.randn_like(probs) * self.exploration_noise
                probs = F.softmax(torch.log(probs + 1e-8) + noise, dim=-1)
            dist = torch.distributions.Categorical(probs)
            action = dist.sample(); log_prob = dist.log_prob(action)

        return int(action.item()), float(log_prob.item()), float(value)

    def save_experience(self, s, a, logp, v, r, done, ns):
        self.memory.append((s, a, logp, v, r, done, ns))

    def compute_advantages(self):
        if not self.memory: return None
        filtered_memory = []
        for m in self.memory:
            s, a, logp, v, r, done, ns = m
            if not np.isfinite(np.asarray(s, dtype=np.float32)).all():
                continue
            if not np.isfinite(np.asarray(ns, dtype=np.float32)).all():
                continue
            if not np.isfinite([float(logp), float(v), float(r)]).all():
                continue
            filtered_memory.append(m)

        if len(filtered_memory) < config.N_STEPS:
            print("[warn] skipping PPO update because replay batch contains invalid transitions.")
            self.memory = []
            return None

        states = np.array([m[0] for m in filtered_memory], dtype=np.float32)
        actions = torch.tensor([m[1] for m in filtered_memory], dtype=torch.long)
        old_log_probs = torch.tensor([m[2] for m in filtered_memory], dtype=torch.float32)
        values = torch.tensor([m[3] for m in filtered_memory], dtype=torch.float32)
        rewards = torch.tensor([m[4] for m in filtered_memory], dtype=torch.float32)
        dones = torch.tensor([m[5] for m in filtered_memory], dtype=torch.float32)

        self.obs_rms.update(states)
        norm_states = torch.from_numpy(self.obs_rms.normalize(states)).float()

        if filtered_memory[-1][5]:
            next_value = 0.0
        else:
            next_state = np.asarray(filtered_memory[-1][6], dtype=np.float32)
            if next_state.ndim == 1: next_state = next_state[None, :]
            self.obs_rms.update(next_state)
            ns = self.obs_rms.normalize(next_state).astype(np.float32)
            with torch.no_grad():
                next_value = self.critic(torch.from_numpy(ns)).item()

        advantages = torch.zeros_like(rewards)
        returns = torch.zeros_like(rewards)
        gae = 0.0
        for t in reversed(range(len(rewards))):
            next_v = next_value if t == len(rewards)-1 else values[t+1]
            next_done = dones[t] if t == len(rewards)-1 else dones[t+1]
            delta = rewards[t] + config.GAMMA * next_v * (1 - next_done) - values[t]
            gae = delta + config.GAMMA * config.GAE_LAMBDA * (1 - dones[t]) * gae
            advantages[t] = gae; returns[t] = gae + values[t]

        raw_states = torch.from_numpy(states).float()
        return raw_states, norm_states, actions, old_log_probs, advantages, returns

    def update(self):
        if len(self.memory) < config.N_STEPS: return {}
        data = self.compute_advantages()
        if data is None: return {}
        raw_states, states, actions, old_log_probs, advantages, returns = data
        if (not torch.isfinite(states).all()
            or not torch.isfinite(old_log_probs).all()
            or not torch.isfinite(advantages).all()
            or not torch.isfinite(returns).all()):
            print("[warn] skipping PPO update because normalized rollout tensors are not finite.")
            self.memory = []
            return {}
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        N = len(self.memory)
        metrics = {'loss':0.0,'actor_loss':0.0,'critic_loss':0.0,'entropy':0.0,
                   'approx_kl':0.0,'clip_frac':0.0,'updates':0}

        for _ in range(config.N_EPOCHS):
            perm = torch.randperm(N)
            for i in range(0, N, config.BATCH_SIZE):
                idx = perm[i:i+config.BATCH_SIZE]
                if len(idx) < 2: continue
                b_states = states[idx]; b_actions = actions[idx]
                b_old = old_log_probs[idx]; b_adv = advantages[idx]; b_ret = returns[idx]
                raw_b_states = raw_states[idx]

                logits = self.actor.forward_logits(b_states)
                logits = self._mask_logits_if_needed(logits, raw_b_states)
                if not torch.isfinite(logits).all():
                    print("[warn] skipping minibatch because actor logits are not finite.")
                    continue
                dist = torch.distributions.Categorical(logits=logits)
                new_logp = dist.log_prob(b_actions)
                entropy = dist.entropy().mean()

                ratios = torch.exp(new_logp - b_old)
                surr1 = ratios * b_adv
                surr2 = torch.clamp(ratios, 1 - config.CLIP_RANGE, 1 + config.CLIP_RANGE) * b_adv
                actor_loss = -torch.min(surr1, surr2).mean()

                values = self.critic(b_states).squeeze()
                critic_loss = F.mse_loss(values, b_ret)

                loss = actor_loss + config.VF_COEF * critic_loss - config.ENT_COEF * entropy
                if not torch.isfinite(loss):
                    print("[warn] skipping minibatch because PPO loss is not finite.")
                    continue

                approx_kl = (b_old - new_logp).mean().item()
                clip_frac = (torch.abs(ratios - 1.0) > config.CLIP_RANGE).float().mean().item()

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), config.MAX_GRAD_NORM)
                nn.utils.clip_grad_norm_(self.critic.parameters(), config.MAX_GRAD_NORM)
                self.optimizer.step()

                metrics['loss'] += loss.item()
                metrics['actor_loss'] += actor_loss.item()
                metrics['critic_loss'] += critic_loss.item()
                metrics['entropy'] += entropy.item()
                metrics['approx_kl'] += approx_kl
                metrics['clip_frac'] += clip_frac
                metrics['updates'] += 1

                if approx_kl > config.PPO_TARGET_KL: break

        self.scheduler.step()
        self.exploration_noise = max(self.min_exploration_noise,
                                     self.exploration_noise * self.exploration_decay)
        self.memory = []
        self.episode_count += 1

        if metrics['updates'] > 0:
            for k in list(metrics.keys()):
                if k not in ('updates',): metrics[k] /= metrics['updates']
        metrics['lr'] = float(self.optimizer.param_groups[0]['lr'])
        return metrics

    def obs_norm_state(self): return self.obs_rms.state_dict()
    def load_obs_norm_state(self, d): self.obs_rms.load_state_dict(d)

# ========================= Environment helpers =========================
def _phase2_bowl_reward(d, center, half_width, gain):
    err = abs(d - center)
    if err >= half_width:
        return 0.0
    scale = 1.0 - (err / half_width) ** 2
    return gain * scale

def _nan_safe_propagate(
    sim: Simulation,
    nsteps: int,
    dcdfreq: int,
    prop_index: int,
    atom_pairs=None,            # [(a1,a2),(b1,b2)]
    out_cv1=None,               # list to append CV1 samples (Å)
    out_cv2=None,               # list to append CV2 samples (Å)
    out_times_ps=None,          # list to append times (ps)
    dt_min=u.picoseconds * 0.0005,
):
    """Run nsteps in dcdfreq chunks with NaN recovery and a tqdm progress bar.

    We step in chunks (simulation.step(dcdfreq)) so reporters fire consistently and we can show progress.
    """
    integ = sim.integrator
    orig_dt = integ.getStepSize()
    dt = orig_dt

    # ---- Parity with combined.py: local minimization + velocity re-init ----
    try:
        openmm.LocalEnergyMinimizer.minimize(
            sim.context, 10.0 * u.kilojoule_per_mole, 200
        )
    except Exception:
        pass
    try:
        sim.context.setVelocitiesToTemperature(T)
    except Exception:
        pass

    # number of chunks (at least 1)
    n_chunks = max(1, int(nsteps // max(1, int(dcdfreq))))
    chunk_steps = int(dcdfreq)
    t_ps = 0.0
    need_positions = True

    retries_left = int(MAX_INTEGRATOR_RETRIES)
    completed_chunks = 0
    failed = False

    for _ in tqdm(
        range(n_chunks),
        desc=f"MD Step {prop_index:>2d}",
        colour="red",
        ncols=80,
    ):
        try:
            sim.step(chunk_steps)
            st = sim.context.getState(getPositions=need_positions)
            pos_nm = st.getPositions(asNumpy=True).value_in_unit(u.nanometer)
            if not np.isfinite(np.asarray(pos_nm)).all():
                raise openmm.OpenMMException("NaN positions detected")
        except Exception:
            if retries_left <= 0:
                failed = True
                break
            retries_left -= 1
            dt = max(dt * 0.5, dt_min)
            try:
                integ.setStepSize(dt)
            except Exception:
                failed = True
                break
            # reinit velocities to recover
            try:
                sim.context.setVelocitiesToTemperature(T)
            except Exception:
                pass
            # retry once
            try:
                sim.step(chunk_steps)
                st = sim.context.getState(getPositions=need_positions)
                pos_nm = st.getPositions(asNumpy=True).value_in_unit(u.nanometer)
                if not np.isfinite(np.asarray(pos_nm)).all():
                    raise openmm.OpenMMException("NaN positions after recovery")
            except Exception:
                failed = True
                break

        t_ps += float(chunk_steps) * float(orig_dt.value_in_unit(u.picoseconds))
        completed_chunks += 1
        if out_times_ps is not None:
            out_times_ps.append(float(t_ps))

        # --- record live CVs at the same cadence as CV1 plotting (every chunk) ---
        if atom_pairs is not None and (out_cv1 is not None or out_cv2 is not None):
            # CV1
            (a1, a2) = atom_pairs[0]
            d1_A = np.linalg.norm(pos_nm[a1] - pos_nm[a2]) * 10.0
            if out_cv1 is not None:
                out_cv1.append(float(d1_A))

            # CV2
            (b1, b2) = atom_pairs[1]
            d2_A = np.linalg.norm(pos_nm[b1] - pos_nm[b2]) * 10.0
            if out_cv2 is not None:
                out_cv2.append(float(d2_A))

    # restore original dt
    try:
        integ.setStepSize(orig_dt)
    except Exception:
        pass
    return {
        "failed": bool(failed),
        "completed_chunks": int(completed_chunks),
        "requested_chunks": int(n_chunks),
    }

# ========================= ProteinEnvironmentRedesigned (2D) =========================
class ProteinEnvironmentRedesigned:
    """
    Same environment contract as 1D, but with *two* CV distances.

    - CV1 drives milestones/phase transitions (default)
    - CV2 supplies additional shaping and can be biased as well
    """

    def __init__(self):
        # resolve CV pairs
        if ATOM_PAIRS and len(ATOM_PAIRS) >= 2:
            (a1, a2), (b1, b2) = ATOM_PAIRS[0], ATOM_PAIRS[1]
        else:
            a1, a2 = ATOM1_INDEX, ATOM2_INDEX
            b1, b2 = ATOM3_INDEX, ATOM4_INDEX

        if b1 is None or b2 is None:
            raise ValueError(
                "2D CV requires ATOM3_INDEX and ATOM4_INDEX (or ATOM_PAIRS with >=2 pairs)."
            )

        self.atom1_idx, self.atom2_idx = int(a1), int(a2)
        self.atom3_idx, self.atom4_idx = int(b1), int(b2)

        self.platform = None  # lazy init
        self.psf = None
        self.pdb = None
        self.params = None

        # pre-load topology/system XML once
        self._load_and_serialize_base_system()

        # episode / milestone bookkeeping
        self.milestones_hit = set()
        self.in_zone_count = 0

        # histories
        self.distance_history = SliceableDeque(maxlen=2000)
        self.distance2_history = SliceableDeque(maxlen=2000)

        self.bias_log = []
        self.all_biases_in_episode = []

        self.step_counter = 0

        # Phase & progress
        self.phase = 1
        self.in_zone_steps = 0
        self.no_improve_counter = 0
        self.episode_target_center_A = float(TARGET_CENTER)
        self.episode_target_half_width_A = float(TARGET_ZONE_HALF_WIDTH)
        self.episode_target_min_A = float(TARGET_MIN)
        self.episode_target_max_A = float(TARGET_MAX)
        self.episode_target_hit = False
        self.episode_target_stage = None
        self.last_step_md_failed = False
        self.episode_md_failed = False

        # Locks
        self.locked_milestone_idx = -1
        self.backstops_A = []
        self.best_distance_ever = float(CURRENT_DISTANCE)
        self.best_distance2_ever = float(CURRENT_DISTANCE2 if CURRENT_DISTANCE2 is not None else FINAL_TARGET2)
        self.best_positions = None

        # MD state
        self.current_positions = None
        self.current_velocities = None

        # Zone walls cleared (CV1)
        self.zone_floor_A = None
        self.zone_ceiling_A = None

        # Zone walls cleared (CV2)
        self.zone2_floor_A = None
        self.zone2_ceiling_A = None

        # caches
        self.simulation = None
        self._last_topology = None
        self._last_positions = None

        # DCD bookkeeping
        self.current_episode_index = None
        self.current_dcd_index = 0
        self.current_dcd_paths = []
        self.current_run_name = None

        # dynamic force bookkeeping for persistent per-episode simulation
        self._dynamic_force_names = {}
        self._bias_slots_used = 0
        self._bias_primary_frontier_A = None

        # targets for CV2 (resolved at reset from structure if needed)
        self._cv2_center = None
        self.cv2_center_on = False

        # per-episode live trajectories (sampled every dcdfreq_mfpt chunk)
        self.episode_trajectory_segments = []          # CV1 segments (already used by main.py)
        self.episode_trajectory_segments_cv2 = []      # CV2 segments (NEW)

        # initialize current distances from the start structure
        self.reset(seed_from_max_A=None, carry_state=False, episode_index=None)

    def _load_and_serialize_base_system(self):
        # do not create platform or Simulation here (import-safe)
        print("Setting up protein MD system...")
        psf = CharmmPsfFile(psf_file)
        pdb = PDBFile(pdb_file)
        params = load_charmm_params(toppar_file)

        # Match combined.py behavior: non-periodic system setup
        system = psf.createSystem(
            params,
            nonbondedMethod=CutoffNonPeriodic,
            nonbondedCutoff=nonbondedCutoff,
            constraints=None,
        )

        # ---- Backbone positional restraints (parity with combined.py) ----
        add_backbone_posres(
            system,
            psf,
            pdb,
            backbone_constraint_strength,
            skip_indices={self.atom1_idx, self.atom2_idx, self.atom3_idx, self.atom4_idx},
        )

        self.psf = psf
        self.pdb = pdb
        self.params = params
        self.base_system_xml = openmm.XmlSerializer.serialize(system)
        print("Protein MD system setup complete.")

    # ---------- CV accessors ----------
    def _current_distances_A(self):
        """Compute (d1_A, d2_A) from the current context/positions."""
        if self.simulation is None:
            # construct a temporary context to measure from pdb positions
            plat = self.platform or get_best_platform(verbose=False)
            system = openmm.XmlSerializer.deserialize(self.base_system_xml)
            integ = openmm.LangevinIntegrator(T, fricCoef, stepsize)
            sim = Simulation(self.psf.topology, system, integ, plat)
            sim.context.setPositions(self._last_positions if self._last_positions is not None else self.pdb.positions)
            state = sim.context.getState(getPositions=True)
            pos = state.getPositions(asNumpy=True).value_in_unit(u.nanometer)
        else:
            state = self.simulation.context.getState(getPositions=True)
            pos = state.getPositions(asNumpy=True).value_in_unit(u.nanometer)

        return self._distances_from_positions_nm(pos)

    def _distances_from_positions_nm(self, pos_nm):
        p = np.asarray(pos_nm)
        d1 = np.linalg.norm(p[self.atom1_idx] - p[self.atom2_idx]) * 10.0
        d2 = np.linalg.norm(p[self.atom3_idx] - p[self.atom4_idx]) * 10.0
        return float(d1), float(d2)

    def _validated_distances_from_positions(self, positions):
        try:
            pos_nm = positions.value_in_unit(u.nanometer)
        except Exception:
            pos_nm = np.asarray(positions, dtype=np.float64)

        pos_nm = np.asarray(pos_nm, dtype=np.float64)
        if pos_nm.ndim != 2 or pos_nm.shape[1] != 3:
            return None, None
        if not np.isfinite(pos_nm).all():
            return None, None

        d1, d2 = self._distances_from_positions_nm(pos_nm)
        if not np.isfinite([d1, d2]).all():
            return None, None
        return float(d1), float(d2)

    def _set_episode_cv1_target(self, target_center_A=None, target_half_width_A=None, target_stage=None):
        center = float(TARGET_CENTER if target_center_A is None else target_center_A)
        half_width = float(TARGET_ZONE_HALF_WIDTH if target_half_width_A is None else target_half_width_A)
        self.episode_target_center_A = center
        self.episode_target_half_width_A = half_width
        self.episode_target_min_A = center - half_width
        self.episode_target_max_A = center + half_width
        self.episode_target_stage = None if target_stage is None else int(target_stage)

    def current_target_zone(self):
        return (
            float(self.episode_target_center_A),
            float(self.episode_target_min_A),
            float(self.episode_target_max_A),
            float(self.episode_target_half_width_A),
        )

    def final_target_zone(self):
        return (
            float(TARGET_CENTER),
            float(TARGET_MIN),
            float(TARGET_MAX),
            float(TARGET_ZONE_HALF_WIDTH),
        )

    # ---------- State ----------
    def get_state(self):
        # normalize targets
        d1 = self.current_distance
        d2 = self.current_distance2
        target_center_A, target_min_A, target_max_A, target_half_width_A = self.final_target_zone()

        self.distance_history.append(d1)
        self.distance2_history.append(d2)

        # trends
        if len(self.distance_history) >= 3:
            trend1 = (self.distance_history[-1] - self.distance_history[-3]) / 2.0
        else:
            trend1 = 0.0
        if len(self.distance2_history) >= 3:
            trend2 = (self.distance2_history[-1] - self.distance2_history[-3]) / 2.0
        else:
            trend2 = 0.0

        stability = 0.5
        if len(self.distance_history) >= 5:
            stability = 1.0 / (1.0 + np.std(list(self.distance_history)[-5:]))

        # --- anchored progress for BOTH CVs ---
        p1 = (d1 - CURRENT_DISTANCE) / max(1e-6, (target_center_A - CURRENT_DISTANCE))  # CV1 outward
        p1 = float(np.clip(p1, 0.0, 1.0))

        # CV2 inward (distance reduction): progress increases as d2 decreases
        den2 = max(1e-6, (CURRENT_DISTANCE2 - FINAL_TARGET2))
        p2 = (CURRENT_DISTANCE2 - d2) / den2
        p2 = float(np.clip(p2, 0.0, 1.0))

        # Stage-1 training is driven by CV1 unbinding.
        overall = p1

        # CV2 zone indicator
        in_cv2_zone = 0.0
        if self._cv2_min is not None and self._cv2_max is not None:
            in_cv2_zone = float(self._cv2_min <= d2 <= self._cv2_max)

        state = np.array(
            [
                d1 / max(1e-6, FINAL_TARGET),                                # 0
                max(0.0, overall),                                           # 1
                abs(d1 - target_center_A) / max(1e-6, target_half_width_A),  # 2
                np.clip(overall, 0.0, 1.0),                                  # 3
                trend1 / 0.1,                                                # 4
                float(target_min_A <= d1 <= target_max_A),                   # 5
                float(self.no_improve_counter > 0),                          # 6
                stability,                                                   # 7
                (d2 / max(1e-6, (self._cv2_center or max(1e-6, d2)))),       # 8
                trend2 / 0.1,                                                # 9
                in_cv2_zone,                                                 # 10
            ],
            dtype=np.float32,
        )
        return state

    # ---------- Forces ----------
    def _add_gaussian_force_1cv(self, system, amplitude_kcal, center_A, width_A, atom_i, atom_j):
        """
        Add ONE 1D Gaussian bias on the distance between (atom_i, atom_j).
        amplitude_kcal is interpreted in kcal/mol (converted to kJ/mol internally).
        center_A and width_A are in Å.
        """
        uid = str(uuid.uuid4())[:8]
        A_name = f"A_{uid}"
        mu_name = f"mu_{uid}"
        sig_name = f"sigma_{uid}"

        expr = f"{A_name}*exp(-((r-{mu_name})^2)/(2*{sig_name}^2))"
        cf = openmm.CustomBondForce(expr)
        cf.addGlobalParameter(A_name, float(amplitude_kcal) * 4.184)
        cf.addGlobalParameter(mu_name, float(center_A) / 10.0)
        cf.addGlobalParameter(sig_name, max(1e-6, float(width_A) / 10.0))
        cf.addBond(int(atom_i), int(atom_j))
        system.addForce(cf)
        return system

    def _add_gaussian_biases(self, system, atom_i, atom_j, biases):
        """
        Add MANY 1D Gaussian biases on the same distance as per-bond parameters.
        This avoids CUDA kernel parameter overflow from too many global parameters.
        """
        if not biases:
            return system

        expr = "A*exp(-((r-mu)^2)/(2*sigma^2))"
        cf = openmm.CustomBondForce(expr)
        cf.addPerBondParameter("A")
        cf.addPerBondParameter("mu")
        cf.addPerBondParameter("sigma")

        for amp_kcal, center_A, width_A in biases:
            cf.addBond(
                int(atom_i),
                int(atom_j),
                [
                    float(amp_kcal) * 4.184,
                    float(center_A) / 10.0,
                    max(1e-6, float(width_A) / 10.0),
                ],
            )

        system.addForce(cf)
        return system

    def _add_gaussian_bias_2d(self, system, amp_kcal, center1_A, center2_A, sigma_x_A, sigma_y_A):
        cv1 = openmm.CustomBondForce("r")
        cv1.addBond(self.atom1_idx, self.atom2_idx)
        cv2 = openmm.CustomBondForce("r")
        cv2.addBond(self.atom3_idx, self.atom4_idx)

        uid = str(uuid.uuid4())[:8]
        A_name = f"A_{uid}"
        x0_name = f"x0_{uid}"
        y0_name = f"y0_{uid}"
        sx_name = f"sx_{uid}"
        sy_name = f"sy_{uid}"

        expr = f"{A_name}*exp(-((cv1-{x0_name})^2)/(2*{sx_name}^2) - ((cv2-{y0_name})^2)/(2*{sy_name}^2))"
        force = openmm.CustomCVForce(expr)
        force.addCollectiveVariable("cv1", cv1)
        force.addCollectiveVariable("cv2", cv2)
        force.addGlobalParameter(A_name, float(amp_kcal) * 4.184)
        force.addGlobalParameter(x0_name, float(center1_A) / 10.0)
        force.addGlobalParameter(y0_name, float(center2_A) / 10.0)
        force.addGlobalParameter(sx_name, max(1e-6, float(sigma_x_A) / 10.0))
        force.addGlobalParameter(sy_name, max(1e-6, float(sigma_y_A) / 10.0))
        system.addForce(force)
        return system

    def _add_gaussian_biases_2d(self, system, biases):
        if not biases:
            return system
        for amp, c1, c2, sx, sy in biases:
            system = self._add_gaussian_bias_2d(system, amp, c1, c2, sx, sy)
        return system
    
    def _add_backstop_force(self, system, m_eff_A):
        uid = str(uuid.uuid4())[:8]
        kname = f"k_back_{uid}"
        mname = f"m_{uid}"
        expr = f"{kname}*({mname} - r)^2*step({mname} - r)"
        cf = openmm.CustomBondForce(expr)
        cf.addGlobalParameter(kname, float(BACKSTOP_K))
        cf.addGlobalParameter(mname, float(m_eff_A) / 10.0)
        cf.addBond(self.atom1_idx, self.atom2_idx)
        system.addForce(cf)
        return system

    def _add_zone_upper_cap(self, system, u_eff_A):
        uid = str(uuid.uuid4())[:8]
        kname = f"k_cap_{uid}"
        uname = f"u_{uid}"
        expr = f"{kname}*(r - {uname})^2*step(r - {uname})"
        cf = openmm.CustomBondForce(expr)
        cf.addGlobalParameter(kname, float(ZONE_K))
        cf.addGlobalParameter(uname, float(u_eff_A) / 10.0)
        cf.addBond(self.atom1_idx, self.atom2_idx)
        system.addForce(cf)
        return system

    def _add_zone_lower_cap(self, system, l_eff_A):
        uid = str(uuid.uuid4())[:8]
        kname = f"k_floor_{uid}"
        lname = f"l_{uid}"
        expr = f"{kname}*({lname} - r)^2*step({lname} - r)"
        cf = openmm.CustomBondForce(expr)
        cf.addGlobalParameter(kname, float(ZONE_K))
        cf.addGlobalParameter(lname, float(l_eff_A) / 10.0)
        cf.addBond(self.atom1_idx, self.atom2_idx)
        system.addForce(cf)
        return system
    
    def _add_zone_upper_cap_cv2(self, system, u_eff_A):
        uid = str(uuid.uuid4())[:8]
        kname = f"k2_cap_{uid}"
        uname = f"u2_{uid}"
        expr = f"{kname}*(r - {uname})^2*step(r - {uname})"
        cf = openmm.CustomBondForce(expr)
        cf.addGlobalParameter(kname, float(CV2_ZONE_K))
        cf.addGlobalParameter(uname, float(u_eff_A) / 10.0)
        cf.addBond(self.atom3_idx, self.atom4_idx)
        system.addForce(cf)
        return system

    def _add_zone_lower_cap_cv2(self, system, l_eff_A):
        uid = str(uuid.uuid4())[:8]
        kname = f"k2_floor_{uid}"
        lname = f"l2_{uid}"
        expr = f"{kname}*({lname} - r)^2*step({lname} - r)"
        cf = openmm.CustomBondForce(expr)
        cf.addGlobalParameter(kname, float(CV2_ZONE_K))
        cf.addGlobalParameter(lname, float(l_eff_A) / 10.0)
        cf.addBond(self.atom3_idx, self.atom4_idx)
        system.addForce(cf)
        return system
    
    def _add_center_harmonic_cv2(self, system, center_A):
        uid = str(uuid.uuid4())[:8]
        kname = f"k2_center_{uid}"
        cname = f"c2_{uid}"
        expr = f"{kname}*(r - {cname})^2"
        cf = openmm.CustomBondForce(expr)
        cf.addGlobalParameter(kname, float(CV2_CENTER_K))
        cf.addGlobalParameter(cname, float(center_A) / 10.0)
        cf.addBond(self.atom3_idx, self.atom4_idx)
        system.addForce(cf)
        return system

    def _add_persistent_bias_force_2d(self, system, max_terms):
        cv1 = openmm.CustomBondForce("r")
        cv1.addBond(self.atom1_idx, self.atom2_idx)
        cv2 = openmm.CustomBondForce("r")
        cv2.addBond(self.atom3_idx, self.atom4_idx)

        term_exprs = []
        bias_names = []
        for idx in range(int(max_terms)):
            names = {
                "amp": f"bias_A_{idx}",
                "x0": f"bias_x0_{idx}",
                "y0": f"bias_y0_{idx}",
                "sx": f"bias_sx_{idx}",
                "sy": f"bias_sy_{idx}",
            }
            bias_names.append(names)
            term_exprs.append(
                f"{names['amp']}*exp(-((cv1-{names['x0']})^2)/(2*{names['sx']}^2) - ((cv2-{names['y0']})^2)/(2*{names['sy']}^2))"
            )

        force = openmm.CustomCVForce(" + ".join(term_exprs) if term_exprs else "0")
        force.addCollectiveVariable("cv1", cv1)
        force.addCollectiveVariable("cv2", cv2)

        for names in bias_names:
            force.addGlobalParameter(names["amp"], 0.0)
            force.addGlobalParameter(names["x0"], 0.0)
            force.addGlobalParameter(names["y0"], 0.0)
            force.addGlobalParameter(names["sx"], 1.0)
            force.addGlobalParameter(names["sy"], 1.0)

        system.addForce(force)
        self._dynamic_force_names["bias2d"] = bias_names
        return system

    def _add_persistent_backstop_force(self, system, max_terms):
        names_list = []
        expr_terms = []
        for idx in range(int(max_terms)):
            names = {
                "k": f"back_k_{idx}",
                "m": f"back_m_{idx}",
            }
            names_list.append(names)
            expr_terms.append(f"{names['k']}*({names['m']} - r)^2*step({names['m']} - r)")

        force = openmm.CustomBondForce(" + ".join(expr_terms) if expr_terms else "0")
        for names in names_list:
            force.addGlobalParameter(names["k"], 0.0)
            force.addGlobalParameter(names["m"], 0.0)
        force.addBond(self.atom1_idx, self.atom2_idx)
        system.addForce(force)
        self._dynamic_force_names["backstops"] = names_list
        return system

    def _add_switchable_cap_force(self, system, name_prefix, atom_i, atom_j, direction, default_k):
        kname = f"{name_prefix}_k"
        lname = f"{name_prefix}_limit"
        if direction == "upper":
            expr = f"{kname}*(r - {lname})^2*step(r - {lname})"
        else:
            expr = f"{kname}*({lname} - r)^2*step({lname} - r)"

        force = openmm.CustomBondForce(expr)
        force.addGlobalParameter(kname, 0.0)
        force.addGlobalParameter(lname, 0.0)
        force.addBond(int(atom_i), int(atom_j))
        system.addForce(force)
        self._dynamic_force_names[name_prefix] = {"k": kname, "limit": lname, "default_k": float(default_k)}
        return system

    def _add_switchable_center_force(self, system, name_prefix, atom_i, atom_j, default_k):
        kname = f"{name_prefix}_k"
        cname = f"{name_prefix}_center"
        force = openmm.CustomBondForce(f"{kname}*(r - {cname})^2")
        force.addGlobalParameter(kname, 0.0)
        force.addGlobalParameter(cname, 0.0)
        force.addBond(int(atom_i), int(atom_j))
        system.addForce(force)
        self._dynamic_force_names[name_prefix] = {"k": kname, "center": cname, "default_k": float(default_k)}
        return system

    def _build_episode_system(self):
        self._dynamic_force_names = {}
        system = openmm.XmlSerializer.deserialize(self.base_system_xml)
        system = self._add_persistent_backstop_force(system, len(DISTANCE_INCREMENTS))
        system = self._add_switchable_cap_force(system, "zone1_lower", self.atom1_idx, self.atom2_idx, "lower", ZONE_K)
        system = self._add_switchable_cap_force(system, "zone1_upper", self.atom1_idx, self.atom2_idx, "upper", ZONE_K)
        system = self._add_switchable_cap_force(system, "zone2_lower", self.atom3_idx, self.atom4_idx, "lower", CV2_ZONE_K)
        system = self._add_switchable_cap_force(system, "zone2_upper", self.atom3_idx, self.atom4_idx, "upper", CV2_ZONE_K)
        system = self._add_switchable_center_force(system, "cv2_center", self.atom3_idx, self.atom4_idx, CV2_CENTER_K)
        system = self._add_persistent_bias_force_2d(system, MAX_BIAS_TERMS_PER_EPISODE)
        return system

    def _set_context_parameter(self, name, value):
        if self.simulation is None:
            return
        self.simulation.context.setParameter(str(name), float(value))

    def _sync_backstop_parameters(self):
        names_list = self._dynamic_force_names.get("backstops", [])
        active = list(self.backstops_A if ENABLE_MILESTONE_LOCKS else [])
        for idx, names in enumerate(names_list):
            if idx < len(active):
                self._set_context_parameter(names["k"], float(BACKSTOP_K))
                self._set_context_parameter(names["m"], float(active[idx]) / 10.0)
            else:
                self._set_context_parameter(names["k"], 0.0)
                self._set_context_parameter(names["m"], 0.0)

    def _sync_cap_parameters(self, key, enabled, limit_A):
        names = self._dynamic_force_names.get(key)
        if not names:
            return
        if enabled and limit_A is not None:
            self._set_context_parameter(names["k"], names["default_k"])
            self._set_context_parameter(names["limit"], float(limit_A) / 10.0)
        else:
            self._set_context_parameter(names["k"], 0.0)
            self._set_context_parameter(names["limit"], 0.0)

    def _sync_center_parameters(self):
        names = self._dynamic_force_names.get("cv2_center")
        if not names:
            return
        if CV2_CENTER_RESTRAINT and getattr(self, "cv2_center_on", False) and self._cv2_center is not None:
            self._set_context_parameter(names["k"], names["default_k"])
            self._set_context_parameter(names["center"], float(self._cv2_center) / 10.0)
        else:
            self._set_context_parameter(names["k"], 0.0)
            self._set_context_parameter(names["center"], 0.0)

    def _sync_bias_parameters(self):
        slots = self._dynamic_force_names.get("bias2d", [])
        active = list(self.all_biases_in_episode)
        for idx, names in enumerate(slots):
            if idx < len(active):
                amp, c1, c2, sx, sy = active[idx]
                self._set_context_parameter(names["amp"], float(amp) * 4.184)
                self._set_context_parameter(names["x0"], float(c1) / 10.0)
                self._set_context_parameter(names["y0"], float(c2) / 10.0)
                self._set_context_parameter(names["sx"], max(1e-6, float(sx) / 10.0))
                self._set_context_parameter(names["sy"], max(1e-6, float(sy) / 10.0))
            else:
                self._set_context_parameter(names["amp"], 0.0)
                self._set_context_parameter(names["x0"], 0.0)
                self._set_context_parameter(names["y0"], 0.0)
                self._set_context_parameter(names["sx"], 1.0)
                self._set_context_parameter(names["sy"], 1.0)

    def _sync_dynamic_force_parameters(self):
        if self.simulation is None:
            return
        self._sync_backstop_parameters()
        self._sync_cap_parameters("zone1_lower", ZONE_CONFINEMENT, self.zone_floor_A)
        self._sync_cap_parameters("zone1_upper", ZONE_CONFINEMENT, self.zone_ceiling_A)
        self._sync_cap_parameters("zone2_lower", CV2_ZONE_CONFINEMENT, self.zone2_floor_A)
        self._sync_cap_parameters("zone2_upper", CV2_ZONE_CONFINEMENT, self.zone2_ceiling_A)
        self._sync_center_parameters()
        self._sync_bias_parameters()

    def _build_episode_simulation(self):
        gamma = fricCoef
        if getattr(self, "cv2_center_on", False) or (self.phase == 2):
            gamma = 10.0 / u.picoseconds

        system = self._build_episode_system()
        integrator = openmm.LangevinIntegrator(T, gamma, stepsize)
        sim = Simulation(self.psf.topology, system, integrator, self.platform)
        self.simulation = sim
        self._last_topology = self.psf.topology

        if self._last_positions is not None:
            sim.context.setPositions(self._last_positions)
        else:
            sim.context.setPositions(self.pdb.positions)

        self._sync_dynamic_force_parameters()
        return sim

    def _system_with_all_forces(self):
        system = openmm.XmlSerializer.deserialize(self.base_system_xml)

        if ENABLE_MILESTONE_LOCKS:
            for m_eff_A in self.backstops_A:
                system = self._add_backstop_force(system, m_eff_A)

        if ZONE_CONFINEMENT:
            if self.zone_floor_A is not None:
                system = self._add_zone_lower_cap(system, self.zone_floor_A)
            if self.zone_ceiling_A is not None:
                system = self._add_zone_upper_cap(system, self.zone_ceiling_A)
        
        if CV2_ZONE_CONFINEMENT:
            if self.zone2_floor_A is not None:
                system = self._add_zone_lower_cap_cv2(system, self.zone2_floor_A)
            if self.zone2_ceiling_A is not None:
                system = self._add_zone_upper_cap_cv2(system, self.zone2_ceiling_A)

        if CV2_CENTER_RESTRAINT and getattr(self, "cv2_center_on", False):
            system = self._add_center_harmonic_cv2(system, self._cv2_center)

        bias_2d = list(self.all_biases_in_episode)
        if bias_2d:
            system = self._add_gaussian_biases_2d(system, bias_2d)

        return system

    # ---------- action → 2D Gaussian parameters ----------
    def smart_progressive_bias(self, action: int):
        target_center_A, target_min_A, target_max_A, _ = self.final_target_zone()
        A_bins = AMP_BINS
        dx_bins = DX_BINS
        dy_bins = DY_BINS
        s_bins = SIGMA_BINS

        n_dx = len(dx_bins)
        n_dy = len(dy_bins)
        n_s = len(s_bins)
        n_total = len(A_bins) * n_dx * n_dy * n_s

        action = int(max(0, min(action, n_total - 1)))
        amp_idx = action // (n_dx * n_dy * n_s)
        rem = action % (n_dx * n_dy * n_s)
        dx_idx = rem // (n_dy * n_s)
        rem2 = rem % (n_dy * n_s)
        dy_idx = rem2 // n_s
        s_idx = rem2 % n_s

        base_amp = float(A_bins[amp_idx])
        base_dx = float(dx_bins[dx_idx])
        base_dy = float(dy_bins[dy_idx])
        base_sigma = float(s_bins[s_idx])
        dx_min = float(min(dx_bins))
        dx_max = float(max(dx_bins))
        dx_span = max(1e-6, dx_max - dx_min)
        forward_signal = float(np.clip((base_dx - dx_min) / dx_span, 0.0, 1.0))

        # Progress scaling (same logic as before)
        p1 = (self.current_distance - CURRENT_DISTANCE) / max(1e-6, (target_center_A - CURRENT_DISTANCE))
        p1 = float(np.clip(p1, 0.0, 1.0))

        den2 = max(1e-6, (CURRENT_DISTANCE2 - FINAL_TARGET2))
        p2 = (CURRENT_DISTANCE2 - self.current_distance2) / den2
        p2 = float(np.clip(p2, 0.0, 1.0))

        progress = p1
        remaining = max(0.0, float(target_center_A) - float(self.current_distance))
        target_span = max(1e-6, float(target_center_A) - float(CURRENT_DISTANCE))
        amp = base_amp * (3.0 - 2.0 * np.clip(progress, 0.0, 1.0))
        sigma = base_sigma * (1.5 - np.clip(progress, 0.0, 1.0))

        if self.no_improve_counter >= 2:
            escalation = min(1.0 + 0.7 * self.no_improve_counter, MAX_ESCALATION_FACTOR)
            amp *= escalation

        in_plateau = (
            float(STRONG_BIAS_PLATEAU_MIN_A)
            <= float(self.current_distance)
            <= float(STRONG_BIAS_PLATEAU_MAX_A)
        )
        if (
            self.no_improve_counter >= int(STRONG_BIAS_STALL_TRIGGER)
            and in_plateau
            and remaining >= float(STRONG_BIAS_REMAINING_MIN_A)
        ):
            plateau_boost = 1.0 + float(STRONG_BIAS_GAIN) * min(float(self.no_improve_counter), 3.0)
            amp *= plateau_boost

        amp = float(np.clip(amp, MIN_AMP, min(MAX_AMP, APPLIED_MAX_AMP)))
        sigma = float(np.clip(sigma, MIN_WIDTH, MAX_WIDTH))

        in_zone1 = target_min_A <= self.current_distance <= target_max_A
        if in_zone1 or amp <= 0.0:
            return []

        stall_level = max(0.0, float(self.no_improve_counter) - 1.0)

        near_offset = (
            float(RIBBON_CV1_NEAR_OFFSET_A)
            + float(RIBBON_CV1_NEAR_OFFSET_SCALE) * forward_signal
        )
        sigma_target_scale = float(PRIMARY_OFFSET_SIGMA_SCALE_PLATEAU if in_plateau else PRIMARY_OFFSET_SIGMA_SCALE)
        sigma_target_offset = sigma * (
            sigma_target_scale
            + 0.10 * forward_signal
            + 0.04 * min(stall_level, 3.0)
        )
        near_offset = float(max(near_offset, sigma_target_offset))
        span_offset = (
            float(RIBBON_CV1_SPAN_A)
            + float(RIBBON_CV1_SPAN_SCALE) * forward_signal
            + float(RIBBON_CV1_STALL_SPAN_SCALE) * stall_level
            + 0.20 * min(1.0, remaining / target_span)
        )
        span_offset = float(np.clip(span_offset, 0.25, RIBBON_CV1_BACK_OFFSET_MAX_A))

        sigma_x_values = [
            float(sigma),
            float(np.clip(sigma * RIBBON_SECONDARY_SIGMA_X_SCALE, MIN_WIDTH, MAX_WIDTH)),
            float(np.clip(sigma * RIBBON_TERTIARY_SIGMA_X_SCALE, MIN_WIDTH, MAX_WIDTH)),
        ]
        sigma_y = float(np.clip(sigma * RIBBON_SIGMA_Y_SCALE, MIN_WIDTH, MAX_WIDTH))

        spread = float(RIBBON_CV2_SPREAD_A + RIBBON_CV2_SPREAD_SCALE * abs(float(base_dy)))
        spread = float(np.clip(spread, 0.25, 2.0))
        y_center = float(self.current_distance2 + (RIBBON_CV2_SHIFT_SCALE * float(base_dy)))

        max_center1 = max(0.52, float(self.current_distance) - 0.02)
        frontier_step = (
            float(BIAS_FRONTIER_STEP_A)
            + float(BIAS_FRONTIER_STALL_STEP_A) * min(stall_level, 3.0)
            + (float(BIAS_FRONTIER_PLATEAU_STEP_A) if in_plateau else 0.0)
        )
        primary_center = float(np.clip(self.current_distance - near_offset, 0.5, max_center1))
        if self._bias_primary_frontier_A is not None and remaining > 0.35:
            frontier_min = min(max_center1, float(self._bias_primary_frontier_A) + frontier_step)
            primary_center = float(np.clip(max(primary_center, frontier_min), 0.5, max_center1))

        secondary_offset = max(near_offset + 0.55 * span_offset, self.current_distance - primary_center + 0.20)
        tertiary_offset = max(near_offset + span_offset, secondary_offset + 0.25)
        x_offsets = [
            self.current_distance - primary_center,
            secondary_offset,
            tertiary_offset,
        ]
        y_offsets = [0.0, spread, -spread]
        amp_scales = [
            1.0,
            float(RIBBON_SECONDARY_AMP_FRACTION + 0.04 * min(stall_level, 2.0)),
            float(RIBBON_TERTIARY_AMP_FRACTION + 0.05 * min(stall_level, 2.0)),
        ]

        biases = []
        for idx in range(int(MAX_GAUSSIANS_PER_ACTION)):
            center1 = float(np.clip(self.current_distance - x_offsets[idx], 0.5, max_center1))
            center2 = float(np.clip(y_center + y_offsets[idx], 0.5, 50.0))
            amp_i = float(np.clip(amp * amp_scales[idx], MIN_AMP, min(MAX_AMP, APPLIED_MAX_AMP)))
            biases.append(
                (
                    "gaussian2d",
                    amp_i,
                    float(center1),
                    float(center2),
                    float(sigma_x_values[idx]),
                    float(sigma_y),
                )
            )

        if biases:
            self._bias_primary_frontier_A = float(biases[0][2])

        return biases[:int(MAX_GAUSSIANS_PER_ACTION)]

    # ---------- episode control ----------
    def _seed_persistent_locks(self, seed_from_max_A):
        # derive backstops from milestones below the seed distance
        self.backstops_A = []
        for m in DISTANCE_INCREMENTS:
            if m <= seed_from_max_A - LOCK_MARGIN:
                self.backstops_A.append(m - LOCK_MARGIN)

    def _update_episode_best(self):
        if self.current_distance > getattr(self, "best_distance_ever", float("-inf")):
            self.best_distance_ever = float(self.current_distance)
            self.best_positions = None if self.current_positions is None else self.current_positions

        if self.current_distance2 < getattr(self, "best_distance2_ever", float("inf")):
            self.best_distance2_ever = float(self.current_distance2)

    def reset(
        self,
        seed_from_max_A=None,
        carry_state=False,
        episode_index=None,
        target_center_A=None,
        target_half_width_A=None,
        target_stage=None,
    ):
        # set / clear milestone bookkeeping
        self.milestones_hit = set()
        self.milestones_hit_cv2 = set()
        self.in_zone_count = 0
        self.step_counter = 0
        self._cv2_bias_center = None
        self._bias_primary_frontier_A = None
        self.cv2_center_on = False
        self.episode_target_hit = False
        self.last_step_md_failed = False
        self.episode_md_failed = False
        self._set_episode_cv1_target(
            target_center_A=target_center_A,
            target_half_width_A=target_half_width_A,
            target_stage=target_stage,
        )

        self.phase = 1
        self.in_zone_steps = 0
        self.no_improve_counter = 0
        self.bias_log = []
        self.all_biases_in_episode = []
        self._bias_slots_used = 0

        # Locks
        self.locked_milestone_idx = -1
        self.backstops_A = []

        # MD state
        if not carry_state:
            self.current_positions = None
            self._last_positions = None
        else:
            # Resume from the best-reaching structure when available.
            if self.best_positions is not None:
                self._last_positions = self.best_positions
            else:
                self._last_positions = self.current_positions
            
        self.current_velocities = None   # never carry velocities
        self._last_topology = None

        # Zone walls cleared (CV1)
        self.zone_floor_A = None
        self.zone_ceiling_A = None

        # Clear CV2 zone walls too
        self.zone2_floor_A = None
        self.zone2_ceiling_A = None

        # caches cleared
        self.simulation = None
        self._dynamic_force_names = {}

        # Episode / DCD bookkeeping
        self.current_episode_index = episode_index
        self.current_dcd_index = 0
        self.current_dcd_paths = []
        if episode_index is None:
            self.current_run_name = None

        # clear per-episode trajectory segments
        self.episode_trajectory_segments = []
        self.episode_trajectory_segments_cv2 = []

        # seed cross-episode locks
        if seed_from_max_A is not None:
            self._seed_persistent_locks(seed_from_max_A)
            if SEED_ZONE_CAP_IF_BEST_IN_ZONE:
                if TARGET_MIN <= seed_from_max_A <= TARGET_MAX:
                    self.zone_floor_A = (TARGET_MIN + ZONE_MARGIN_LOW)
                    self.zone_ceiling_A = (TARGET_MAX - ZONE_MARGIN_HIGH)

        # runs.txt bookkeeping
        if DCD_SAVE and episode_index is not None:
            dcd_dir = RESULTS_TRAJ_DIR
            _ensure_dir(dcd_dir)
            run_name = f"{RUN_NAME_PREFIX}{episode_index:04d}"
            self.current_run_name = run_name
            runs_txt = RUNS_TXT
            existing = set()
            if os.path.exists(runs_txt):
                with open(runs_txt, "r") as fh:
                    existing = {ln.strip() for ln in fh if ln.strip()}
            if run_name not in existing:
                with open(runs_txt, "a") as fh:
                    fh.write(run_name + "\n")

        # resolve platform lazily
        if self.platform is None:
            self.platform = get_best_platform(verbose=True)

        self._build_episode_simulation()

        # initialize distances
        d1, d2 = self._current_distances_A()
        self.current_distance = float(d1)
        self.current_distance2 = float(d2)
        self._update_episode_best()

        # resolve CV2 targets now
        c2 = CURRENT_DISTANCE2 if CURRENT_DISTANCE2 is not None else self.current_distance2
        f2 = FINAL_TARGET2 if FINAL_TARGET2 is not None else c2
        center2 = TARGET2_CENTER if TARGET2_CENTER is not None else f2
        self._cv2_center = float(center2)
        self._cv2_min = float(center2 - TARGET2_ZONE_HALF_WIDTH) if TARGET2_ZONE_HALF_WIDTH is not None else None
        self._cv2_max = float(center2 + TARGET2_ZONE_HALF_WIDTH) if TARGET2_ZONE_HALF_WIDTH is not None else None

        # --- mirror CV1 "seed-best-in-zone" cap logic for CV2 ---
        if SEED_ZONE_CAP_IF_BEST_IN_ZONE:
            if (self._cv2_min is not None) and (self._cv2_max is not None):
                if self._cv2_min <= self.current_distance2 <= self._cv2_max:
                    self.zone2_floor_A   = self._cv2_min + CV2_ZONE_MARGIN_LOW
                    self.zone2_ceiling_A = self._cv2_max - CV2_ZONE_MARGIN_HIGH

        self._sync_dynamic_force_parameters()

        return self.get_state()

    # ---------- step ----------
    def step(self, action_index):
        action_index = int(action_index)
        self.last_step_md_failed = False

        biases = self.smart_progressive_bias(action_index)
        if (len(self.all_biases_in_episode) + len(biases)) > int(MAX_BIAS_TERMS_PER_EPISODE):
            raise RuntimeError(
                f"Exceeded preallocated 2D bias slots: "
                f"{len(self.all_biases_in_episode) + len(biases)} > {MAX_BIAS_TERMS_PER_EPISODE}. "
                f"Increase MAX_GAUSSIANS_PER_ACTION or MAX_BIAS_TERMS_PER_EPISODE."
            )

        self.step_counter += 1
        for b in biases:
            if len(b) == 6:
                kind, amp_kcal, center1_A, center2_A, sigma_x_A, sigma_y_A = b
            else:
                raise ValueError(f"Unexpected bias tuple length={len(b)}: {b}")

            amp_kcal = float(min(float(amp_kcal), float(APPLIED_MAX_AMP)))
            sigma_x_A = float(max(float(sigma_x_A), 0.3))
            sigma_y_A = float(max(float(sigma_y_A), 0.3))

            self.all_biases_in_episode.append(
                (float(amp_kcal), float(center1_A), float(center2_A), float(sigma_x_A), float(sigma_y_A))
            )

            # bias_log keeps (step, kind, amp, center1, center2, sigma_x, sigma_y)
            self.bias_log.append(
                (self.step_counter, str(kind), float(amp_kcal), float(center1_A), float(center2_A), float(sigma_x_A), float(sigma_y_A))
            )
        self._bias_slots_used = len(self.all_biases_in_episode)

        gamma = fricCoef
        if getattr(self, "cv2_center_on", False) or (self.phase == 2):
            gamma = 10.0 / u.picoseconds  # stronger damping only in the locked regime
        sim = self.simulation
        if sim is None:
            sim = self._build_episode_simulation()
        try:
            sim.integrator.setFriction(gamma)
        except Exception:
            pass
        self._sync_dynamic_force_parameters()

        #local minimization before launching dynamics
        try:
            openmm.LocalEnergyMinimizer.minimize(sim.context, 10.0 * u.kilojoule_per_mole, 200)
        except Exception:
            pass
        
        # Velocities: ALWAYS reinitialize (do NOT carry)
        sim.context.setVelocitiesToTemperature(T, SEED)

        # DCD reporter per RL action
        sim.reporters.clear()
        if DCD_SAVE and self.current_run_name is not None:
            dcd_dir = RESULTS_TRAJ_DIR
            _ensure_dir(dcd_dir)
            self.current_dcd_index += 1
            dcd_name = f"{self.current_run_name}_s{self.current_dcd_index:03d}.dcd"
            dcd_path = os.path.join(dcd_dir, dcd_name)
            sim.reporters.append(DCDReporter(dcd_path, int(DCD_REPORT_INTERVAL)))
            self.current_dcd_paths.append(dcd_path)

        # propagate
        prev_d1 = self.current_distance
        prev_d2 = self.current_distance2

        # live per-chunk samples for this RL action
        seg_cv1 = []
        seg_cv2 = []
        pairs = [(self.atom1_idx, self.atom2_idx), (self.atom3_idx, self.atom4_idx)]
        seg_t_ps = []

        pre_state = sim.context.getState(getPositions=True)
        fallback_positions = pre_state.getPositions(asNumpy=True)

        prop_result = _nan_safe_propagate(
            sim,
            int(propagation_step),
            dcdfreq=int(dcdfreq_mfpt),
            prop_index=self.step_counter,
            atom_pairs=pairs,
            out_cv1=seg_cv1,
            out_cv2=seg_cv2,
            out_times_ps=seg_t_ps,
            dt_min=MIN_STEPSIZE,
        )       

        if len(seg_cv1) > 0:
            self.episode_trajectory_segments.append(seg_cv1)
        if len(seg_cv2) > 0:
            self.episode_trajectory_segments_cv2.append(seg_cv2)

        st = sim.context.getState(getPositions=True)
        candidate_positions = st.getPositions(asNumpy=True)
        d1, d2 = self._validated_distances_from_positions(candidate_positions)
        md_failed = bool(prop_result.get("failed")) and int(prop_result.get("completed_chunks", 0)) == 0
        if d1 is None or d2 is None or md_failed:
            self.last_step_md_failed = True
            self.episode_md_failed = True
            print(
                f"[warn] terminating episode after MD failure at action {self.step_counter}: "
                f"completed_chunks={int(prop_result.get('completed_chunks', 0))}/"
                f"{int(prop_result.get('requested_chunks', 0))}"
            )
            try:
                sim.context.setPositions(fallback_positions)
            except Exception:
                pass

            self.current_positions = fallback_positions
            self._last_positions = fallback_positions
            self._last_topology = self.psf.topology
            self.current_velocities = None

            safe_d1, safe_d2 = self._validated_distances_from_positions(fallback_positions)
            self.current_distance = float(prev_d1 if safe_d1 is None else safe_d1)
            self.current_distance2 = float(prev_d2 if safe_d2 is None else safe_d2)
            self.no_improve_counter += 1
            reward = float(MD_FAILURE_PENALTY + STEP_PENALTY)
            done = True
            return self.get_state(), reward, done, [self.current_distance, self.current_distance2]

        self.current_positions = candidate_positions
        self._last_positions = self.current_positions
        self._last_topology = self.psf.topology
        self.current_velocities = None 

        self.current_distance = float(d1)
        self.current_distance2 = float(d2)
        self._update_episode_best()

        # reward / termination (CV1 primary, CV2 shaping)
        delta1 = self.current_distance - prev_d1
        outward = max(0.0, delta1)
        inward = max(0.0, -delta1)

        delta2 = self.current_distance2 - prev_d2
        inward2 = max(0.0, -delta2)    # good: d2 decreases
        outward2 = max(0.0, delta2)    # bad: d2 increases

        reward = 0.0
        done = False
        stage_center_A, stage_min_A, stage_max_A, stage_half_width_A = self.current_target_zone()
        target_center_A, target_min_A, target_max_A, target_half_width_A = self.final_target_zone()

        stage_in_zone1 = (stage_min_A <= self.current_distance <= stage_max_A)
        in_zone1 = (target_min_A <= self.current_distance <= target_max_A)
        in_zone2 = (self._cv2_min <= self.current_distance2 <= self._cv2_max)
        if stage_in_zone1:
            self.episode_target_hit = True

        if in_zone1 and self.phase == 1:
            self.phase = 2
            self.in_zone_count = 0

        # CV2 penalty outside corridor (gentle shaping)
        if self._cv2_min is not None and self._cv2_max is not None:
            if self.current_distance2 < self._cv2_min:
                reward -= CV2_DEVIATION_PENALTY * (self._cv2_min - self.current_distance2)
            elif self.current_distance2 > self._cv2_max:
                reward -= CV2_DEVIATION_PENALTY * (self.current_distance2 - self._cv2_max)

        if self.phase == 1:
            # CV1 unbinding is the primary objective.
            reward += PROGRESS_REWARD * outward

            # CV2 is a soft shaping term only.
            reward += CV2_PROGRESS_REWARD_SCALE * PROGRESS_REWARD * inward2

            # milestones on CV1
            for m in DISTANCE_INCREMENTS:
                if prev_d1 < m <= self.current_distance and (m not in self.milestones_hit):
                    reward += MILESTONE_REWARD
                    self.milestones_hit.add(m)

            if outward > 0.0:
                reward += VELOCITY_BONUS
                self.no_improve_counter = 0
            else:
                reward += BACKTRACK_PENALTY * inward
                self.no_improve_counter += 1

            reward += CV2_PROGRESS_REWARD_SCALE * BACKTRACK_PENALTY * outward2

            reward += STEP_PENALTY
        
        else:
            # phase 2: maximize stability at the CV1 target.
            reward += _phase2_bowl_reward(self.current_distance, target_center_A, target_half_width_A, CENTER_GAIN)

            if (CV2_PROGRESS_REWARD_SCALE > 0.0
                and self._cv2_center is not None
                and TARGET2_ZONE_HALF_WIDTH is not None):
                reward += _phase2_bowl_reward(self.current_distance2, self._cv2_center, TARGET2_ZONE_HALF_WIDTH, CENTER_GAIN)

            if in_zone1:
                self.in_zone_count = getattr(self, "in_zone_count", 0) + 1
                reward += 0.5 * CONSISTENCY_BONUS
                if self.in_zone_count >= STABILITY_STEPS:
                    reward += 1000.0
                    done = True

            if abs(self.current_distance - target_center_A) < PHASE2_TOL:
                reward += 1500.0
                done = True

            reward += STEP_PENALTY

        return self.get_state(), float(reward), bool(done), [self.current_distance, self.current_distance2]

# ========================= Standalone smoke-test (optional) =========================
if __name__ == "__main__":
    # This block only runs if you execute: python combined2.py
    # It is safe to import combined2.py from main.py / evaluate.py.
    print("combined2.py loaded as __main__ (2D CV smoke test)")
    env = ProteinEnvironmentRedesigned()
    s = env.reset(episode_index=1)
    print("Initial state shape:", s.shape, "CV1/CV2:", env.current_distance, env.current_distance2)

