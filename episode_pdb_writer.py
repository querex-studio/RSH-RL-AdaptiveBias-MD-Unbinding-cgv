# episode_pdb_writer.py  (final robust)
import os
from datetime import datetime

try:
    from openmm.app import PDBFile
    from openmm.unit import Quantity, nanometer
except Exception:
    from simtk.openmm.app import PDBFile  # type: ignore
    from simtk.unit import Quantity, nanometer  # type: ignore


def _now_tag() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _find_attr(obj, names):
    for n in names:
        if hasattr(obj, n):
            val = getattr(obj, n)
            if val is not None:
                return val
    return None


def _get_topology(env, simulation=None):
    # 1) simulation.topology
    if simulation is not None:
        topo = getattr(simulation, "topology", None)
        if topo is not None:
            return topo
    # 2) env.psf.topology
    psf = getattr(env, "psf", None)
    if psf is not None and hasattr(psf, "topology"):
        return psf.topology
    # 3) env.topology
    topo = getattr(env, "topology", None)
    if topo is not None:
        return topo
    # 4) cached
    topo = getattr(env, "_last_topology", None)
    if topo is not None:
        return topo

    raise RuntimeError("No Topology found (simulation.topology / env.psf.topology / env.topology / env._last_topology).")


def _get_positions_from_simulation(simulation):
    state = simulation.context.getState(getPositions=True)
    pos = state.getPositions(asNumpy=True)
    if not isinstance(pos, Quantity):
        raise RuntimeError("Simulation positions lack units.")
    return pos


def _get_positions_from_context(context):
    state = context.getState(getPositions=True)
    pos = state.getPositions(asNumpy=True)
    if not isinstance(pos, Quantity):
        raise RuntimeError("Context positions lack units.")
    return pos


def _coerce_positions_quantity(pos_array):
    """
    Accept numpy array shape (N,3) without units, wrap as Quantity[nm].
    We assume nanometer (OpenMM internal). If your cache is in Å,
    scale externally before caching or adjust here.
    """
    try:
        import numpy as np
        if hasattr(pos_array, "shape") and pos_array.shape[-1] == 3:
            # Wrap as Quantity in nm
            return pos_array * nanometer
    except Exception:
        pass
    raise RuntimeError("Cached positions are not a valid (N,3) array.")


def write_episode_pdb(env, out_dir: str, episode_idx: int) -> str:
    _ensure_dir(out_dir)

    # Prefer a live Simulation
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
        # Try a Context
        context = _find_attr(env, ("context", "_context", "sim_context", "_last_context"))
        if context is not None and topo is None:
            topo = _get_topology(env, simulation=None)
        if context is not None and pos is None:
            try:
                pos = _get_positions_from_context(context)
            except Exception:
                pos = None

    if pos is None:
        # Final fallback: cached array (no units) e.g., env._last_positions
        cached = _find_attr(env, ("_last_positions", "last_positions", "positions_cache"))
        if cached is not None:
            pos = _coerce_positions_quantity(cached)

    if topo is None or pos is None:
        raise RuntimeError(
            "Could not assemble Topology+Positions. "
            "Checked Simulation/Context and cached positions. "
            "Ensure the env caches _last_topology and _last_positions as shown."
        )

    tag = _now_tag()
    fname = os.path.join(out_dir, f"{tag}_episode_{episode_idx:04d}.pdb")
    with open(fname, "w") as fh:
        PDBFile.writeFile(topo, pos, fh, keepIds=True)

    print(f"[episode_pdb_writer] Saved end-of-episode PDB: {fname}")
    return fname
