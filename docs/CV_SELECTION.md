# CV Selection

This document explains how to choose or auto-rank `CV2` for Mg-phosphate unbinding.

Primary source files:

- [analysis/cv2_autoselect.py](../analysis/cv2_autoselect.py)
- [scripts/cv2_autoselect.py](../scripts/cv2_autoselect.py)
- [combined_2d.py](../combined_2d.py)

Related follow-up:

- [TICA_ANALYSIS.md](./TICA_ANALYSIS.md): slow-coordinate validation of the selected CVs
- [PCA_ANALYSIS.md](./PCA_ANALYSIS.md): variance-coordinate validation and PCA-space restart candidates
- [HYBRID_RESTART_SELECTION.md](./HYBRID_RESTART_SELECTION.md): TICA-selected restart PDB export after CV/TICA analysis

## Current CVs

The current PPO model uses explicit distance CV definitions:

- `CV1`: Mg to phosphate P distance, currently atom `7799` to atom `7840`
- `CV2`: Mg to phosphate O2 distance, currently atom `7799` to atom `7842`

Topology inspection showed:

- Mg: atom index `7799`, `MG`, segid `HETA`
- phosphate P: atom index `7840`, `P`, residue `H2PO4`, segid `HETC`
- phosphate O2: atom index `7842`, `O2`, residue `H2PO4`, segid `HETC`
- phosphate oxygens: `O1`, `O2`, `O3`, `O4` in segid `HETC`
- initial Mg-O2 distance from `traj_0.restart.pdb`: about `1.861 A`

## Preferred Candidate Class

For Mg-phosphate unbinding, Mg coordination to phosphate oxygens is chemically meaningful:

- Mg to phosphate O2 distance is directly compatible with the current distance-based `CV2` force and is now the active configured `CV2`.
- Smooth Mg-phosphate oxygen coordination is chemically richer, but it is not directly compatible with the current distance-only CV2 bias force without extending the OpenMM CV expression.

The current Mg-O2 CV2 configuration is:

```python
ATOM3_INDEX = 7799
ATOM4_INDEX = 7842
CURRENT_DISTANCE2 = 1.861
FINAL_TARGET2 = 6.507
CV2_PROGRESS_DIRECTION = "increase"
```

## Auto-Selection Command

After training has produced DCD files, run:

```powershell
python scripts\cv2_autoselect.py --config-module combined_2d --max-traj 0
```

Lower-cost run:

```powershell
python scripts\cv2_autoselect.py --config-module combined_2d --max-traj 20 --stride 5
```

The selector ranks:

- Mg to phosphate O2 distance
- Mg to each phosphate oxygen distance
- smooth Mg coordination number to phosphate oxygens
- nearby Mg/protein and phosphate/protein pathway distances

Ranking uses trajectory-derived dynamics and chemical priors:

- candidate variability
- time-lagged autocorrelation
- net change over the sampled run
- partial penalty for duplicating `CV1`
- higher chemical priority for Mg-phosphate oxygen candidates

## Outputs

Outputs go under:

```text
results_PPO/analysis_runs/<timestamp>/
```

Key files:

- `data/cv2_auto_selection_candidates.csv`
- `data/cv2_auto_selection_summary.json`
- `data/cv2_auto_selection_report.md`
- `figs/analysis/cv2_auto_selection_top_timeseries.png`

If the best candidate is a distance, the report writes a ready-to-use config snippet:

```python
ATOM3_INDEX = ...
ATOM4_INDEX = ...
CURRENT_DISTANCE2 = ...
FINAL_TARGET2 = ...
CV2_PROGRESS_DIRECTION = "increase"  # or "decrease"
```

## Direction Control

`combined_2d.py` supports:

```python
CV2_PROGRESS_DIRECTION = "increase"
```

Use:

- `"decrease"` for contact-closing CVs
- `"increase"` for unbinding-like distances such as Mg to phosphate O2

The active Mg-O2 CV2 uses `"increase"` because Mg-O2 distance should increase during unbinding.

## Relationship to TICA

Use the auto-selector and TICA together:

- `cv2_autoselect.py` ranks candidate CV2 definitions.
- `tica.py` checks whether slow modes are captured by `CV1`, `CV2`, or hidden pathway features.

Recommended order after exploratory training:

1. `python scripts\cv2_autoselect.py --config-module combined_2d --max-traj 0`
2. update `CV2` only if the recommendation is chemically and dynamically plausible
3. `python scripts\tica.py --config-module combined_2d --max-traj 0 --lag 5`
4. confirm whether the selected `CV2` correlates with a leading TIC
5. optionally run `python scripts\pca_hybrid_restart.py --config-module combined_2d --max-episode 10 --top-restarts 10` to export PCA-space structural-diversity restart candidates
6. optionally run `python scripts\hybrid_restart.py --config-module combined_2d --max-episode 10 --top-restarts 10` to export restart candidates from the TICA run
