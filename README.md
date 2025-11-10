# RL for Protein–Ligand Unbinding

## 🧬 Overview
This project models the **protein–ligand unbinding process** as a reinforcement learning (RL) problem grounded in molecular dynamics (MD) physics.  
Using **OpenMM** for atomistic simulation and a **Proximal Policy Optimization (PPO)** agent for control, the system learns to apply adaptive **Gaussian bias potentials** to drive the ligand from the bound state to the unbound (target) zone while maintaining physical plausibility.

## ⚙️ Physical Background
Protein–ligand interactions are governed by complex energy landscapes.  
Traditional enhanced sampling methods (e.g., metadynamics) use pre-defined bias potentials to accelerate transitions.  
Here, the bias is **learned adaptively** by an RL agent. Each action defines a Gaussian bias potential with:
- **Amplitude (energy in kcal/mol)**
- **Center (distance in Å)**
- **Width (spread of bias in Å)**

These biases reshape the system’s potential energy surface to promote unbinding.  
The environment enforces physical realism using:
- **Milestone locks** – harmonic backstops that prevent regression behind achieved distances.  
- **Zone confinement** – high-stiffness boundaries near the target zone to ensure stable unbinding.  
- **Reward shaping** – encourages steady outward motion, penalizes backtracking, and rewards physical convergence.

The process represents a **physics-informed control** of molecular energy flow, merging MD with data-driven exploration.

## 🧠 Components
| Module | Description |
|---------|--------------|
| `main.py` | Training loop for PPO agent with progressive unbinding control and automatic plotting. |
| `agent.py` | Actor–Critic PPO implementation with entropy regularization, advantage estimation, and exploration decay. |
| `env_protein.py` | OpenMM-based environment defining forces, biases, and distance evolution under physical constraints. |
| `config.py` | Physical constants, atom indices, hyperparameters, and RL configuration. |
| `util_protein.py` | Visualization utilities for bias profiles, trajectories, locks, and performance metrics. |

## 📊 Outputs
During training, the following are generated:
- **Trajectory plots** showing distance evolution per episode.  
- **Bias landscapes** visualizing total and component potentials.  
- **Timelines** for Gaussian parameters (amplitude, center, width).  
- **JSON metadata** for reproducibility of each episode.

All results are saved under:
plots/ # Quick per-episode visual outputs
results_PPO/ # Structured results and KPIs


## 🚀 Usage
```bash
python main.py
```

Adjust hyperparameters and simulation parameters in config.py before training.
Ensure that OpenMM is installed and that step3_input.psf, traj_0.restart.pdb, and toppar.str files are correctly specified.

🧩 Key Physics Principles
- Gaussian bias as a controllable local energy perturbation.
- Milestone-based harmonic constraints for irreversible progress.
- Two-phase reward structure: unbinding (Phase 1) and zone stabilization (Phase 2).
- Reinforcement learning guided by thermodynamic and kinetic constraints.
