# Adaptive CVgen Project Review

## Scope of this review

This review covers the local `Adaptive_CVgen-main` folder:

- `2024-Shen-AdaptiveCVgen.pdf`: the main PNAS paper, "Adaptive CVgen: Leveraging reinforcement learning for advanced sampling in protein folding and chemical reactions."
- `README.md`: repository note and citation.
- `CVgen_data_process.ipynb`: post-processing, figure generation, CV construction utilities, RMSD/native-contact analysis, and RL ablation visualizations.
- `CVgen_vs_others.ipynb`: comparison analysis against Least Count and D. E. Shaw/Anton datasets.

Important repository limitation: the README says the full Adaptive CVgen implementation is still being organized and will be released later. The local notebooks confirm that this folder is mainly a data-processing and visualization release. It reads precomputed trajectories, log files, and pickle files from `./data`, `./iTOL_plot`, `../least_count`, and one hard-coded external Linux path. It does not contain the complete OpenMM simulation loop, seed-selection driver, or full RL optimizer used to generate the trajectories.

## High-level project purpose

Adaptive CVgen is an enhanced sampling framework for molecular dynamics (MD). Its purpose is to accelerate discovery of scientifically relevant transitions, especially protein folding pathways and complex chemical reactions, without imposing a fixed low-dimensional reaction coordinate or biasing potential.

The paper frames two central problems:

1. Reaction-coordinate design: many enhanced sampling methods depend on a small set of manually chosen collective variables (CVs). If those CVs are incomplete or wrong, the simulation can miss important routes through phase space.
2. Exploration versus exploitation: adaptive sampling must exploit promising progress toward a target state while still exploring new regions so it can escape local minima.

Adaptive CVgen addresses these together by generating many local and long-range CVs, learning their weights from historical trajectory data, and selecting new seed conformations based on a learned reward score.

The paper demonstrates the method on six proteins and a C60 synthesis process. For proteins, the starting states are disordered conformations and the target is the native structure or AlphaFold2 reference for the WW domain variant GTT case. The six protein systems used in the notebooks are:

| PDB ID | Sequence length in notebook |
|---|---:|
| `1fme` | 28 |
| `2f4k` | 35 |
| `2f21` | 35 |
| `2hba` | 39 |
| `2p6j` | 52 |
| `2a3d` | 73 |

## What Adaptive CVgen does algorithmically

The paper describes the method as a repeated adaptive sampling loop:

1. Generate a high-dimensional CV set.
2. Launch a round of unbiased MD simulations from selected starting conformations.
3. Update CV weights from historical trajectories.
4. Score all candidate conformations.
5. Cluster the sampled conformations in TICA space and choose new seeds.
6. Repeat until a convergence or stopping criterion is reached.

### Step 1: CV generation

Adaptive CVgen tries to avoid a single hand-picked reaction coordinate. Instead, it builds many CVs that cover local and long-range structural changes.

From the paper:

- CVs are designed to capture local structural changes, so the algorithm can detect which part of the molecule is evolving.
- CVs also span different lengths and positions, so long-range correlations and larger-scale structural organization are represented.
- Because a large CV set can contain correlated variables, the paper applies SLSQP optimization to compute decorrelation weights, represented as `I`.

From the notebooks:

- The `CVgenerator` class constructs residue-window CVs from a reference PDB structure.
- With automatic settings, window sizes are `5, 9, 13, ...` residues up to the full protein length, and step sizes are approximately half the window size.
- `regular` mode slides windows across the full sequence and adds beta-strand CVs when beta strands are detected.
- `structure` mode uses detected alpha-helical and beta-sheet secondary-structure segments.
- Alpha and beta regions are identified using `mdtraj.compute_dssp`.
- Beta-strand pairing uses `mdtraj.baker_hubbard` hydrogen-bond detection, converts atom-level hydrogen bonds to residue pairs, aligns paired beta strands, and then builds paired beta-window CVs.
- The first `CVgenerator` definition uses C-alpha atoms as the important atoms for CV RMSD. A later duplicate class definition switches to heavy atoms and can export per-CV PDB files into `CV_structures`. This duplicate definition is useful for visualization but is a reproducibility risk if the notebook is rerun out of order.

### Step 2: MD round execution

The paper runs multiple parallel replicas per round. In the reported protein experiments, it uses 16 replicas. The notebook parameters mirror that:

```python
protein_lst = ['1fme', '2f4k', '2f21', '2hba', '2p6j', '2a3d']
intervals = [400, 400, 400, 400, 400, 800]
replicas = [16, 16, 16, 16, 16, 16]
dts = [0.000025, 0.000025, 0.000025, 0.000025, 0.000025, 0.00005]
```

The notebooks do not run OpenMM simulations. They load trajectories already generated elsewhere, for example:

- `./data/<protein>/<protein>.dcd`
- `./data/<protein>/No_delt/history.dcd`
- `./data/<protein>/alpha0/<protein>.dcd`
- `./data/compare_DEshaw/<protein>/history.dcd`

### Step 3: Update CV weights

The paper uses several central variables:

- `gamma`: prior reward term for each CV. This expresses whether a CV is physically favorable according to the current system objective.
- `delta`: data-driven penalty/resistance term for each CV. It reflects how long or how often the system remains in states represented by that CV. The paper sets the penalty rate `omega = -1`, so repeated residence can reduce the corresponding CV's weight.
- `alpha`: balance coefficient between prior reward and data-driven penalty.
- `W`: final/posterior CV weights.
- `R`: learned reward score for candidate conformations.
- `Theta`: matrix of instantaneous rewards for conformations under CVs.

The paper expresses the dynamic weight structure as:

```text
W = W_prior * W_dynamic
W_prior = I * gamma
W = I * gamma * exp(alpha * delta)
R = W * Theta
```

In protein folding examples, each conformation's CV-level reward is based on RMSD to the corresponding native segment:

```text
theta_i^j = phi(q_i^j)
q_i^j = RMSD(segment_i of conformation_j, native segment_i)
phi(x) = 1 / x
```

Lower RMSD gives a higher reward.

### Step 4: Score conformations

After weighting, Adaptive CVgen scores candidate conformations by multiplying CV weights with CV-specific reward values:

```text
R_{1xM} = W_{1xN} * Theta_{NxM}
```

This gives one score per sampled conformation. The score is not simply "closest RMSD overall." It is a weighted aggregate over all generated CVs, where the weights have been modified by historical sampling and the RL balance mechanism.

### Step 5: TICA clustering and seed selection

The paper uses TICA to project historical trajectories onto slow dynamical modes, then clusters the projected conformations. From each cluster it randomly chooses `k = 8` conformations, ranks those by `R`, and then picks the top `x` conformations as seeds for the next simulation round. In the paper, `x` corresponds to the number of replicas, 16.

The notebooks implement the analysis side of this:

- TICA is performed using `deeptime.decomposition.TICA`.
- C-alpha pair contact features are built using PyEMMA, with contact threshold `0.8` nm.
- Protein-specific TICA lag times are `[40, 40, 40, 40, 40, 80]` in the Adaptive CVgen/Least Count analyses.
- The D. E. Shaw comparison uses `lagtime = 1` and also uses C-alpha contact features.

### Step 6: Stop criterion

The paper states that sampling stops when the system reaches a predefined structure-quality threshold, such as RMSD or native-contact threshold, or a user-defined simulation duration.

The notebooks use RMSD and native-contact traces to identify closest-to-native frames. For example, in the main 300 K Adaptive CVgen analysis, the notebook records the first frame close to the best observed RMSD and the first frame close to the maximum native-contact fraction.

## Role of reinforcement learning

The paper calls the method reinforcement learning because the algorithm repeatedly observes trajectory history, updates a reward/penalty model, and selects future sampling seeds according to a learned reward score. It is important to be precise: the local notebooks do not contain a deep RL policy network, neural Q-function, actor-critic model, or standalone RL training script. The paper's RL role is closer to adaptive reward learning/control over CV weights and seed selection.

The RL mechanism has two linked strategies.

### RL strategy 1: learn `alpha`

The first RL strategy learns the balance coefficient `alpha`.

`gamma` encourages sampling along CVs with favorable prior reward. `delta` suppresses CVs that have become resistant or over-occupied based on historical sampling. If either term dominates too strongly, the simulation can become unbalanced:

- Too much reliance on `gamma` can over-exploit known promising directions and get trapped.
- Too much penalty can push exploration away from useful coordinates and waste sampling.

The paper defines a custom objective `S(alpha)` that decomposes the effect of candidate `alpha` values into exploitation and exploration contributions. The optimal `alpha` acts like a regularizer: it prevents extreme CV weights and keeps the search from depending too heavily on one subset of CVs.

In the initial round, `alpha = 0`. In later rounds, it is updated from historical conformations.

### RL strategy 2: learn conformation rewards through `R`

Once `alpha` is chosen, the algorithm computes posterior CV weights:

```text
W = I * gamma * exp(alpha * delta)
```

Then it computes:

```text
R = W * Theta
```

`R` is the learned conformation reward used to rank candidate seed structures for the next round. This is where the RL loop directly affects sampling behavior: it changes which conformations are selected as starting points for future MD replicas.

### State, action, and reward interpretation

A practical RL interpretation of the paper is:

- State: historical MD trajectories, CV projections, TICA/clustering distribution, CV residence/resistance information, and current CV weights.
- Action: update CV weights and choose the next batch of seed conformations.
- Reward: CV-level structural progress, usually inverse RMSD for protein segments, plus regularized balancing between exploitation and exploration.
- Policy: the induced seed-selection rule that ranks clustered candidates by `R` and starts the next round from the best-scoring candidates.

This is not the same as applying an artificial biasing force during MD. Adaptive CVgen keeps the MD trajectories unbiased and intervenes only between rounds by choosing which conformations to extend.

## TICA in Adaptive CVgen

Time-lagged independent component analysis, or TICA, is the main dimensionality-reduction method used in the paper and notebooks. The project does not use PCA for its phase-space analysis. It uses TICA because protein folding and chemical reaction sampling are dynamical problems: the important coordinates are not necessarily the coordinates with the largest instantaneous variance, but the coordinates that describe slow transitions between metastable conformational states.

### Why TICA is used

In Adaptive CVgen, TICA is used after MD trajectories have been generated. It converts high-dimensional trajectory features into a low-dimensional map of slow molecular motion. The paper then uses this map for clustering and seed selection:

1. Collect all historical trajectories sampled so far.
2. Build structural features from the trajectories.
3. Fit TICA to identify the slowest dynamical directions.
4. Project sampled conformations onto the first two TICA dimensions.
5. Cluster the projected conformations.
6. Sample candidate conformations from clusters, rank them by the learned Adaptive CVgen reward `R`, and select the best candidates as seeds for the next MD round.

The notebooks implement this same analysis pattern. They use C-alpha contact features from PyEMMA, then fit `deeptime.decomposition.TICA` with `dim=2`. For the Adaptive CVgen and Least Count analyses, the protein-specific TICA lag times are:

```python
lagtimes = [40, 40, 40, 40, 40, 80]
```

The D. E. Shaw/Anton comparison uses `lagtime = 1` and fits TICA on the Shaw/Anton trajectory features, then projects Adaptive CVgen trajectories into that same TICA coordinate system. This makes the comparison stricter because Adaptive CVgen is evaluated in a dynamical basis learned from long unbiased reference trajectories.

### TICA algorithm logic

At a high level, TICA finds linear combinations of trajectory features that stay correlated with themselves after a time lag. If the feature vector at time `t` is `x_t`, TICA searches for a projection vector `v` such that the projected coordinate:

```text
y_t = v^T x_t
```

has high autocorrelation with:

```text
y_{t+tau} = v^T x_{t+tau}
```

where `tau` is the chosen lag time.

Conceptually, the algorithm:

1. Centers the trajectory features.
2. Computes the instantaneous covariance matrix:

```text
C_00 = E[x_t x_t^T]
```

3. Computes the time-lagged covariance matrix:

```text
C_0tau = E[x_t x_{t+tau}^T]
```

4. Solves a generalized eigenvalue problem of the form:

```text
C_0tau v = lambda C_00 v
```

5. Sorts the eigenvectors by eigenvalue magnitude. Larger eigenvalues correspond to slower decorrelation and therefore slower dynamical modes.
6. Projects conformations onto the leading eigenvectors to obtain low-dimensional TICA coordinates.

In MD language, the leading TICA components are approximate slow collective coordinates. They are useful because conformational transitions, folding events, and barrier-crossing behavior often occur along slow modes rather than along the directions of highest raw variance.

### TICA versus PCA

PCA and TICA can look superficially similar because both produce linear projections and both can reduce high-dimensional features to two or three plotted dimensions. Their optimization targets are different.

| Method | What it optimizes | Uses time lag? | Typical meaning in MD |
|---|---|---:|---|
| PCA | Maximum instantaneous variance | No | Large-amplitude structural fluctuations |
| TICA | Maximum time-lagged autocorrelation | Yes | Slow dynamical processes and metastable transitions |

PCA solves an eigenproblem on the covariance of the data at the same time point:

```text
C_00 v = lambda v
```

This means PCA finds directions where the data cloud is widest. That can be useful for coarse structural visualization, but high variance does not necessarily mean slow or mechanistically important. For example, flexible loop motion can dominate PCA even if it is not the folding coordinate.

TICA solves a time-lagged problem:

```text
C_0tau v = lambda C_00 v
```

This means TICA favors coordinates that retain memory over time. In MD, that property is often closer to the reaction-coordinate problem because metastable-state transitions are slow and temporally correlated.

### Advantages of TICA in MD simulations

TICA has several advantages for this project:

- It emphasizes slow transitions. Folding and conformational rearrangements often happen on slow timescales, so TICA better captures the coordinates relevant to state changes.
- It supports metastable-state discovery. Clustering in TICA space tends to group conformations according to slow dynamical basins, which is useful for adaptive sampling.
- It reduces noisy high-dimensional features. Contact maps and atom-pair features can be very high dimensional; TICA compresses them into a small number of interpretable slow modes.
- It is compatible with Markov-state-model thinking. TICA is commonly used before clustering and MSM construction because it creates coordinates that preserve slow kinetics better than PCA.
- It improves comparison across methods. In this repository, TICA provides a common phase-space map for comparing Adaptive CVgen, RL-disabled Adaptive CVgen, fixed-`alpha` variants, Least Count, native states, and Shaw/Anton trajectories.
- It provides a useful visualization space. The TICA plots show whether a method reaches the native-state region, remains trapped, explores irrelevant basins, or covers similar dynamical regions to a reference dataset.

### Role of TICA in the RL sampling loop

TICA is not the RL model itself. The RL part of Adaptive CVgen is the learned adjustment of CV weights and conformation rewards through variables such as `delta`, `alpha`, `W`, and `R`. TICA provides the phase-space organization used around that reward model.

The interaction is:

1. MD produces trajectory history.
2. TICA projects that history into slow coordinates.
3. Clustering partitions the sampled slow-coordinate space into regions.
4. Adaptive CVgen computes reward values for candidate conformations using the learned CV weights.
5. The highest-reward candidates from the clustered space become seeds for the next MD round.

This combination matters because reward alone could over-focus on a narrow set of structurally promising conformations, while clustering alone could over-explore low-value regions. TICA-based clustering gives the method a dynamical map of where conformations sit; RL-based scoring decides which clustered conformations are worth extending.

In practical RL terms:

- TICA helps define a compact state-space representation of historical sampling coverage.
- Clusters in TICA space help maintain diversity among candidate seeds.
- The reward `R` chooses promising conformations within that diversity.
- The `delta` penalty and learned `alpha` help prevent repeated exploitation of directions where the system is stuck.

This is one reason Adaptive CVgen can balance exploration and exploitation without directly biasing the MD force field.

### Caveats

TICA is powerful but not automatic proof that the projection is the true reaction coordinate. Its quality depends on the selected features, the lag time, the amount and diversity of sampled data, and whether enough relevant transitions are present in the trajectories. A poor feature set or inappropriate lag time can hide important processes. In this project, the authors mitigate this by using contact-based features, protein-specific lag times for the main comparisons, and direct validation with RMSD/native-contact metrics and Shaw/Anton trajectory projections.

## What the notebooks do

### `CVgen_data_process.ipynb`

This notebook has four main roles.

First, it defines shared analysis utilities:

- `extract_values(filename, variable_name)`: parses vectors such as `theta`, `delt`, and `W` from log files.
- `extract_consecutive_parts(arr)`: splits consecutive residue-index runs.
- `CVgenerator`: generates CV atom/residue selections from a reference PDB.
- `best_hummer_q(traj, native)`: computes fraction of native contacts using the Best-Hummer-Eaton definition with heavy-atom contacts separated by more than 3 residues, native cutoff `0.45` nm, `BETA_CONST = 50`, and `LAMBDA_CONST = 1.8`.

Second, it evaluates Adaptive CVgen at 300 K:

- Loads `./data/<protein>/<protein>.dcd`.
- Generates CV selections from `./iTOL_plot/<protein>/p.pdb`.
- Computes per-CV RMSD traces and whole-structure RMSD.
- Computes native-contact fraction `Q`.
- Saves plots such as `CVs_rmsd.png` and `CVs_q.png`.

Stored notebook outputs for the 300 K Adaptive CVgen analysis:

| Protein | First near-min RMSD frame | Approx. aggregate time | Minimum RMSD |
|---|---:|---:|---:|
| `1fme` | 194406 | 4.860 microseconds | 1.3196 |
| `2f4k` | 149592 | 3.740 microseconds | 0.5690 |
| `2f21` | 257750 | 6.444 microseconds | 1.6194 |
| `2hba` | 167827 | 4.196 microseconds | 0.3913 |
| `2p6j` | 225212 | 5.630 microseconds | 2.0002 |
| `2a3d` | 354565 | 17.728 microseconds | 2.0076 |

Third, it evaluates RL-disabled and fixed-`alpha` ablations:

- `No_delt` is used as the RL-disabled case, corresponding to no `delta` penalty effect, equivalent to setting `alpha = 0` for that term.
- Additional folders `alpha0`, `alpha0.1`, and `alpha1` are used to compare fixed `alpha` values against learned `alpha`.
- The notebook computes RMSD, TICA coverage, epsilon sampling benefit, and normalized efficiency `eta`.

Stored notebook outputs for the RL-disabled case:

| Protein | First near-min RMSD frame | Approx. aggregate time | Minimum RMSD |
|---|---:|---:|---:|
| `1fme` | 74477 | 1.862 microseconds | 1.3544 |
| `2f4k` | 137150 | 3.429 microseconds | 0.6214 |
| `2f21` | 197841 | 4.946 microseconds | 1.6278 |
| `2hba` | 510561 | 12.764 microseconds | 4.9183 |
| `2p6j` | 676188 | 16.905 microseconds | 4.2621 |
| `2a3d` | 105149 | 5.257 microseconds | 7.4146 |

This aligns with the paper's argument: simpler systems may still fold under the broader CV framework, but disabling the data-driven `delta` penalty harms the more complex systems and can leave them stuck far from the native state.

Fourth, it creates figures for the paper-style mechanistic analyses:

- TICA phase-space comparison with and without `delta`.
- Evolution of prior weights `theta`, penalty terms `delt`, and posterior weights `W` extracted from log files.
- Cumulative deviation metrics analogous to the paper's `xi` quantities.
- Fixed-`alpha` versus learned-`alpha` phase-space and RMSD comparisons.
- The 10-run rerun analysis for the `2f4k` system.
- CV residue-level sequence visualization.

### `CVgen_vs_others.ipynb`

This notebook compares Adaptive CVgen with two external references.

The first comparison is Adaptive CVgen versus Least Count:

- Loads Adaptive CVgen feature pickles from `./data/<protein>/data_output.pkl`.
- Loads Least Count feature pickles from `../least_count/<protein>/data_output.pkl`.
- Builds TICA projections over concatenated Adaptive CVgen, Least Count, and native-state features.
- Uses 20,000 MiniBatchKMeans clusters to estimate `zeta`, the fraction of discovered state clusters over rounds.
- Produces `AC&LC_<PROTEIN>.png` and `.jpg` figures.

The second comparison is Adaptive CVgen versus D. E. Shaw/Anton trajectories:

- Loads external Anton data from `/home/xhshi/whshen/data/<PROTEIN>/...`.
- Loads Adaptive CVgen comparison trajectories from `./data/compare_DEshaw/<protein>/OUT_PUT/traj`.
- Featurizes both into C-alpha contact features with cutoff `0.8` nm.
- Fits TICA on the Shaw/Anton data, then projects Adaptive CVgen trajectories and native states into that same TICA space.
- Produces `compare_DEshaw_cutoff0.8.png` and `.jpg`.

Stored notebook output shows Shaw/Anton concatenated trajectory lengths:

| Protein | Shaw/Anton projected frames |
|---|---:|
| `1fme` | 1625195 |
| `2f4k` | 627907 |
| `2f21` | 5686712 |
| `2hba` | 14755430 |
| `2p6j` | 1636026 |
| `2a3d` | 3535447 |

## Paper results captured by the analysis

The paper's main claims, as reflected in the notebook analyses, are:

- Adaptive CVgen folded all six protein systems from disordered initial states at 300 K.
- Temperature-matched comparison against Shaw conditions also achieved near-native structures, typically around or below 2 Angstrom RMSD.
- A 10-run test on Villin (`2f4k`) reached the native state in 9 of 10 trials, reported as 90% robustness.
- The C60 synthesis case produced a structure matching standard C60.
- The `delta` penalty/resistance term helps escape local minima by reducing overused or resistant CV directions.
- Learned `alpha` improves efficiency compared with fixed values such as `0`, `0.1`, or `1`.
- Adaptive CVgen explores relevant conformational phase space more effectively than Least Count in the tested protein cases.
- When projected onto TICA modes learned from D. E. Shaw data, Adaptive CVgen covers relevant regions, though some areas are sparse or missing because the method is optimized for efficient targeted exploration rather than exhaustive unbiased occupancy.

The local notebook outputs for the temperature-matched D. E. Shaw comparison are:

| Protein | Temperature setting | First near-min RMSD frame | Approx. aggregate time | Minimum RMSD |
|---|---:|---:|---:|---:|
| `1fme` | 325 K | 124181 | 3.105 microseconds | 1.3934 |
| `2f4k` | 360 K | 59602 | 1.490 microseconds | 0.6721 |
| `2f21` | 360 K | 242000 | 6.050 microseconds | 1.6421 |
| `2hba` | 355 K | 47825 | 1.196 microseconds | 0.3793 |
| `2p6j` | 360 K | 255225 | 6.381 microseconds | 1.6753 |
| `2a3d` | 370 K | 66400 | 3.320 microseconds | 2.0548 |

## Dependencies and reproducibility

The notebooks depend on:

- `numpy`
- `matplotlib`
- `mdtraj`
- `pyemma`
- `deeptime`
- `scipy`
- `scikit-learn`
- `alphashape`
- `shapely`
- `tqdm`
- `openbabel`/`pybel` for part of the Shaw comparison notebook
- trajectory and structure data files not included in the current folder

The notebooks are not currently runnable from the local folder alone because the referenced data directories are absent. The README also says the full source code is not yet released.

Paths expected by the notebooks include:

- `./data/<protein>/...`
- `./data/<protein>/No_delt/...`
- `./data/<protein>/alpha0/...`
- `./data/<protein>/alpha0.1/...`
- `./data/<protein>/alpha1/...`
- `./data/compare_DEshaw/<protein>/...`
- `./iTOL_plot/<protein>/...`
- `../least_count/<protein>/data_output.pkl`
- `/home/xhshi/whshen/data/<PROTEIN>/...`

Because plots are saved with relative filenames such as `CVs_rmsd.png`, `RL_<PROTEIN>.png`, and `compare_DEshaw_cutoff0.8.png`, the output location depends on the notebook's working directory.

## Code review notes

The notebooks are valuable as figure-generation and analysis records, but they are not packaged as a reusable Python project. Key review points:

- The full Adaptive CVgen simulator is absent. The notebooks analyze outputs; they do not implement the full round-by-round MD/seed-selection loop.
- The same `CVgenerator` class is defined twice in `CVgen_data_process.ipynb`. The first version uses C-alpha atoms for CV selections, while the later version uses heavy atoms and writes `CV_structures`. This can change results depending on execution order.
- Several analyses rely on variables computed in earlier cells, so out-of-order execution can silently produce misleading results.
- Some paths are hard-coded to the authors' filesystem, especially the D. E. Shaw/Anton data path.
- In `CVgen_vs_others.ipynb`, the contour-generation cell for Least Count uses `all_tics[i][LC_lens[i]:-1]`. Since `all_out` is concatenated as Adaptive CVgen followed by Least Count followed by native, the expected start index for the Least Count segment appears to be `AC_lens[i]`, not `LC_lens[i]`. The scatter plot later uses `AC_lens[i]:-1`, so this looks like a likely contour-indexing bug.
- The notebooks store large outputs and display data inside `.ipynb`, which makes them large and hard to review.
- There is no environment file, data manifest, or executable reproduction script in this folder.
- The paper references SI Appendix details for SLSQP decorrelation weights and reward-term definitions, but the local folder does not include the SI PDF.

## Practical mental model

Adaptive CVgen can be understood as an adaptive, unbiased MD sampling controller:

1. It builds many residue-level CVs so it has many possible structural directions to monitor.
2. It scores new conformations by how much they improve relevant CV-level structure, usually through inverse RMSD to native segments in the protein examples.
3. It uses historical sampling to penalize directions where the system appears stuck or over-sampled.
4. It learns `alpha` to balance the physical prior reward and this historical penalty.
5. It projects sampled conformations into TICA space, clusters them, and selects high-reward representatives as seeds for the next MD round.

The result is not a biased MD trajectory. It is an adaptive selection process over unbiased MD rounds. The reinforcement learning component decides where to spend future simulation effort by modifying CV weights and seed selection.

## Bottom line

The scientific project is a strong adaptive sampling framework centered on high-dimensional CV generation plus RL-guided reweighting. The RL contribution is the dynamic control of CV importance and seed selection through `delta`, `alpha`, and the learned reward vector `R`.

The local repository, however, is not the full method implementation. It is best treated as a paper companion for processing already-generated trajectories, reproducing figures, comparing ablations, and explaining how CVs, RMSD/native-contact metrics, TICA phase-space projections, and benchmark comparisons were analyzed.
