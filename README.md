# NK Model: 2026 AI-Assisted Reprise

This repository contains a modern, high-performance Rust/Python implementation of the NK spin-glass model, designed to study the absorbing-diffusing phase transition under athermal quasistatic (AQS) cyclic deformation. 

This project is a **2026 AI-assisted rewrite** of the original C implementation created during my [PhD thesis](https://github.com/davidefiocco/phd-thesis). It packages the core physics into a fast Rust simulation engine (`nk_core`) exposed to Python via PyO3, orchestrating large-scale simulations, parallelization, and analysis through a Python pipeline.

## Scientific Background & Findings

This codebase is grounded in the literature findings published in:
- *Fiocco, Foffi, Sastry (2013)* PRE 88, 020301
- *Fiocco, Foffi, Sastry (2014)* PRL 112, 025702
- *PhD Thesis: Oscillatory deformation of amorphous materials: a numerical investigation - https://doi.org/10.5075/epfl-thesis-6101*

The NK model provides a discrete, mathematically tractable energy landscape to study phenomena observed in amorphous solids (like Lennard-Jones glasses) under oscillatory shear. 

### Major Findings Over Original Thesis
In the original thesis work, studying the NK model under AQS deformation yielded crucial insights but was severely bottlenecked. The steepest-descent minimization on a discrete landscape required evaluating $O(N^2)$ adjacent states per step, making large system sizes ($N$) computationally prohibitive without massive parallelization.

This 2026 rewrite overcomes these legacy limitations by combining Rust's performance (including optimized configuration-model graph generation and topological `HashSet` energy updates) with Python multiprocessing. As a result, we can now simulate  **$N=320$** spins and cleanly observe phenomena that were previously obscured by finite-size effects:

1. **Absorbing-Diffusing Transition:** At a critical strain amplitude $\gamma_c$, the system undergoes a sharp transition. Below $\gamma_c$, the system falls into an *absorbing state* (a limit cycle), retaining memory of its initial effective temperature. Above $\gamma_c$, it enters a *diffusing state* (a chaotic-like exploration of the landscape), forgetting its initial conditions.
2. **Finite-Size Scaling:** With the newly accessible large $N$ regimes, the transition sharpness is confirmed to increase with $N$. We can now empirically establish the finite-size scaling of the critical strain, mapped approximately as $\gamma_c \sim 3.2 N^{-0.45}$.

## Key Physics

- **Model**: NK spin glass with $N$ binary spins, $K$-regular random couplings, and a strain parameter $\gamma$ that modulates the energy landscape via $E_i = -\frac{1}{2}(1 + \sin(2\pi(a_i + \gamma b_i)))$.
- **Constraint**: Exactly $N/2$ spins are +1 (conserved magnetization). Dynamics proceed via pair flips (one spin up, one spin down).
- **AQS protocol**: Cycle $\gamma$ between $0$ and $\gamma_{max}$ in steps of $d\gamma$, minimizing energy (steepest descent) at each strain step.
- **Order parameter**: Hamming distance $d$ between configurations one full cycle apart. $d=0$ indicates an absorbing (periodic) state; $d>0$ indicates a diffusing state.

## Installation

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install maturin numpy pandas matplotlib scipy tqdm pyarrow

# Build Rust extension
maturin develop --release
```

## Project Structure

```
nk_core/src/                      # Rust simulation engine
  lib.rs                          #   Python bindings (PyO3)
  lattice.rs                      #   NKLattice, SpinConfig, couplings
  energy.rs                       #   Energy computation and delta-E
  dynamics.rs                     #   MC sweep and steepest-descent minimization
  deformation.rs                  #   AQS oscillatory deformation protocol

scripts/                          # Python pipeline and plotting
  generate_disorders.py           #   Step 1: Create lattices + equilibrate at T
  run_trajectories.py             #   Step 2: Run long AQS deformation trajectories
  analyze_trajectories.py         #   Step 3: Post-process trajectories into phase diagrams
  energy_convergence.py           #   Plot: E/N vs accumulated strain
  order_parameter_evolution.py    #   Plot: d vs accumulated strain
  phase_diagram.py                #   Plot: d/N vs gamma_max
  scaling_analysis.py             #   Cross-N scaling and data collapse
  validate_physics.py             #   Quick physics sanity checks

results/                          # Generated data (gitignored, kept locally)
  disorders/                      #   Equilibrated configs (.npz), by N
  trajectories/                   #   Full AQS trajectories (.parquet), by N

figures/                          # Generated plots (gitignored, regenerable)

pyproject.toml                    # Build config (maturin)
Cargo.toml                        # Rust dependencies
```

## Simulation Pipeline

The canonical workflow that produced all results:

```
generate_disorders.py  -->  run_trajectories.py  -->  analyze_trajectories.py
     (Step 1)                   (Step 2)                  (Step 3)
  Create lattices           Run AQS deformation        Compute d/N vs gamma_max
  Equilibrate at T          Save full trajectories      Plot phase diagrams
  Save .npz                 Save .parquet               Save figures
```

### Step 1: Generate disorder realizations

```bash
# Generate 200 disorders for N=80, equilibrated at T=0.6 and T=1.0
python scripts/generate_disorders.py --N 80 --n-disorders 200

# Quick test
python scripts/generate_disorders.py --N 20 --n-disorders 10
```

### Step 2: Run AQS deformation trajectories

```bash
# Run trajectories for specific gamma_max values
python scripts/run_trajectories.py --N 80 --gamma-max 0.3 0.4 0.5 0.6 --gamma-acc 1000
```

This is the expensive step (~hours to days for large N). Results are saved incrementally
as `.parquet` files per gamma_max value.

### Step 3: Analyze and plot

```bash
# Generate phase diagram from saved trajectories
python scripts/analyze_trajectories.py --N 80

# Plot existing data only (no recomputation)
python scripts/analyze_trajectories.py --N 80 --plot-only
```

### Standalone figure scripts

```bash
# Energy convergence (E/N vs accumulated strain)
python scripts/energy_convergence.py

# Order parameter evolution (d vs accumulated strain)
python scripts/order_parameter_evolution.py

# Phase diagram from processed data
python scripts/phase_diagram.py

# Cross-N scaling analysis (gamma_c vs N, data collapse, order parameter comparison)
python scripts/scaling_analysis.py                    # All 4 figures
python scripts/scaling_analysis.py --figure complete   # 4-panel scaling + log-log
python scripts/scaling_analysis.py --figure collapse   # Rescaled data collapse
```

### Physics validation

```bash
python scripts/validate_physics.py
```

Runs quick checks that absorbing and diffusing states are correctly observed
and that a transition occurs near the expected gamma_c.
