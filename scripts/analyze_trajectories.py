#!/usr/bin/env python3
"""
Analyze Trajectories Script

Post-process saved trajectories to generate:
1. Energy convergence plots (E/N vs γ_acc)
2. Phase diagrams (d vs γ_max) using ONLY equilibrated data

Allows manual specification of γ_eq per γ_max for precise control.

Usage:
    # Plot energy convergence to determine γ_eq values
    python scripts/analyze_trajectories.py --n 40 --energy-plot
    
    # Generate phase diagram with default γ_eq detection
    python scripts/analyze_trajectories.py --n 40 --phase-diagram
    
    # Generate phase diagram with manual γ_eq values
    python scripts/analyze_trajectories.py --n 40 --phase-diagram --gamma-eq "0.3:50,0.4:100,0.5:200"
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Optional
import json

# Directories
RESULTS_DIR = Path(__file__).parent.parent / "results" / "trajectories"
FIGURES_DIR = Path(__file__).parent.parent / "figures"


def load_trajectories(n: int, gamma_max: float) -> pd.DataFrame:
    """Load trajectory data for a specific (N, gamma_max)."""
    filepath = RESULTS_DIR / f"N{n:03d}" / f"gamma_max_{gamma_max:.2f}.parquet"
    if not filepath.exists():
        raise FileNotFoundError(f"No data for N={n}, γ_max={gamma_max}: {filepath}")
    return pd.read_parquet(filepath)


def load_all_trajectories(n: int) -> Dict[float, pd.DataFrame]:
    """Load all trajectory data for a given N."""
    data_dir = RESULTS_DIR / f"N{n:03d}"
    if not data_dir.exists():
        raise FileNotFoundError(f"No data directory for N={n}: {data_dir}")
    
    trajectories = {}
    for f in sorted(data_dir.glob("gamma_max_*.parquet")):
        gamma_max = float(f.stem.replace("gamma_max_", ""))
        trajectories[gamma_max] = pd.read_parquet(f)
    
    if not trajectories:
        raise FileNotFoundError(f"No trajectory files found in {data_dir}")
    
    return trajectories


def detect_equilibration_auto(energy_traj: np.ndarray, gamma_acc_traj: np.ndarray, 
                               window_frac: float = 0.15) -> float:
    """
    Auto-detect equilibration time from energy trajectory.
    
    Uses derivative-based detection: find where |dE/dγ| drops below threshold.
    Returns γ_eq (the γ_acc value after which system is equilibrated).
    """
    n_points = len(energy_traj)
    if n_points < 20:
        return gamma_acc_traj[n_points // 2]
    
    window = max(5, int(n_points * window_frac))
    
    # Compute smoothed derivative
    derivatives = []
    for i in range(window, n_points - window):
        # Linear fit slope in window
        x = gamma_acc_traj[i-window:i+window]
        y = energy_traj[i-window:i+window]
        if len(x) > 1 and (x[-1] - x[0]) > 0:
            slope = (y[-1] - y[0]) / (x[-1] - x[0])
            derivatives.append((gamma_acc_traj[i], abs(slope)))
    
    if not derivatives:
        return gamma_acc_traj[int(n_points * 0.5)]
    
    # Find where derivative becomes small (< 10% of max)
    gammas, derivs = zip(*derivatives)
    derivs = np.array(derivs)
    threshold = 0.1 * np.max(derivs) if np.max(derivs) > 0 else 1e-6
    
    for i, d in enumerate(derivs):
        if d < threshold:
            # Check it stays low
            if np.mean(derivs[i:] < threshold) > 0.7:
                return gammas[i]
    
    # Fallback: use 50% of trajectory
    return gamma_acc_traj[int(n_points * 0.5)]


def plot_energy_convergence(n: int, temperatures: list = [0.6, 1.0], 
                            n_disorders_plot: int = 50, save: bool = True):
    """
    Plot energy convergence curves (E/N vs γ_acc) for all γ_max values.
    Shows individual trajectories + mean to help determine γ_eq.
    """
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    
    trajectories = load_all_trajectories(n)
    gamma_max_values = sorted(trajectories.keys())
    
    # Color map
    cmap = plt.cm.viridis
    colors = {gm: cmap(i / len(gamma_max_values)) for i, gm in enumerate(gamma_max_values)}
    
    fig, axes = plt.subplots(1, len(temperatures), figsize=(7 * len(temperatures), 6), squeeze=False)
    
    for t_idx, T in enumerate(temperatures):
        ax = axes[0, t_idx]
        
        for gamma_max in gamma_max_values:
            df = trajectories[gamma_max]
            df_t = df[df['temperature'] == T]
            
            if df_t.empty:
                continue
            
            # Plot individual trajectories (subset)
            for i, (_, row) in enumerate(df_t.iterrows()):
                if i >= n_disorders_plot:
                    break
                gamma_acc = np.array(row['gamma_acc'])
                energy = np.array(row['energy'])
                ax.plot(gamma_acc, energy, '.', color=colors[gamma_max], 
                       alpha=0.3, markersize=2)
            
            # Plot mean trajectory
            # Interpolate to common grid
            max_gamma = min(row['gamma_acc'][-1] for _, row in df_t.iterrows())
            gamma_grid = np.linspace(1, max_gamma, 200)
            
            energies_interp = []
            for _, row in df_t.iterrows():
                gamma_acc = np.array(row['gamma_acc'])
                energy = np.array(row['energy'])
                interp = np.interp(gamma_grid, gamma_acc, energy)
                energies_interp.append(interp)
            
            mean_energy = np.mean(energies_interp, axis=0)
            ax.plot(gamma_grid, mean_energy, '-', color=colors[gamma_max], 
                   linewidth=2, label=f'γ_max={gamma_max}')
        
        ax.set_xlabel(r'$\gamma_{acc}$', fontsize=12)
        ax.set_ylabel(r'$E/N$', fontsize=12)
        ax.set_title(f'T = {T}', fontsize=14)
        ax.set_xscale('log')
        ax.legend(fontsize=8, loc='upper right', ncol=2)
        ax.grid(True, alpha=0.3)
    
    fig.suptitle(f'Energy Convergence (N={n}, {len(df_t)} disorders)', fontsize=14)
    fig.tight_layout()
    
    if save:
        output_path = FIGURES_DIR / f'energy_convergence_N{n}.png'
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved: {output_path}")
    else:
        plt.show()


def compute_phase_diagram(n: int, gamma_eq_dict: Optional[Dict[float, float]] = None,
                          temperatures: list = [0.6, 1.0]) -> pd.DataFrame:
    """
    Compute phase diagram data: steady-state d/N vs γ_max.
    
    Args:
        n: System size
        gamma_eq_dict: Dict mapping γ_max -> γ_eq. If None, auto-detect.
        temperatures: List of temperatures to include
    
    Returns:
        DataFrame with columns: temperature, gamma_max, d_mean, d_std, d_sem, n_samples
    """
    trajectories = load_all_trajectories(n)
    
    results = []
    
    for gamma_max, df in trajectories.items():
        gamma_eq = gamma_eq_dict.get(gamma_max) if gamma_eq_dict else None
        
        for T in temperatures:
            df_t = df[df['temperature'] == T]
            
            if df_t.empty:
                continue
            
            d_values = []
            for _, row in df_t.iterrows():
                gamma_acc = np.array(row['gamma_acc'])
                d = np.array(row['d'])
                energy = np.array(row['energy'])
                
                # Determine γ_eq for this trajectory
                if gamma_eq is None:
                    traj_gamma_eq = detect_equilibration_auto(energy, gamma_acc)
                else:
                    traj_gamma_eq = gamma_eq
                
                # Extract steady-state d values
                mask = gamma_acc > traj_gamma_eq
                if np.any(mask):
                    d_steady = d[mask]
                    d_values.append(np.mean(d_steady))
            
            if d_values:
                d_arr = np.array(d_values)
                results.append({
                    'temperature': T,
                    'gamma_max': gamma_max,
                    'd_mean': np.mean(d_arr),
                    'd_std': np.std(d_arr),
                    'd_sem': np.std(d_arr) / np.sqrt(len(d_arr)),
                    'n_samples': len(d_arr),
                    'gamma_eq_used': gamma_eq if gamma_eq else 'auto',
                })
    
    return pd.DataFrame(results)


def plot_phase_diagram(n: int, gamma_eq_dict: Optional[Dict[float, float]] = None,
                       temperatures: list = [0.6, 1.0], save: bool = True):
    """
    Plot phase diagram: d/N vs γ_max.
    """
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    
    df = compute_phase_diagram(n, gamma_eq_dict, temperatures)
    
    if df.empty:
        print("No data to plot!")
        return
    
    fig, ax = plt.subplots(figsize=(10, 7))
    
    markers = {0.6: 'o', 1.0: 's'}
    colors_t = {0.6: '#2563eb', 1.0: '#dc2626'}
    
    for T in temperatures:
        df_t = df[df['temperature'] == T]
        if df_t.empty:
            continue
        
        ax.errorbar(
            df_t['gamma_max'],
            df_t['d_mean'],
            yerr=df_t['d_sem'],
            marker=markers.get(T, 'o'),
            color=colors_t.get(T, 'black'),
            label=f'T = {T}',
            capsize=4,
            linewidth=2,
            markersize=8,
            markerfacecolor='white',
            markeredgewidth=2,
        )
    
    ax.set_xlabel(r'$\gamma_{\max}$', fontsize=14)
    ax.set_ylabel(r'$d/N$', fontsize=14)
    ax.set_title(f'Phase Diagram (N={n})', fontsize=16)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(x=0.5, color='gray', linestyle=':', alpha=0.5)
    ax.set_xlim(0.05, 1.05)
    ax.set_ylim(-0.02, 0.55)
    
    # Add γ_eq info
    if gamma_eq_dict:
        info_text = "γ_eq: " + ", ".join([f"{gm}:{ge}" for gm, ge in sorted(gamma_eq_dict.items())])
        ax.text(0.02, 0.98, info_text, transform=ax.transAxes, fontsize=8,
               verticalalignment='top', fontfamily='monospace', alpha=0.7)
    
    fig.tight_layout()
    
    if save:
        output_path = FIGURES_DIR / f'phase_diagram_N{n}.png'
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved: {output_path}")
    else:
        plt.show()
    
    # Print summary
    print(f"\nPhase Diagram Summary (N={n}):")
    print(df.to_string(index=False))


def parse_gamma_eq(gamma_eq_str: str) -> Dict[float, float]:
    """
    Parse γ_eq string like "0.3:50,0.4:100,0.5:200" into dict.
    """
    if not gamma_eq_str:
        return {}
    
    result = {}
    for pair in gamma_eq_str.split(','):
        gm, ge = pair.split(':')
        result[float(gm)] = float(ge)
    return result


def main():
    parser = argparse.ArgumentParser(description='Analyze NK Model Trajectories')
    parser.add_argument('--n', type=int, required=True, help='System size N')
    parser.add_argument('--energy-plot', action='store_true', help='Plot energy convergence')
    parser.add_argument('--phase-diagram', action='store_true', help='Generate phase diagram')
    parser.add_argument('--gamma-eq', type=str, default=None,
                       help='Manual γ_eq values: "γ_max1:γ_eq1,γ_max2:γ_eq2,..." (default: auto)')
    parser.add_argument('--no-save', action='store_true', help='Show plots instead of saving')
    args = parser.parse_args()
    
    gamma_eq_dict = parse_gamma_eq(args.gamma_eq) if args.gamma_eq else None
    save = not args.no_save
    
    print(f"Analyzing N={args.n} trajectories...")
    print(f"Data dir: {RESULTS_DIR / f'N{args.n:03d}'}")
    
    if gamma_eq_dict:
        print(f"Manual γ_eq values: {gamma_eq_dict}")
    else:
        print("Using auto-detected γ_eq values")
    
    if args.energy_plot:
        plot_energy_convergence(args.n, save=save)
    
    if args.phase_diagram:
        plot_phase_diagram(args.n, gamma_eq_dict, save=save)
    
    if not args.energy_plot and not args.phase_diagram:
        print("No action specified. Use --energy-plot and/or --phase-diagram")


if __name__ == '__main__':
    main()





