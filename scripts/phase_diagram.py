#!/usr/bin/env python3
"""
Phase Diagram (Figure 4 bottom from paper)

Reproduces: Steady-state order parameter d vs γ_max for different system sizes.
Shows the absorbing-diffusing phase transition.

REQUIRES: Run generate_disorders.py first to create disorder checkpoints.

Usage:
    python scripts/generate_disorders.py --n 20 --n-disorders 200  # First!
    python scripts/phase_diagram.py --n 20                         # Run N=20
    python scripts/phase_diagram.py --n 20 --plot                  # Plot only
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import multiprocessing as mp

sys.path.insert(0, str(Path(__file__).parent.parent))
import nk_core
from generate_disorders import load_disorder, get_disorder_dir

# Output directories
RESULTS_DIR = Path(__file__).parent.parent / 'results'
FIGURE_DIR = Path(__file__).parent.parent / 'figures'

# Default parameters
GAMMA_MAX_VALUES = [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0]
TEMPERATURES = [0.6, 1.0]


def run_disorder_phase_scan(
    n: int,
    disorder_id: int,
    temperature: float,
    gamma_max_values: list,
    gamma_acc_total: float = 100.0,
    dgamma: float = 0.02,
    n_steady: int = 10,
    n_cycles_initial: int = 20,
) -> list:
    """
    Run deformation for all γ_max values for a single disorder.
    Extract steady-state d.
    """
    try:
        disorder = load_disorder(n, disorder_id, temperature)
    except FileNotFoundError:
        return []
        
    k = disorder['k']
    results = []
    
    for gamma_max in gamma_max_values:
        steps_per_cycle = int(4 * gamma_max / dgamma)
        n_cycles_max = int(gamma_acc_total / (4 * gamma_max))
        cycle_strain = 4 * gamma_max
        
        # Start with fewer cycles to check for absorbing state
        n_cycles_run = min(n_cycles_initial, n_cycles_max)
        total_steps = steps_per_cycle * n_cycles_run
        
        deform_result = nk_core.run_deformation(
            n=n,
            k=k,
            couplings=disorder['couplings'].tolist(),
            a=disorder['a'].tolist(),
            b=disorder['b'].tolist(),
            initial_spins=disorder['config'].tolist(),
            gamma_max=gamma_max,
            dgamma=dgamma,
            total_steps=total_steps,
            dump_zero_strain_only=True
        )
        
        # Check if absorbing state reached
        hamming_cycle = np.array(deform_result['hamming_cycle'])
        n_check = min(5, len(hamming_cycle) - 1)
        is_absorbing = n_check > 0 and np.all(hamming_cycle[-n_check:] == 0)
        
        energy = np.array(deform_result['energy']) / n
        hysteresis = deform_result['hysteresis_area']
        
        if is_absorbing:
            # System is absorbing - d=0, energy constant
            steady_d = np.zeros(n_steady)
            steady_energy = np.full(n_steady, energy[-1])
        elif n_cycles_run < n_cycles_max:
            # Not absorbing - need full run
            total_steps_full = steps_per_cycle * n_cycles_max
            
            deform_result = nk_core.run_deformation(
                n=n,
                k=k,
                couplings=disorder['couplings'].tolist(),
                a=disorder['a'].tolist(),
                b=disorder['b'].tolist(),
                initial_spins=disorder['config'].tolist(),
                gamma_max=gamma_max,
                dgamma=dgamma,
                total_steps=total_steps_full,
                dump_zero_strain_only=True
            )
            
            hamming_cycle = np.array(deform_result['hamming_cycle'])
            energy = np.array(deform_result['energy']) / n
            hysteresis = deform_result['hysteresis_area']
            
            n_use = min(n_steady, len(hamming_cycle) - 1)
            steady_d = hamming_cycle[-n_use:] / n
            steady_energy = energy[-n_use:]
        else:
            # Already ran enough cycles
            n_use = min(n_steady, len(hamming_cycle) - 1)
            steady_d = hamming_cycle[-n_use:] / n
            steady_energy = energy[-n_use:]
        
        results.append({
            'disorder_id': disorder_id,
            'n': n,
            'k': k,
            'temperature': temperature,
            'gamma_max': gamma_max,
            'd_mean': np.mean(steady_d),
            'd_std': np.std(steady_d),
            'energy_mean': np.mean(steady_energy),
            'energy_std': np.std(steady_energy),
            'hysteresis_area': hysteresis,
            'is_absorbing': is_absorbing,
        })
    
    return results


def _worker(args):
    """Worker for parallel execution."""
    n, disorder_id, temperature, gamma_max_values, gamma_acc_total = args
    try:
        results = run_disorder_phase_scan(
            n=n,
            disorder_id=disorder_id,
            temperature=temperature,
            gamma_max_values=gamma_max_values,
            gamma_acc_total=gamma_acc_total,
        )
        if not results:
            return [], f"D{disorder_id:03d} T={temperature}: MISSING"
        
        n_abs = sum(1 for r in results if r['is_absorbing'])
        return results, f"D{disorder_id:03d} T={temperature}: done ({n_abs}/{len(results)} abs)"
    except Exception as e:
        return [], f"D{disorder_id:03d} T={temperature}: ERROR - {e}"


def generate_phase_data(
    n: int,
    temperatures: list = None,
    gamma_max_values: list = None,
    gamma_acc_total: float = 100.0,
    n_disorders: int = 200,
    force: bool = False,
    n_workers: int = None,
) -> pd.DataFrame:
    """Generate phase diagram data for system size N."""
    
    if temperatures is None:
        temperatures = TEMPERATURES
    if gamma_max_values is None:
        gamma_max_values = GAMMA_MAX_VALUES
    
    data_dir = RESULTS_DIR / 'phase_diagram' / f'N{n:03d}'
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Check for existing data
    data_file = data_dir / f'phase_diagram_M{n_disorders}.parquet'
    if data_file.exists() and not force:
        print(f"Loading cached data from {data_file}")
        return pd.read_parquet(data_file)
    
    # Build task list: one task per (disorder, temperature)
    tasks = []
    for d_idx in range(n_disorders):
        for T in temperatures:
            tasks.append((n, d_idx, T, gamma_max_values, gamma_acc_total))
    
    if n_workers is None:
        n_workers = 8
    
    print(f"Running {len(tasks)} disorder-tasks on {n_workers} cores...")
    
    # Run in parallel
    all_results = []
    completed = 0
    with mp.Pool(n_workers) as pool:
        for results, msg in pool.imap_unordered(_worker, tasks):
            completed += 1
            print(f"  [{completed}/{len(tasks)}] {msg}")
            all_results.extend(results)
    
    if not all_results:
        print("\nERROR: No data generated!")
        return pd.DataFrame()
    
    df = pd.DataFrame(all_results)
    
    # Save
    df.to_parquet(data_file, index=False)
    print(f"✓ Saved {len(df)} rows to {data_file}")
    
    return df


def plot_phase_diagram(df: pd.DataFrame):
    """Plot phase diagram: d vs γ_max."""
    
    if df.empty:
        print("No data to plot!")
        return None
    
    temperatures = sorted(df['temperature'].unique())
    
    fig, axes = plt.subplots(1, len(temperatures), figsize=(6*len(temperatures), 5), sharey=True)
    if len(temperatures) == 1:
        axes = [axes]
    
    for ax, T in zip(axes, temperatures):
        df_t = df[df['temperature'] == T]
        
        # Average over disorders
        df_avg = df_t.groupby('gamma_max').agg({
            'd_mean': ['mean', 'std'],
            'energy_mean': ['mean', 'std'],
        }).reset_index()
        df_avg.columns = ['gamma_max', 'd_avg', 'd_err', 'energy_avg', 'energy_err']
        
        # Error on mean
        n_disorders = df_t['disorder_id'].nunique()
        df_avg['d_err'] = df_avg['d_err'] / np.sqrt(n_disorders)
        
        ax.errorbar(
            df_avg['gamma_max'],
            df_avg['d_avg'],
            yerr=df_avg['d_err'],
            marker='o',
            markersize=6,
            capsize=3,
            linewidth=1.5,
            label=f'T={T}'
        )
        
        ax.set_xlabel(r'$\gamma_{max}$', fontsize=14)
        ax.set_xlim(0.25, 1.05)
        ax.grid(True, alpha=0.3)
        ax.set_title(f'T = {T}', fontsize=14)
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    
    axes[0].set_ylabel(r'Order parameter $d/N$', fontsize=14)
    
    n = df['n'].iloc[0]
    k = df['k'].iloc[0]
    n_disorders = df['disorder_id'].nunique()
    fig.suptitle(f'Phase Diagram (N={n}, K={k}, {n_disorders} disorders)', fontsize=14, y=1.02)
    
    plt.tight_layout()
    
    # Save figure
    FIGURE_DIR.mkdir(exist_ok=True)
    output_path = FIGURE_DIR / f'phase_diagram_N{n}.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Figure saved to: {output_path}")
    
    return fig


def main():
    parser = argparse.ArgumentParser(description='Phase diagram analysis (Fig 4 bottom)')
    parser.add_argument('--n', type=int, required=True, help='System size N')
    parser.add_argument('--plot', action='store_true', help='Only plot existing data')
    parser.add_argument('--force', action='store_true', help='Force regenerate all data')
    parser.add_argument('--n-disorders', type=int, default=200, help='Number of disorders')
    parser.add_argument('--gamma-acc', type=float, default=100.0, help='Total γ_acc per run')
    parser.add_argument('--workers', type=int, default=8, help='Number of workers')
    args = parser.parse_args()
    
    print(f"Phase Diagram Analysis (Fig 4 bottom)")
    print(f"{'='*50}")
    print(f"N={args.n}, γ_acc={args.gamma_acc}, disorders={args.n_disorders}")
    print(f"Disorder dir: {get_disorder_dir(args.n)}")
    print()
    
    if args.plot:
        data_dir = RESULTS_DIR / 'phase_diagram' / f'N{args.n:03d}'
        parquet_files = list(data_dir.glob('*.parquet'))
        if not parquet_files:
            print(f"No data files found in {data_dir}. Run without --plot first.")
            return
        df = pd.concat([pd.read_parquet(f) for f in parquet_files], ignore_index=True)
        print(f"Loaded {len(df)} data points")
    else:
        df = generate_phase_data(
            n=args.n,
            gamma_acc_total=args.gamma_acc,
            n_disorders=args.n_disorders,
            force=args.force,
            n_workers=args.workers,
        )
    
    plot_phase_diagram(df)
    plt.show()


if __name__ == '__main__':
    main()
