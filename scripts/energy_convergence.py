#!/usr/bin/env python3
"""
Energy Convergence Plot (Figure 3 from paper)

Reproduces: E/N vs γ_acc for different temperatures and strain amplitudes.
Shows when the system reaches "deformation equilibrium".

REQUIRES: Run generate_disorders.py first to create disorder checkpoints.

Usage:
    python scripts/generate_disorders.py --n 20 --n-disorders 200  # First!
    python scripts/energy_convergence.py --n 20                    # Run N=20
    python scripts/energy_convergence.py --n 20 --plot             # Plot only
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import multiprocessing as mp
import time

sys.path.insert(0, str(Path(__file__).parent.parent))
import nk_core
from generate_disorders import load_disorder, get_disorder_dir

# Output directories
RESULTS_DIR = Path(__file__).parent.parent / 'results'
FIGURE_DIR = Path(__file__).parent.parent / 'figures'

# Default parameters
GAMMA_MAX_VALUES = [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0]
TEMPERATURES = [0.6, 1.0]


def run_disorder_simulations(
    n: int,
    disorder_id: int,
    temperature: float,
    gamma_max_values: list,
    gamma_acc_target: float,
    dgamma: float = 0.02,
    n_cycles_initial: int = 20,
) -> list:
    """
    Run simulations for ALL γ_max values for a single disorder/temperature.
    
    This minimizes I/O overhead by loading the disorder once.
    """
    # Load disorder + equilibrated config (ONCE)
    try:
        disorder = load_disorder(n, disorder_id, temperature)
    except FileNotFoundError:
        return []
        
    k = disorder['k']
    results = []
    
    for gamma_max in gamma_max_values:
        steps_per_cycle = int(4 * gamma_max / dgamma)
        n_cycles_max = int(gamma_acc_target / (4 * gamma_max))
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
        
        gamma_acc = np.array(deform_result['gamma_acc'])
        energy = np.array(deform_result['energy'])
        d_values = hamming_cycle
        
        if is_absorbing and n_cycles_run < n_cycles_max:
            # PAD trajectory
            n_cycles_to_pad = n_cycles_max - n_cycles_run
            last_energy = energy[-1]
            last_gamma_acc = gamma_acc[-1]
            
            padded_gamma_acc = [last_gamma_acc + (i+1)*cycle_strain for i in range(n_cycles_to_pad)]
            padded_energy = [last_energy] * n_cycles_to_pad
            padded_d = [0.0] * n_cycles_to_pad
            
            gamma_acc = np.concatenate([gamma_acc, padded_gamma_acc])
            energy = np.concatenate([energy, padded_energy])
            d_values = np.concatenate([d_values, padded_d])
            
        elif not is_absorbing and n_cycles_run < n_cycles_max:
            # Need full run - continue from scratch (fast enough)
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
            
            gamma_acc = np.array(deform_result['gamma_acc'])
            energy = np.array(deform_result['energy'])
            d_values = np.array(deform_result['hamming_cycle'])
        
        # Build DataFrame
        df = pd.DataFrame({
            'gamma_acc': gamma_acc,
            'energy': energy / n,
            'hamming_cycle': d_values,
            'd': d_values / n,
        })
        df['disorder_id'] = disorder_id
        df['n'] = n
        df['k'] = k
        df['temperature'] = temperature
        df['gamma_max'] = gamma_max
        df['seed'] = disorder['seed']
        df['is_absorbing'] = is_absorbing
        
        results.append(df)
    
    return results


def _worker(args):
    """Worker for single (disorder, T) task."""
    n, disorder_id, temperature, gamma_max_values, gamma_acc_target = args
    try:
        results = run_disorder_simulations(
            n=n,
            disorder_id=disorder_id,
            temperature=temperature,
            gamma_max_values=gamma_max_values,
            gamma_acc_target=gamma_acc_target,
        )
        if not results:
            return [], f"D{disorder_id:03d} T={temperature}: MISSING"
        
        # Count absorbing
        n_abs = sum(1 for df in results if df['is_absorbing'].iloc[0])
        return results, f"D{disorder_id:03d} T={temperature}: done ({n_abs}/{len(results)} absorbing)"
    except Exception as e:
        return [], f"D{disorder_id:03d} T={temperature}: ERROR - {e}"


def generate_data(
    n: int,
    temperatures: list = None,
    gamma_max_values: list = None,
    gamma_acc_target: float = 300.0,
    n_disorders: int = 200,
    force: bool = False,
    n_workers: int = None,
) -> pd.DataFrame:
    """Generate trajectory data."""
    
    if temperatures is None:
        temperatures = TEMPERATURES
    if gamma_max_values is None:
        gamma_max_values = GAMMA_MAX_VALUES
    
    data_dir = RESULTS_DIR / 'energy_convergence' / f'N{n:03d}'
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Check for existing data
    data_file = data_dir / f'energy_conv_M{n_disorders}.parquet'
    if data_file.exists() and not force:
        print(f"Loading cached data from {data_file}")
        return pd.read_parquet(data_file)
    
    # Build task list: one task per (disorder, temperature)
    # Fewer tasks, less overhead, better I/O pattern
    tasks = []
    for d_idx in range(n_disorders):
        for T in temperatures:
            tasks.append((n, d_idx, T, gamma_max_values, gamma_acc_target))
    
    if n_workers is None:
        n_workers = 8
    
    n_total = len(tasks)
    print(f"Running {n_total} disorder-tasks on {n_workers} cores...")
    print(f"  N={n}, {n_disorders} disorders × {len(temperatures)} T (each runs {len(gamma_max_values)} γ_max)")
    print()
    
    # Run in parallel
    all_dfs = []
    completed = 0
    
    start_time = time.time()
    
    with mp.Pool(n_workers) as pool:
        for results, msg in pool.imap_unordered(_worker, tasks):
            completed += 1
            all_dfs.extend(results)
            
            if completed % 10 == 0 or completed == n_total:
                elapsed = time.time() - start_time
                rate = completed / elapsed
                remaining = (n_total - completed) / rate if rate > 0 else 0
                print(f"  [{completed:3d}/{n_total}] {msg} (ETA: {remaining/60:.1f}m)")
    
    if not all_dfs:
        print("\nERROR: No data generated!")
        return pd.DataFrame()
    
    df = pd.concat(all_dfs, ignore_index=True)
    
    # Save
    df.to_parquet(data_file, index=False)
    print(f"\n✓ Saved {len(df)} rows to {data_file}")
    
    return df


def plot_energy_convergence(df: pd.DataFrame):
    """Plot energy convergence with disorder averaging."""
    
    if df.empty:
        print("No data to plot!")
        return None
    
    gamma_max_values = sorted(df['gamma_max'].unique())
    temperatures = sorted(df['temperature'].unique())
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(gamma_max_values)))
    color_map = dict(zip(gamma_max_values, colors))
    
    fig, axes = plt.subplots(1, len(temperatures), figsize=(6*len(temperatures), 5), sharey=True)
    if len(temperatures) == 1:
        axes = [axes]
    
    # Create a common grid for averaging
    gamma_acc_grid = np.linspace(0, df['gamma_acc'].max(), 1000)
    
    for ax, T in zip(axes, temperatures):
        df_t = df[df['temperature'] == T]
        
        for gamma_max in gamma_max_values:
            df_g = df_t[df_t['gamma_max'] == gamma_max]
            
            # Interpolate each trajectory to common grid
            energies_on_grid = []
            
            for _, traj in df_g.groupby(['disorder_id', 'seed']):
                # Sort by gamma_acc
                traj = traj.sort_values('gamma_acc')
                x = traj['gamma_acc'].values
                y = traj['energy'].values
                
                # Interpolate, holding last value constant (forward fill for absorbed)
                y_interp = np.interp(gamma_acc_grid, x, y, right=y[-1])
                energies_on_grid.append(y_interp)
            
            if not energies_on_grid:
                continue
                
            # Compute mean and std
            energies_arr = np.array(energies_on_grid)
            energy_mean = np.mean(energies_arr, axis=0)
            
            # Plot mean only (cleaner)
            ax.plot(
                gamma_acc_grid,
                energy_mean,
                color=color_map[gamma_max],
                label=f'γ_max={gamma_max:.2f}',
                linewidth=2.0,
            )
        
        ax.set_xlabel(r'$\gamma_{acc}$', fontsize=14)
        ax.set_xscale('log')
        ax.set_xlim(1, df['gamma_acc'].max())
        ax.grid(True, alpha=0.3)
        ax.set_title(f'T = {T}', fontsize=14)
    
    axes[0].set_ylabel(r'$E/N$', fontsize=14)
    axes[-1].legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=9)
    
    n = df['n'].iloc[0]
    k = df['k'].iloc[0]
    n_disorders = df['disorder_id'].nunique()
    fig.suptitle(f'Energy Convergence (N={n}, K={k}, {n_disorders} disorders)', fontsize=14, y=1.02)
    
    plt.tight_layout()
    
    # Save figure
    FIGURE_DIR.mkdir(exist_ok=True)
    output_path = FIGURE_DIR / f'energy_convergence_N{n}.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Figure saved to: {output_path}")
    
    return fig


def main():
    parser = argparse.ArgumentParser(description='Energy convergence analysis (Fig 3)')
    parser.add_argument('--n', type=int, required=True, help='System size N')
    parser.add_argument('--plot', action='store_true', help='Only plot existing data')
    parser.add_argument('--force', action='store_true', help='Force regenerate all data')
    parser.add_argument('--n-disorders', type=int, default=200, help='Number of disorders')
    parser.add_argument('--gamma-acc', type=float, default=300.0, help='Target γ_acc (default: 300)')
    parser.add_argument('--workers', type=int, default=8, help='Number of workers (default: 8)')
    args = parser.parse_args()
    
    print(f"Energy Convergence Analysis (Fig 3)")
    print(f"{'='*50}")
    print(f"N={args.n}, γ_acc={args.gamma_acc}, disorders={args.n_disorders}")
    print(f"Disorder dir: {get_disorder_dir(args.n)}")
    print()
    
    if args.plot:
        data_dir = RESULTS_DIR / 'energy_convergence' / f'N{args.n:03d}'
        parquet_files = list(data_dir.glob('*.parquet'))
        if not parquet_files:
            print(f"No data files found in {data_dir}. Run without --plot first.")
            return
        df = pd.concat([pd.read_parquet(f) for f in parquet_files], ignore_index=True)
        print(f"Loaded {len(df)} data points")
    else:
        df = generate_data(
            n=args.n,
            gamma_acc_target=args.gamma_acc,
            n_disorders=args.n_disorders,
            force=args.force,
            n_workers=args.workers,
        )
    
    plot_energy_convergence(df)
    plt.show()


if __name__ == '__main__':
    main()
