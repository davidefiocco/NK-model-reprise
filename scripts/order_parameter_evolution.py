#!/usr/bin/env python3
"""
Order Parameter Evolution (Figure 4 top from paper)

Reproduces: Order parameter d vs (γ_acc - γ̃_acc) for different γ_max.
Shows transient behavior and approach to steady state.

REQUIRES: Run generate_disorders.py first to create disorder checkpoints.

Usage:
    python scripts/generate_disorders.py --n 20 --n-disorders 10   # First!
    python scripts/order_parameter_evolution.py --n 20             # Run N=20
    python scripts/order_parameter_evolution.py --n 20 --plot      # Plot only
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
GAMMA_MAX_VALUES = [0.3, 0.4, 0.5, 0.6, 0.8, 1.0]  # Fewer for clearer visualization
TEMPERATURES = [0.6, 1.0]


def run_order_evolution(
    n: int,
    disorder_id: int,
    temperature: float,
    gamma_max_values: list,
    gamma_acc_target: float = 200.0,
    dgamma: float = 0.02,
) -> list:
    """
    Run deformation and track order parameter d over time.
    """
    # Load disorder + equilibrated config
    disorder = load_disorder(n, disorder_id, temperature)
    k = disorder['k']
    
    results = []
    
    for gamma_max in gamma_max_values:
        steps_per_cycle = int(4 * gamma_max / dgamma)
        n_cycles = int(gamma_acc_target / (4 * gamma_max))
        total_steps = steps_per_cycle * n_cycles
        
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
        
        df = pd.DataFrame({
            'gamma_acc': deform_result['gamma_acc'],
            'd': np.array(deform_result['hamming_cycle']) / n,
            'energy': np.array(deform_result['energy']) / n,
        })
        df['disorder_id'] = disorder_id
        df['n'] = n
        df['k'] = k
        df['temperature'] = temperature
        df['gamma_max'] = gamma_max
        
        results.append(df)
    
    return results


def _worker(args):
    """Worker for parallel execution."""
    n, disorder_id, temperature, gamma_max_values, gamma_acc_target = args
    try:
        results = run_order_evolution(
            n=n,
            disorder_id=disorder_id,
            temperature=temperature,
            gamma_max_values=gamma_max_values,
            gamma_acc_target=gamma_acc_target,
        )
        return results, f"D{disorder_id:03d} T={temperature}: done"
    except FileNotFoundError:
        return [], f"D{disorder_id:03d} T={temperature}: MISSING disorder"
    except Exception as e:
        return [], f"D{disorder_id:03d} T={temperature}: ERROR - {e}"


def generate_data(
    n: int,
    temperatures: list = None,
    gamma_max_values: list = None,
    gamma_acc_target: float = 200.0,
    n_disorders: int = 5,  # Few disorders for trajectory visualization
    force: bool = False,
    n_workers: int = None,
) -> pd.DataFrame:
    """Generate order parameter evolution data."""
    
    if temperatures is None:
        temperatures = TEMPERATURES
    if gamma_max_values is None:
        gamma_max_values = GAMMA_MAX_VALUES
    
    data_dir = RESULTS_DIR / 'order_evolution' / f'N{n:03d}'
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Check for existing data
    data_file = data_dir / f'order_evolution_M{n_disorders}.parquet'
    if data_file.exists() and not force:
        print(f"Loading cached data from {data_file}")
        return pd.read_parquet(data_file)
    
    # Build task list
    tasks = []
    for d_idx in range(n_disorders):
        for T in temperatures:
            tasks.append((n, d_idx, T, gamma_max_values, gamma_acc_target))
    
    if n_workers is None:
        n_workers = 8
    
    print(f"Running {len(tasks)} disorder-tasks on {n_workers} cores...")
    
    # Run in parallel
    all_dfs = []
    completed = 0
    with mp.Pool(n_workers) as pool:
        for results, msg in pool.imap_unordered(_worker, tasks):
            completed += 1
            print(f"  [{completed}/{len(tasks)}] {msg}")
            all_dfs.extend(results)
    
    if not all_dfs:
        print("\nERROR: No data generated!")
        return pd.DataFrame()
    
    df = pd.concat(all_dfs, ignore_index=True)
    
    # Save
    df.to_parquet(data_file, index=False)
    print(f"✓ Saved {len(df)} rows to {data_file}")
    
    return df


def plot_order_evolution(df: pd.DataFrame):
    """Plot order parameter d vs γ_acc."""
    
    if df.empty:
        print("No data to plot!")
        return None
    
    temperatures = sorted(df['temperature'].unique())
    gamma_max_values = sorted(df['gamma_max'].unique())
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(gamma_max_values)))
    color_map = dict(zip(gamma_max_values, colors))
    
    fig, axes = plt.subplots(1, len(temperatures), figsize=(6*len(temperatures), 5), sharey=True)
    if len(temperatures) == 1:
        axes = [axes]
    
    for ax, T in zip(axes, temperatures):
        df_t = df[df['temperature'] == T]
        
        for gamma_max in gamma_max_values:
            df_g = df_t[df_t['gamma_max'] == gamma_max]
            
            # Average over disorders
            df_avg = df_g.groupby('gamma_acc').agg({
                'd': ['mean', 'std']
            }).reset_index()
            df_avg.columns = ['gamma_acc', 'd_mean', 'd_std']
            
            # Subsample for plotting
            step = max(1, len(df_avg) // 50)
            
            ax.plot(
                df_avg['gamma_acc'].values[::step],
                df_avg['d_mean'].values[::step],
                color=color_map[gamma_max],
                label=f'γ_max={gamma_max:.2f}',
                linewidth=1.5,
            )
            ax.fill_between(
                df_avg['gamma_acc'].values[::step],
                (df_avg['d_mean'] - df_avg['d_std']).values[::step],
                (df_avg['d_mean'] + df_avg['d_std']).values[::step],
                color=color_map[gamma_max],
                alpha=0.2,
            )
        
        ax.set_xlabel(r'$\gamma_{acc}$', fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.set_title(f'T = {T}', fontsize=14)
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    
    axes[0].set_ylabel(r'Order parameter $d/N$', fontsize=14)
    axes[-1].legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=10)
    
    n = df['n'].iloc[0]
    k = df['k'].iloc[0]
    n_disorders = df['disorder_id'].nunique()
    fig.suptitle(f'Order Parameter Evolution (N={n}, K={k}, {n_disorders} disorders)', fontsize=14, y=1.02)
    
    plt.tight_layout()
    
    # Save figure
    FIGURE_DIR.mkdir(exist_ok=True)
    output_path = FIGURE_DIR / f'order_parameter_evolution_N{n}.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Figure saved to: {output_path}")
    
    return fig


def main():
    parser = argparse.ArgumentParser(description='Order parameter evolution (Fig 4 top)')
    parser.add_argument('--n', type=int, required=True, help='System size N')
    parser.add_argument('--plot', action='store_true', help='Only plot existing data')
    parser.add_argument('--force', action='store_true', help='Force regenerate all data')
    parser.add_argument('--n-disorders', type=int, default=5, help='Number of disorders (default: 5)')
    parser.add_argument('--gamma-acc', type=float, default=200.0, help='Target γ_acc')
    parser.add_argument('--workers', type=int, default=8, help='Number of parallel workers')
    args = parser.parse_args()
    
    print(f"Order Parameter Evolution (Fig 4 top)")
    print(f"{'='*50}")
    print(f"N={args.n}, γ_acc={args.gamma_acc}, disorders={args.n_disorders}")
    print(f"Disorder dir: {get_disorder_dir(args.n)}")
    print()
    
    if args.plot:
        data_dir = RESULTS_DIR / 'order_evolution' / f'N{args.n:03d}'
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
    
    plot_order_evolution(df)
    plt.show()


if __name__ == '__main__':
    main()
