#!/usr/bin/env python3
"""
Run Trajectories Script

Runs long simulations and saves FULL trajectories for later analysis.
Decouples expensive simulation from analysis/plotting.

Output: One parquet file per (N, gamma_max) containing all disorders and temperatures.
    Columns: disorder_id, temperature, gamma_acc, energy, d (full_cycle_displacement/N)

Usage:
    python scripts/run_trajectories.py --n 20 --n-disorders 200 --gamma-acc 1000 --workers 8
    python scripts/run_trajectories.py --n 40 --n-disorders 200 --gamma-acc 1000 --workers 8
    python scripts/run_trajectories.py --n 80 --n-disorders 100 --gamma-acc 1000 --workers 8
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from multiprocessing import Pool, cpu_count
import time
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
import nk_core
from scripts.generate_disorders import load_disorder, get_disorder_dir

# Output directory
RESULTS_DIR = Path(__file__).parent.parent / "results" / "trajectories"


def run_single_trajectory(
    n: int,
    disorder_id: int,
    temperature: float,
    gamma_max: float,
    gamma_acc_target: float,
    dgamma: float = 0.02,
) -> dict:
    """
    Run ONE trajectory and return the FULL time series.
    
    Returns dict with:
        disorder_id, temperature, gamma_max,
        gamma_acc (array), energy (array), d (array)
    """
    disorder = load_disorder(n, disorder_id, temperature)
    k = disorder['k']
    
    # Calculate steps
    steps_per_cycle = int(4 * gamma_max / dgamma)
    n_cycles = int(gamma_acc_target / (4 * gamma_max)) + 1
    total_steps = n_cycles * steps_per_cycle
    
    # Run deformation - dump at zero-strain points only
    result = nk_core.run_deformation(
        n=n, k=k,
        couplings=disorder['couplings'].tolist(),
        a=disorder['a'].tolist(),
        b=disorder['b'].tolist(),
        initial_spins=disorder['config'].tolist(),
        gamma_max=gamma_max,
        dgamma=dgamma,
        total_steps=total_steps,
        dump_zero_strain_only=True,
    )
    
    # Extract arrays
    gamma_acc = np.array(result['gamma_acc'])
    energy = np.array(result['energy']) / n  # E/N
    d = np.array(result['hamming_cycle']) / n  # d/N
    
    return {
        'disorder_id': disorder_id,
        'temperature': temperature,
        'gamma_acc': gamma_acc,
        'energy': energy,
        'd': d,
    }


def _worker(args):
    """Worker function for parallel execution."""
    n, disorder_id, temperature, gamma_max, gamma_acc_target = args
    try:
        return run_single_trajectory(n, disorder_id, temperature, gamma_max, gamma_acc_target)
    except FileNotFoundError as e:
        print(f"Missing: disorder={disorder_id}, T={temperature}: {e}", flush=True)
        return None
    except Exception as e:
        print(f"Error: disorder={disorder_id}, T={temperature}, γ_max={gamma_max}: {e}", flush=True)
        import traceback
        traceback.print_exc()
        return None


def run_for_gamma_max(
    n: int,
    gamma_max: float,
    temperatures: list,
    gamma_acc_target: float,
    n_disorders: int,
    n_workers: int,
    output_dir: Path,
) -> Path:
    """
    Run all trajectories for a single gamma_max value and save to parquet.
    Returns path to saved file.
    """
    output_file = output_dir / f"gamma_max_{gamma_max:.2f}.parquet"
    
    # Build task list
    tasks = []
    for d_idx in range(n_disorders):
        for T in temperatures:
            tasks.append((n, d_idx, T, gamma_max, gamma_acc_target))
    
    n_tasks = len(tasks)
    print(f"\n  γ_max = {gamma_max}: {n_tasks} trajectories...", flush=True)
    
    all_results = []
    start_time = time.time()
    
    with Pool(n_workers) as pool:
        for i, result in enumerate(pool.imap_unordered(_worker, tasks)):
            if result is not None:
                all_results.append(result)
            
            # Progress every 25%
            if (i + 1) % max(1, n_tasks // 4) == 0:
                elapsed = time.time() - start_time
                print(f"    {i+1}/{n_tasks} ({100*(i+1)/n_tasks:.0f}%)", flush=True)
    
    elapsed = time.time() - start_time
    print(f"    Completed in {elapsed:.1f}s ({len(all_results)} trajectories)", flush=True)
    
    if not all_results:
        print(f"    WARNING: No results for γ_max={gamma_max}!")
        return None
    
    # Convert to DataFrame with trajectory arrays
    # Each row is one trajectory, arrays stored as lists
    rows = []
    for r in all_results:
        rows.append({
            'disorder_id': r['disorder_id'],
            'temperature': r['temperature'],
            'gamma_acc': r['gamma_acc'].tolist(),
            'energy': r['energy'].tolist(),
            'd': r['d'].tolist(),
        })
    
    df = pd.DataFrame(rows)
    df.to_parquet(output_file, index=False)
    print(f"    Saved: {output_file.name}", flush=True)
    
    return output_file


def main():
    parser = argparse.ArgumentParser(description='Run NK Model Trajectories')
    parser.add_argument('--n', type=int, required=True, help='System size N')
    parser.add_argument('--n-disorders', type=int, default=200, help='Number of disorder realizations')
    parser.add_argument('--gamma-acc', type=float, default=1000, help='Target accumulated strain')
    parser.add_argument('--workers', type=int, default=None, help='Number of parallel workers')
    parser.add_argument('--gamma-max-values', type=str, default=None, 
                        help='Comma-separated γ_max values (default: 0.1-1.0)')
    args = parser.parse_args()
    
    if args.workers is None:
        args.workers = max(1, cpu_count() - 1)
    
    # Parameters
    temperatures = [0.6, 1.0]
    
    if args.gamma_max_values:
        gamma_max_values = [float(x) for x in args.gamma_max_values.split(',')]
    else:
        # Dense sampling around transition
        gamma_max_values = [0.1, 0.2, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.7, 0.8, 0.9, 1.0]
    
    # Output directory
    output_dir = RESULTS_DIR / f"N{args.n:03d}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("NK Model Trajectory Generation")
    print("=" * 60)
    print(f"N = {args.n}, K = 10")
    print(f"γ_acc = {args.gamma_acc}")
    print(f"Disorders = {args.n_disorders}")
    print(f"Temperatures: {temperatures}")
    print(f"γ_max values: {gamma_max_values}")
    print(f"Workers: {args.workers}")
    print(f"Output dir: {output_dir}")
    print(f"Disorder dir: {get_disorder_dir(args.n)}")
    
    total_start = time.time()
    
    # Run for each gamma_max (separately to save memory and allow incremental progress)
    for gamma_max in gamma_max_values:
        run_for_gamma_max(
            n=args.n,
            gamma_max=gamma_max,
            temperatures=temperatures,
            gamma_acc_target=args.gamma_acc,
            n_disorders=args.n_disorders,
            n_workers=args.workers,
            output_dir=output_dir,
        )
    
    total_elapsed = time.time() - total_start
    print(f"\n{'=' * 60}")
    print(f"Total time: {total_elapsed/60:.1f} minutes")
    print(f"Output directory: {output_dir}")
    print("=" * 60)


if __name__ == '__main__':
    main()





