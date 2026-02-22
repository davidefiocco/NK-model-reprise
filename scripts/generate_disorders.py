#!/usr/bin/env python3
"""
Generate and save disorders (lattices + equilibrated configs) organized by system size.

Output structure:
  results/disorders/
    N020/
      disorder_000.npz
      equil_D000_T0.6.npz
      equil_D000_T1.0.npz
      ...
    N040/
      disorder_000.npz
      ...

Usage:
    python scripts/generate_disorders.py --n 20 --n-disorders 200
    python scripts/generate_disorders.py --n 40 --n-disorders 200
"""

import argparse
import numpy as np
from pathlib import Path
import sys
import multiprocessing as mp

sys.path.insert(0, str(Path(__file__).parent.parent))
import nk_core

# Base output directory
DISORDER_BASE_DIR = Path(__file__).parent.parent / 'results' / 'disorders'

# Default parameters
DEFAULT_K = 10
DEFAULT_TEMPERATURES = [0.6, 1.0]
DEFAULT_T_EQUIL = 20


def get_disorder_dir(n: int) -> Path:
    """Get the disorder directory for a given system size."""
    return DISORDER_BASE_DIR / f'N{n:03d}'


def generate_single_disorder(
    disorder_id: int,
    n: int,
    k: int,
    temperatures: list,
    t_equil: int,
    force: bool,
) -> str:
    """Generate lattice and equilibrated configs for a single disorder."""
    
    disorder_dir = get_disorder_dir(n)
    disorder_dir.mkdir(parents=True, exist_ok=True)
    
    seed = disorder_id * 1000
    
    # Check if disorder file already exists
    disorder_file = disorder_dir / f'disorder_{disorder_id:04d}.npz'
    
    if disorder_file.exists() and not force:
        # Load existing lattice
        data = np.load(disorder_file)
        couplings = data['couplings']
        a = data['a']
        b = data['b']
    else:
        # Generate new lattice
        lattice = nk_core.create_lattice(n, k, seed)
        couplings = lattice['couplings']
        a = lattice['a']
        b = lattice['b']
        
        # Save lattice
        np.savez(
            disorder_file,
            n=n, k=k, seed=seed,
            couplings=couplings,
            a=a, b=b
        )
    
    # Generate equilibrated configs for each temperature
    for T in temperatures:
        equil_file = disorder_dir / f'equil_D{disorder_id:04d}_T{T}.npz'
        
        if equil_file.exists() and not force:
            continue
        
        # Create initial config
        initial_config = nk_core.create_spin_config(n, seed + 500)
        
        # Equilibrate
        equil_result = nk_core.run_equilibration(
            n=n,
            k=k,
            couplings=couplings.tolist(),
            a=a.tolist(),
            b=b.tolist(),
            initial_spins=initial_config.tolist(),
            temperature=T,
            t_steps=t_equil,
            seed=seed + 1000 + int(T * 100)
        )
        
        # Save equilibrated config
        np.savez(
            equil_file,
            disorder_id=disorder_id,
            temperature=T,
            config=equil_result['final_config'],
            final_energy=equil_result['energies'][-1],
        )
    
    return f"disorder {disorder_id}: done"


def _worker(args):
    """Worker for parallel generation."""
    try:
        return generate_single_disorder(*args)
    except Exception as e:
        return f"disorder {args[0]}: ERROR - {e}"


def generate_all_disorders(
    n: int,
    k: int = DEFAULT_K,
    n_disorders: int = 200,
    temperatures: list = None,
    t_equil: int = DEFAULT_T_EQUIL,
    n_workers: int = None,
    force: bool = False,
):
    """Generate all disorders for a given system size."""
    
    if temperatures is None:
        temperatures = DEFAULT_TEMPERATURES
    
    disorder_dir = get_disorder_dir(n)
    disorder_dir.mkdir(parents=True, exist_ok=True)
    
    # Build task list
    tasks = [
        (d_id, n, k, temperatures, t_equil, force)
        for d_id in range(n_disorders)
    ]
    
    if n_workers is None:
        n_workers = min(8, mp.cpu_count())
    
    print(f"Generating {n_disorders} disorders for N={n} on {n_workers} cores...")
    print(f"  K={k}, T={temperatures}")
    print(f"  Output: {disorder_dir}")
    print()
    
    # Run in parallel
    completed = 0
    with mp.Pool(n_workers) as pool:
        for msg in pool.imap_unordered(_worker, tasks):
            completed += 1
            if completed % 20 == 0 or completed == n_disorders:
                print(f"  [{completed}/{n_disorders}] {msg}")
    
    print(f"\n✓ Generated {n_disorders} disorders for N={n}")


def load_disorder(n: int, disorder_id: int, temperature: float):
    """Load a disorder and its equilibrated config for a given system size."""
    
    disorder_dir = get_disorder_dir(n)
    disorder_file = disorder_dir / f'disorder_{disorder_id:04d}.npz'
    equil_file = disorder_dir / f'equil_D{disorder_id:04d}_T{temperature}.npz'
    
    if not disorder_file.exists():
        raise FileNotFoundError(f"Disorder {disorder_id} for N={n} not found: {disorder_file}")
    if not equil_file.exists():
        raise FileNotFoundError(f"Equilibrated config not found: {equil_file}")
    
    disorder_data = np.load(disorder_file)
    equil_data = np.load(equil_file)
    
    return {
        'n': int(disorder_data['n']),
        'k': int(disorder_data['k']),
        'seed': int(disorder_data['seed']),
        'couplings': disorder_data['couplings'],
        'a': disorder_data['a'],
        'b': disorder_data['b'],
        'config': equil_data['config'],
        'temperature': float(equil_data['temperature']),
    }


def main():
    parser = argparse.ArgumentParser(description='Generate disorders organized by system size')
    parser.add_argument('--n', type=int, required=True, help='System size N')
    parser.add_argument('--k', type=int, default=DEFAULT_K, help=f'Neighbors per site (default: {DEFAULT_K})')
    parser.add_argument('--n-disorders', type=int, default=200, help='Number of disorders (default: 200)')
    parser.add_argument('--temperatures', type=float, nargs='+', default=DEFAULT_TEMPERATURES,
                        help=f'Equilibration temperatures (default: {DEFAULT_TEMPERATURES})')
    parser.add_argument('--t-equil', type=int, default=DEFAULT_T_EQUIL,
                        help=f'MC equilibration steps (default: {DEFAULT_T_EQUIL})')
    parser.add_argument('--workers', type=int, default=None, help='Number of parallel workers')
    parser.add_argument('--force', action='store_true', help='Force regenerate all')
    args = parser.parse_args()
    
    print("="*60)
    print(f"Generate Disorders for N={args.n}")
    print("="*60)
    
    generate_all_disorders(
        n=args.n,
        k=args.k,
        n_disorders=args.n_disorders,
        temperatures=args.temperatures,
        t_equil=args.t_equil,
        n_workers=args.workers,
        force=args.force,
    )


if __name__ == '__main__':
    main()
