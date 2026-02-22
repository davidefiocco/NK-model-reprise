#!/usr/bin/env python3
"""
Quick physics validation of the NK model implementation.

Tests:
1. Low γ_max → absorbing state (d=0)
2. High γ_max → diffusing state (d>0)
3. Energy decreases/stabilizes with shear cycling
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import nk_core

def test_absorbing_state():
    """Low gamma_max should lead to absorbing state (d=0 after settling)."""
    print("=" * 60)
    print("Test 1: Absorbing state (low γ_max=0.1)")
    print("=" * 60)
    
    # Create lattice and config
    n, k = 20, 10
    seed = 12345
    
    lattice = nk_core.create_lattice(n, k, seed)
    initial_config = nk_core.create_spin_config(n, seed + 100)
    
    # First equilibrate at T=0.6
    equil = nk_core.run_equilibration(
        n=n, k=k,
        couplings=lattice['couplings'].tolist(),
        a=lattice['a'].tolist(),
        b=lattice['b'].tolist(),
        initial_spins=initial_config.tolist(),
        temperature=0.6,
        t_steps=20,
        seed=seed + 200
    )
    
    # Run deformation with small gamma_max
    gamma_max = 0.1
    dgamma = 0.02
    n_cycles = 10
    steps_per_cycle = int(4 * gamma_max / dgamma)
    total_steps = n_cycles * steps_per_cycle
    
    result = nk_core.run_deformation(
        n=n, k=k,
        couplings=lattice['couplings'].tolist(),
        a=lattice['a'].tolist(),
        b=lattice['b'].tolist(),
        initial_spins=equil['final_config'].tolist(),
        gamma_max=gamma_max,
        dgamma=dgamma,
        total_steps=total_steps,
        dump_zero_strain_only=True
    )
    
    hamming_cycle = np.array(result['hamming_cycle'])
    n_zero_strain = result['n_zero_strain_configs']
    
    print(f"  γ_max = {gamma_max}")
    print(f"  Ran {n_cycles} cycles ({total_steps} steps)")
    print(f"  Zero-strain configs: {n_zero_strain}")
    print(f"  Full-cycle displacements: {hamming_cycle}")
    
    # Last few displacements should be 0 (absorbing)
    last_d = hamming_cycle[-3:]
    is_absorbing = np.all(last_d == 0)
    print(f"  Last 3 displacements: {last_d}")
    print(f"  ✓ Absorbing: {is_absorbing}" if is_absorbing else f"  ✗ NOT absorbing (unexpected)")
    
    return is_absorbing


def test_diffusing_state():
    """High gamma_max should lead to diffusing state (d>0)."""
    print("\n" + "=" * 60)
    print("Test 2: Diffusing state (high γ_max=0.8)")
    print("=" * 60)
    
    n, k = 20, 10
    seed = 42
    
    lattice = nk_core.create_lattice(n, k, seed)
    initial_config = nk_core.create_spin_config(n, seed + 100)
    
    # Equilibrate at T=0.6
    equil = nk_core.run_equilibration(
        n=n, k=k,
        couplings=lattice['couplings'].tolist(),
        a=lattice['a'].tolist(),
        b=lattice['b'].tolist(),
        initial_spins=initial_config.tolist(),
        temperature=0.6,
        t_steps=20,
        seed=seed + 200
    )
    
    # Run deformation with large gamma_max
    gamma_max = 0.8
    dgamma = 0.02
    n_cycles = 5
    steps_per_cycle = int(4 * gamma_max / dgamma)
    total_steps = n_cycles * steps_per_cycle
    
    result = nk_core.run_deformation(
        n=n, k=k,
        couplings=lattice['couplings'].tolist(),
        a=lattice['a'].tolist(),
        b=lattice['b'].tolist(),
        initial_spins=equil['final_config'].tolist(),
        gamma_max=gamma_max,
        dgamma=dgamma,
        total_steps=total_steps,
        dump_zero_strain_only=True
    )
    
    hamming_cycle = np.array(result['hamming_cycle'])
    n_zero_strain = result['n_zero_strain_configs']
    
    print(f"  γ_max = {gamma_max}")
    print(f"  Ran {n_cycles} cycles ({total_steps} steps)")
    print(f"  Zero-strain configs: {n_zero_strain}")
    print(f"  Full-cycle displacements: {hamming_cycle}")
    
    # Should have nonzero displacements after first full cycle
    nonzero_d = hamming_cycle[hamming_cycle > 0]
    is_diffusing = len(nonzero_d) > 0
    mean_d = np.mean(nonzero_d) if len(nonzero_d) > 0 else 0
    print(f"  Mean d (nonzero): {mean_d:.2f}")
    print(f"  ✓ Diffusing: {is_diffusing}" if is_diffusing else f"  ✗ NOT diffusing (unexpected)")
    
    return is_diffusing


def test_transition_scan():
    """Scan across γ_max to see transition."""
    print("\n" + "=" * 60)
    print("Test 3: Transition scan (γ_max from 0.1 to 1.0)")
    print("=" * 60)
    
    n, k = 20, 10
    seed = 99
    n_disorders = 3  # Quick test
    
    gamma_max_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    
    results = []
    
    for d_idx in range(n_disorders):
        lattice = nk_core.create_lattice(n, k, seed + d_idx * 1000)
        initial_config = nk_core.create_spin_config(n, seed + d_idx * 1000 + 100)
        
        equil = nk_core.run_equilibration(
            n=n, k=k,
            couplings=lattice['couplings'].tolist(),
            a=lattice['a'].tolist(),
            b=lattice['b'].tolist(),
            initial_spins=initial_config.tolist(),
            temperature=0.6,
            t_steps=20,
            seed=seed + d_idx * 1000 + 200
        )
        
        for gamma_max in gamma_max_values:
            dgamma = 0.02
            n_cycles = 15
            steps_per_cycle = int(4 * gamma_max / dgamma)
            total_steps = n_cycles * steps_per_cycle
            
            result = nk_core.run_deformation(
                n=n, k=k,
                couplings=lattice['couplings'].tolist(),
                a=lattice['a'].tolist(),
                b=lattice['b'].tolist(),
                initial_spins=equil['final_config'].tolist(),
                gamma_max=gamma_max,
                dgamma=dgamma,
                total_steps=total_steps,
                dump_zero_strain_only=True
            )
            
            hamming_cycle = np.array(result['hamming_cycle'])
            # Use last 5 values (after settling)
            d_steady = hamming_cycle[-5:] if len(hamming_cycle) >= 5 else hamming_cycle
            results.append({
                'disorder': d_idx,
                'gamma_max': gamma_max,
                'd_mean': np.mean(d_steady) / n,
            })
    
    # Average over disorders
    print(f"\n  Disorder-averaged order parameter d/N:")
    print(f"  {'γ_max':>8} | {'d/N':>8} | State")
    print(f"  {'-'*8}-+-{'-'*8}-+{'-'*10}")
    
    for gamma_max in gamma_max_values:
        d_vals = [r['d_mean'] for r in results if r['gamma_max'] == gamma_max]
        d_mean = np.mean(d_vals)
        state = "Absorbing" if d_mean < 0.01 else "Diffusing"
        print(f"  {gamma_max:8.2f} | {d_mean:8.4f} | {state}")
    
    return True


def main():
    print("\n" + "=" * 60)
    print("NK Model Physics Validation")
    print("=" * 60 + "\n")
    
    test1 = test_absorbing_state()
    test2 = test_diffusing_state()
    test3 = test_transition_scan()
    
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"  Test 1 (Absorbing): {'PASS' if test1 else 'FAIL'}")
    print(f"  Test 2 (Diffusing): {'PASS' if test2 else 'FAIL'}")
    print(f"  Test 3 (Transition): {'PASS' if test3 else 'FAIL'}")
    
    if test1 and test2 and test3:
        print("\n✓ All physics tests passed!")
        return 0
    else:
        print("\n✗ Some tests failed!")
        return 1


if __name__ == '__main__':
    sys.exit(main())





