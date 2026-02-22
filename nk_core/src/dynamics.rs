use crate::energy::{compute_energy, compute_energy_change};
use crate::lattice::{NKLattice, SpinConfig};
use rand::Rng;
use rand_xoshiro::Xoshiro256PlusPlus;

/// Perform one Monte Carlo sweep at temperature T
/// Makes N attempted spin pair flips with Metropolis acceptance
pub fn mc_sweep(
    lattice: &NKLattice,
    config: &mut SpinConfig,
    temperature: f64,
    gamma: f64,
    rng: &mut Xoshiro256PlusPlus,
) {
    let n = lattice.n;
    
    for _ in 0..n {
        // Pick two different sites with opposite spins
        let (i, j) = loop {
            let i = rng.gen_range(0..n);
            let j = rng.gen_range(0..n);
            if i != j && config.spins[i] != config.spins[j] {
                break (i, j);
            }
        };
        
        // Compute energy change
        let delta_e = compute_energy_change(lattice, config, i, j, gamma);
        
        // Metropolis acceptance
        let accept = if delta_e <= 0.0 {
            true
        } else {
            let p = (-delta_e / temperature).exp();
            rng.gen::<f64>() < p
        };
        
        if accept {
            config.flip(i);
            config.flip(j);
        }
    }
}

/// Perform steepest descent energy minimization
/// Returns true if a descent step was made, false if at local minimum
pub fn minimize_step(
    lattice: &NKLattice,
    config: &mut SpinConfig,
    gamma: f64,
) -> bool {
    let n = lattice.n;
    let mut best_i = n;
    let mut best_j = n;
    let mut best_delta_e = 0.0;
    
    // Find the pair flip that gives the largest energy decrease
    for i in 0..n {
        for j in (i + 1)..n {
            if config.spins[i] != config.spins[j] {
                let delta_e = compute_energy_change(lattice, config, i, j, gamma);
                
                if delta_e < best_delta_e {
                    best_delta_e = delta_e;
                    best_i = i;
                    best_j = j;
                }
            }
        }
    }
    
    // If we found a descent direction, take it
    if best_i < n && best_j < n {
        config.flip(best_i);
        config.flip(best_j);
        true
    } else {
        false
    }
}

/// Minimize energy until local minimum is reached
/// Returns number of steps taken
pub fn minimize_to_local_minimum(
    lattice: &NKLattice,
    config: &mut SpinConfig,
    gamma: f64,
) -> usize {
    let mut steps = 0;
    while minimize_step(lattice, config, gamma) {
        steps += 1;
    }
    steps
}

/// Run equilibration at temperature T for t_steps MC sweeps
/// Returns vector of (energy, config) pairs at each step
pub fn equilibrate(
    lattice: &NKLattice,
    initial_config: &SpinConfig,
    temperature: f64,
    t_steps: usize,
    rng: &mut Xoshiro256PlusPlus,
) -> Vec<(f64, SpinConfig)> {
    let mut config = initial_config.clone();
    let mut trajectory = Vec::with_capacity(t_steps);
    
    for _ in 0..t_steps {
        // MC sweep
        mc_sweep(lattice, &mut config, temperature, 0.0, rng);
        
        // Minimize to get inherent structure
        let mut min_config = config.clone();
        minimize_to_local_minimum(lattice, &mut min_config, 0.0);
        
        let energy = compute_energy(lattice, &min_config, 0.0);
        trajectory.push((energy, min_config));
    }
    
    trajectory
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lattice::NKLattice;
    use rand::SeedableRng;
    
    #[test]
    fn test_mc_sweep() {
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(12345);
        let lattice = NKLattice::new(20, 10, &mut rng);
        let mut config = SpinConfig::new(20, &mut rng);
        
        let energy_before = compute_energy(&lattice, &config, 0.0);
        
        // At high temperature, should accept many moves
        mc_sweep(&lattice, &mut config, 10.0, 0.0, &mut rng);
        
        let energy_after = compute_energy(&lattice, &config, 0.0);
        
        // Energy should change (very unlikely to stay exactly same)
        assert_ne!(energy_before, energy_after);
    }
    
    #[test]
    fn test_minimization() {
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(42);
        let lattice = NKLattice::new(20, 10, &mut rng);
        let mut config = SpinConfig::new(20, &mut rng);
        
        let energy_before = compute_energy(&lattice, &config, 0.0);
        
        let steps = minimize_to_local_minimum(&lattice, &mut config, 0.0);
        
        let energy_after = compute_energy(&lattice, &config, 0.0);
        
        // Energy should decrease or stay same
        assert!(energy_after <= energy_before);
        
        // Should have taken at least one step (unlikely to start at minimum)
        assert!(steps > 0);
        
        // Verify we're at a local minimum: no single pair flip should decrease energy
        let n = lattice.n;
        for i in 0..n {
            for j in (i + 1)..n {
                if config.spins[i] != config.spins[j] {
                    let delta_e = compute_energy_change(&lattice, &config, i, j, 0.0);
                    assert!(delta_e >= -1e-10); // Allow small numerical error
                }
            }
        }
    }
    
    #[test]
    fn test_equilibration() {
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(123);
        let lattice = NKLattice::new(20, 10, &mut rng);
        let initial_config = SpinConfig::new(20, &mut rng);
        
        let trajectory = equilibrate(&lattice, &initial_config, 0.6, 10, &mut rng);
        
        assert_eq!(trajectory.len(), 10);
        
        // All energies should be finite
        for (energy, _) in &trajectory {
            assert!(energy.is_finite());
        }
    }
}


