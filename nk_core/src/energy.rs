use crate::lattice::{NKLattice, SpinConfig};
use std::f64::consts::PI;

/// Compute total energy of the system
/// E = -0.5 * sum_i [1 + sin(2π(a[conf_i] + γ * b[conf_i]))]
pub fn compute_energy(lattice: &NKLattice, config: &SpinConfig, gamma: f64) -> f64 {
    let mut energy = 0.0;
    
    for i in 0..lattice.n {
        let conf = lattice.get_conf_value(i, &config.spins);
        let phase = 2.0 * PI * (lattice.a[conf] + gamma * lattice.b[conf]);
        energy -= 1.0 + phase.sin();
    }
    
    0.5 * energy
}

/// Compute energy change if we flip spins i and j
/// Only need to recompute energy of sites that are neighbors of i or j
/// 
/// A site m is affected by flipping spins i,j iff i or j is in m's coupling list.
/// Due to symmetric coupling, this is equivalent to m being in couplings[i] or couplings[j].
/// This is more efficient than the previous implementation which computed neighbors-of-neighbors.
pub fn compute_energy_change(
    lattice: &NKLattice,
    config: &SpinConfig,
    i: usize,
    j: usize,
    gamma: f64,
) -> f64 {
    // Get all sites affected by flipping i or j
    // A site m is affected iff m ∈ couplings[i] ∪ couplings[j]
    // (because flipping i or j changes m's conf_value iff i or j is in m's neighborhood)
    let mut affected_sites = std::collections::HashSet::new();
    
    // Add all sites in couplings[i] (includes i itself and its K neighbors)
    for &site in &lattice.couplings[i] {
        affected_sites.insert(site);
    }
    
    // Add all sites in couplings[j] (includes j itself and its K neighbors)
    for &site in &lattice.couplings[j] {
        affected_sites.insert(site);
    }
    
    // Compute energy before flip for affected sites
    let mut energy_before = 0.0;
    for &site in &affected_sites {
        let conf = lattice.get_conf_value(site, &config.spins);
        let phase = 2.0 * PI * (lattice.a[conf] + gamma * lattice.b[conf]);
        energy_before -= 1.0 + phase.sin();
    }
    
    // Create temporary config with flipped spins
    let mut temp_spins = config.spins.clone();
    temp_spins[i] = 1 - temp_spins[i];
    temp_spins[j] = 1 - temp_spins[j];
    
    // Compute energy after flip for affected sites
    let mut energy_after = 0.0;
    for &site in &affected_sites {
        let conf = lattice.get_conf_value(site, &temp_spins);
        let phase = 2.0 * PI * (lattice.a[conf] + gamma * lattice.b[conf]);
        energy_after -= 1.0 + phase.sin();
    }
    
    0.5 * (energy_after - energy_before)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lattice::NKLattice;
    use rand::SeedableRng;
    use rand_xoshiro::Xoshiro256PlusPlus;
    
    #[test]
    fn test_energy_computation() {
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(12345);
        let lattice = NKLattice::new(20, 10, &mut rng);
        let config = SpinConfig::new(20, &mut rng);
        
        let energy = compute_energy(&lattice, &config, 0.0);
        
        // Energy should be finite and reasonable
        assert!(energy.is_finite());
        assert!(energy < 0.0); // Should be negative due to -0.5 factor
    }
    
    #[test]
    fn test_energy_change_consistency() {
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(42);
        let lattice = NKLattice::new(20, 10, &mut rng);
        let mut config = SpinConfig::new(20, &mut rng);
        
        let gamma = 0.3;
        let energy_before = compute_energy(&lattice, &config, gamma);
        
        let i = 5;
        let j = 10;
        let delta_e = compute_energy_change(&lattice, &config, i, j, gamma);
        
        config.flip(i);
        config.flip(j);
        let energy_after = compute_energy(&lattice, &config, gamma);
        
        let actual_delta = energy_after - energy_before;
        
        // Should be close (within numerical precision)
        assert!((delta_e - actual_delta).abs() < 1e-10);
    }
}


