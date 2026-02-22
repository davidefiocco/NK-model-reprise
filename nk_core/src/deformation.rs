use crate::dynamics::minimize_to_local_minimum;
use crate::energy::compute_energy;
use crate::lattice::{NKLattice, SpinConfig};

/// Result of a single deformation step
#[derive(Clone, Debug)]
pub struct DeformationStep {
    pub gamma: f64,
    pub gamma_acc: f64,
    pub energy: f64,
    pub config: SpinConfig,
}

/// Result of oscillatory deformation including zero-strain configs for order parameter
#[derive(Clone, Debug)]
pub struct DeformationResult {
    /// Full trajectory (or zero-strain only if dump_zero_strain_only=true)
    pub trajectory: Vec<DeformationStep>,
    /// Configs at gamma=0 crossings (for correct order parameter computation)
    /// Stored AFTER minimization at gamma=0, matching C code behavior
    pub zero_strain_configs: Vec<SpinConfig>,
}

/// Check if we're at a zero-strain point (γ=0 crossing)
/// Uses the same logic as C code: γ_acc mod (2*γ_max) ≈ 0
#[inline]
fn is_at_zero_strain(gamma_acc: f64, gamma_max: f64, dgamma: f64) -> bool {
    let cycle_half = 2.0 * gamma_max;
    let remainder = gamma_acc % cycle_half;
    // Near 0 or near cycle_half (which is also 0 mod cycle_half)
    remainder < dgamma / 2.0 || remainder > cycle_half - dgamma / 2.0
}

/// Perform oscillatory athermal quasistatic (AQS) deformation
/// 
/// Protocol: 0 -> +gamma_max -> 0 -> -gamma_max -> 0 (repeating)
/// This exactly matches the C code in dnk.c
/// 
/// Zero-strain detection: γ_acc mod (2*γ_max) ≈ 0
/// This triggers at γ=0 after every HALF cycle (0→+γ→0 and 0→-γ→0)
/// 
/// Order parameter: Compare configs at γ=0 that are 2 half-cycles (= 1 full cycle) apart
pub fn oscillatory_deformation(
    lattice: &NKLattice,
    initial_config: &SpinConfig,
    gamma_max: f64,
    dgamma: f64,
    total_steps: usize,
    dump_zero_strain_only: bool,
) -> DeformationResult {
    let mut config = initial_config.clone();
    let mut gamma: f64 = 0.0;
    let mut gamma_acc: f64 = 0.0;
    let mut direction: i32 = 1; // 1 for increasing, -1 for decreasing
    
    let mut trajectory = Vec::new();
    let mut zero_strain_configs: Vec<SpinConfig> = Vec::new();
    
    // Initial state: minimize at gamma=0, then store
    minimize_to_local_minimum(lattice, &mut config, 0.0);
    zero_strain_configs.push(config.clone());
    
    let energy = compute_energy(lattice, &config, gamma);
    trajectory.push(DeformationStep {
        gamma,
        gamma_acc,
        energy,
        config: config.clone(),
    });
    
    for _step in 0..total_steps {
        // Update direction at turning points (BEFORE incrementing gamma)
        // This matches C code: check at gamma, then change direction, then increment
        if gamma >= gamma_max - dgamma / 2.0 {
            direction = -1;
        } else if gamma <= -gamma_max + dgamma / 2.0 {
            direction = 1;
        }
        
        // Increment strain
        gamma += direction as f64 * dgamma;
        gamma_acc += dgamma;
        
        // Minimize to find new inherent structure at new gamma
        minimize_to_local_minimum(lattice, &mut config, gamma);
        
        // Check if we're at a zero-strain crossing
        let at_zero_strain = is_at_zero_strain(gamma_acc, gamma_max, dgamma);
        
        // Store zero-strain config for order parameter (AFTER minimization)
        if at_zero_strain {
            zero_strain_configs.push(config.clone());
        }
        
        // Store configuration in trajectory if appropriate
        if !dump_zero_strain_only || at_zero_strain {
            let energy = compute_energy(lattice, &config, gamma);
            trajectory.push(DeformationStep {
                gamma,
                gamma_acc,
                energy,
                config: config.clone(),
            });
        }
    }
    
    DeformationResult {
        trajectory,
        zero_strain_configs,
    }
}


/// Compute observables from a deformation trajectory
pub struct DeformationObservables {
    pub gamma_values: Vec<f64>,
    pub energy_values: Vec<f64>,
    pub msd_from_initial: Vec<f64>,
    pub hamming_from_initial: Vec<f64>,
    /// Full-cycle displacement: distance between configs at gamma=0, one complete cycle apart
    /// This is the correct order parameter for the absorbing transition!
    /// Computed from zero_strain_configs comparing config[i] with config[i-2]
    pub full_cycle_displacement: Vec<f64>,
}

impl DeformationObservables {
    /// Create observables from a DeformationResult (preferred method - uses zero_strain_configs)
    pub fn from_result(result: &DeformationResult) -> Self {
        let trajectory = &result.trajectory;
        let zero_configs = &result.zero_strain_configs;
        
        if trajectory.is_empty() {
            return Self {
                gamma_values: vec![],
                energy_values: vec![],
                msd_from_initial: vec![],
                hamming_from_initial: vec![],
                full_cycle_displacement: vec![],
            };
        }
        
        let initial_config = &trajectory[0].config;
        let n = initial_config.spins.len();
        
        let gamma_values: Vec<f64> = trajectory.iter().map(|s| s.gamma).collect();
        let energy_values: Vec<f64> = trajectory.iter().map(|s| s.energy).collect();
        
        // Distance from initial configuration (for reference/diagnostics)
        let hamming_from_initial: Vec<f64> = trajectory
            .iter()
            .map(|s| initial_config.hamming_distance(&s.config) as f64)
            .collect();
        
        // Full-cycle displacement: compare zero-strain configs 2 steps apart
        // (one full cycle = 0 → +γ → 0 → -γ → 0)
        // This matches the C code's comparison of minspinsback0 vs minspinsback2
        let mut full_cycle_displacement: Vec<f64> = Vec::with_capacity(zero_configs.len());
        for i in 0..zero_configs.len() {
            if i < 2 {
                // Not enough history for full cycle comparison
                full_cycle_displacement.push(0.0);
            } else {
                // Compare with config from 2 zero-crossings ago (= one full cycle)
                let dist = zero_configs[i - 2].hamming_distance(&zero_configs[i]) as f64;
                full_cycle_displacement.push(dist);
            }
        }
        
        // MSD is normalized Hamming distance squared
        let msd_from_initial: Vec<f64> = hamming_from_initial
            .iter()
            .map(|&h| (h / n as f64).powi(2))
            .collect();
        
        Self {
            gamma_values,
            energy_values,
            msd_from_initial,
            hamming_from_initial,
            full_cycle_displacement,
        }
    }
    
    /// Legacy method for backward compatibility - filters trajectory for zero-strain configs
    /// DEPRECATED: Use from_result() instead for correct order parameter computation
    #[allow(dead_code)]
    pub fn from_trajectory(trajectory: &[DeformationStep]) -> Self {
        if trajectory.is_empty() {
            return Self {
                gamma_values: vec![],
                energy_values: vec![],
                msd_from_initial: vec![],
                hamming_from_initial: vec![],
                full_cycle_displacement: vec![],
            };
        }
        
        let initial_config = &trajectory[0].config;
        let n = initial_config.spins.len();
        
        let gamma_values: Vec<f64> = trajectory.iter().map(|s| s.gamma).collect();
        let energy_values: Vec<f64> = trajectory.iter().map(|s| s.energy).collect();
        
        // Distance from initial configuration (for reference/diagnostics)
        let hamming_from_initial: Vec<f64> = trajectory
            .iter()
            .map(|s| initial_config.hamming_distance(&s.config) as f64)
            .collect();
        
        // Filter to zero-strain configs only for correct order parameter
        let zero_strain_steps: Vec<&DeformationStep> = trajectory
            .iter()
            .filter(|s| s.gamma.abs() < 1e-6)
            .collect();
        
        // Full-cycle displacement from zero-strain configs
        let mut full_cycle_displacement: Vec<f64> = Vec::with_capacity(zero_strain_steps.len());
        for i in 0..zero_strain_steps.len() {
            if i < 2 {
                full_cycle_displacement.push(0.0);
            } else {
                let dist = zero_strain_steps[i - 2].config.hamming_distance(&zero_strain_steps[i].config) as f64;
                full_cycle_displacement.push(dist);
            }
        }
        
        // MSD is normalized Hamming distance squared
        let msd_from_initial: Vec<f64> = hamming_from_initial
            .iter()
            .map(|&h| (h / n as f64).powi(2))
            .collect();
        
        Self {
            gamma_values,
            energy_values,
            msd_from_initial,
            hamming_from_initial,
            full_cycle_displacement,
        }
    }
}

/// Compute hysteresis loop area from a deformation trajectory
/// Area = integral of stress * dgamma over one complete cycle
pub fn compute_hysteresis_area(trajectory: &[DeformationStep], dgamma: f64) -> f64 {
    // Find one complete cycle at steady state (use last complete cycle)
    // For simplicity, use stress approximation: stress ≈ -dE/dgamma
    
    if trajectory.len() < 2 {
        return 0.0;
    }
    
    // Find cycles where gamma returns to ~0
    let mut zero_crossings = vec![0];
    for (i, step) in trajectory.iter().enumerate().skip(1) {
        if step.gamma.abs() < dgamma / 2.0 {
            zero_crossings.push(i);
        }
    }
    
    if zero_crossings.len() < 2 {
        return 0.0;
    }
    
    // Use last complete cycle
    let start_idx = zero_crossings[zero_crossings.len() - 2];
    let end_idx = zero_crossings[zero_crossings.len() - 1];
    
    if end_idx <= start_idx + 1 {
        return 0.0;
    }
    
    // Compute area using trapezoidal rule
    // Area = sum of |stress| * dgamma
    let mut area = 0.0;
    for i in start_idx..(end_idx - 1) {
        let stress = (trajectory[i + 1].energy - trajectory[i].energy) / dgamma;
        area += stress.abs() * dgamma;
    }
    
    area
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lattice::NKLattice;
    use rand::SeedableRng;
    use rand_xoshiro::Xoshiro256PlusPlus;
    
    #[test]
    fn test_is_at_zero_strain() {
        let gamma_max = 0.3;
        let dgamma = 0.01;
        let half_cycle = 2.0 * gamma_max;  // 0.6
        
        // At start (γ_acc = 0)
        assert!(is_at_zero_strain(0.0, gamma_max, dgamma));
        
        // After one half-cycle (γ_acc = 0.6)
        assert!(is_at_zero_strain(half_cycle, gamma_max, dgamma));
        assert!(is_at_zero_strain(half_cycle - 0.001, gamma_max, dgamma));
        
        // After one full cycle (γ_acc = 1.2)
        assert!(is_at_zero_strain(2.0 * half_cycle, gamma_max, dgamma));
        
        // NOT at zero strain (γ_acc = 0.3, which is at γ = gamma_max)
        assert!(!is_at_zero_strain(0.3, gamma_max, dgamma));
        assert!(!is_at_zero_strain(0.15, gamma_max, dgamma));
    }
    
    #[test]
    fn test_oscillatory_deformation() {
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(12345);
        let lattice = NKLattice::new(20, 10, &mut rng);
        let initial_config = SpinConfig::new(20, &mut rng);
        
        let gamma_max = 0.3;
        let dgamma = 0.01;
        // One complete cycle: 0->0.3->0->-0.3->0 = 4*30 = 120 steps
        let steps = (4.0 * gamma_max / dgamma) as usize;
        
        let result = oscillatory_deformation(
            &lattice,
            &initial_config,
            gamma_max,
            dgamma,
            steps,
            false,
        );
        
        assert!(!result.trajectory.is_empty());
        
        // First gamma should be 0
        assert!(result.trajectory[0].gamma.abs() < 1e-10);
        
        // Should have zero-strain configs stored:
        // Initial (γ_acc=0) + after 1st half (γ_acc=0.6) + after full cycle (γ_acc=1.2) = 3
        assert_eq!(result.zero_strain_configs.len(), 3);
    }
    
    #[test]
    fn test_zero_strain_config_count() {
        // Verify we get the right number of zero-strain configs
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(42);
        let lattice = NKLattice::new(20, 10, &mut rng);
        let initial_config = SpinConfig::new(20, &mut rng);
        
        let gamma_max = 0.2;
        let dgamma = 0.02;
        // Two full cycles = 4 half-cycles
        // Steps per half-cycle = 2 * gamma_max / dgamma = 20
        // Total steps = 80
        let steps = (8.0 * gamma_max / dgamma) as usize;
        
        let result = oscillatory_deformation(
            &lattice,
            &initial_config,
            gamma_max,
            dgamma,
            steps,
            true,  // dump_zero_strain_only
        );
        
        // Should have: initial + 4 half-cycles = 5 zero-strain configs
        assert_eq!(result.zero_strain_configs.len(), 5);
        
        // Trajectory should also have 5 entries (only zero-strain points)
        assert_eq!(result.trajectory.len(), 5);
    }
    
    #[test]
    fn test_observables_computation() {
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(42);
        let lattice = NKLattice::new(20, 10, &mut rng);
        let initial_config = SpinConfig::new(20, &mut rng);
        
        let result = oscillatory_deformation(
            &lattice,
            &initial_config,
            0.2,
            0.02,
            80,  // 2 full cycles
            false,
        );
        
        let obs = DeformationObservables::from_result(&result);
        
        assert_eq!(obs.gamma_values.len(), result.trajectory.len());
        assert_eq!(obs.energy_values.len(), result.trajectory.len());
        assert_eq!(obs.msd_from_initial.len(), result.trajectory.len());
        
        // MSD at initial point should be 0
        assert!(obs.msd_from_initial[0].abs() < 1e-10);
        
        // full_cycle_displacement should match zero_strain_configs length
        assert_eq!(obs.full_cycle_displacement.len(), result.zero_strain_configs.len());
        
        // First two entries should be 0 (no full cycle history yet)
        assert!(obs.full_cycle_displacement[0].abs() < 1e-10);
        assert!(obs.full_cycle_displacement[1].abs() < 1e-10);
        
        // Third entry (index 2) compares config after full cycle with initial
        // This should be defined (may or may not be zero depending on absorbing/diffusing)
        assert!(obs.full_cycle_displacement[2].is_finite());
    }
    
    #[test]
    fn test_hysteresis_area() {
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(123);
        let lattice = NKLattice::new(20, 10, &mut rng);
        let initial_config = SpinConfig::new(20, &mut rng);
        
        let dgamma = 0.01;
        let result = oscillatory_deformation(
            &lattice,
            &initial_config,
            0.3,
            dgamma,
            240,  // 2 full cycles
            false,
        );
        
        let area = compute_hysteresis_area(&result.trajectory, dgamma);
        
        // Area should be non-negative and finite
        assert!(area >= 0.0);
        assert!(area.is_finite());
    }
    
    #[test]
    fn test_full_cycle_displacement_absorbing() {
        // Test that an absorbing state has zero full-cycle displacement after settling
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(999);
        let lattice = NKLattice::new(20, 10, &mut rng);
        let initial_config = SpinConfig::new(20, &mut rng);
        
        // Very small gamma_max should typically lead to absorbing state
        let gamma_max = 0.05;
        let dgamma = 0.01;
        // Run many cycles to ensure settling
        let steps = (20.0 * gamma_max / dgamma) as usize;  // 5 full cycles
        
        let result = oscillatory_deformation(
            &lattice,
            &initial_config,
            gamma_max,
            dgamma,
            steps,
            true,
        );
        
        let obs = DeformationObservables::from_result(&result);
        
        // For very small gamma, system should become absorbing
        // Last few displacements should be 0 (exact same config each cycle)
        if obs.full_cycle_displacement.len() >= 6 {
            let last_displacement = *obs.full_cycle_displacement.last().unwrap();
            // With very small gamma, we expect absorbing state
            assert_eq!(last_displacement, 0.0, 
                "Expected absorbing state (d=0) for gamma_max=0.05, got d={}", last_displacement);
        }
    }
    
    #[test]
    fn test_full_cycle_displacement_diffusing() {
        // Test that large gamma_max leads to diffusing state (d > 0)
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(42);
        let lattice = NKLattice::new(20, 10, &mut rng);
        let initial_config = SpinConfig::new(20, &mut rng);
        
        // Large gamma_max should lead to diffusing state
        let gamma_max = 0.8;
        let dgamma = 0.02;
        // Run several cycles
        let steps = (8.0 * gamma_max / dgamma) as usize;  // 2 full cycles
        
        let result = oscillatory_deformation(
            &lattice,
            &initial_config,
            gamma_max,
            dgamma,
            steps,
            true,
        );
        
        let obs = DeformationObservables::from_result(&result);
        
        // For large gamma, system should be diffusing (d > 0 after first full cycle)
        // Check the last full-cycle displacement
        if obs.full_cycle_displacement.len() >= 3 {
            // At least one of the full-cycle displacements should be > 0
            let has_nonzero = obs.full_cycle_displacement.iter()
                .skip(2)  // Skip first two (no history)
                .any(|&d| d > 0.0);
            assert!(has_nonzero, 
                "Expected diffusing state (d>0) for gamma_max=0.8, got all zeros");
        }
    }
}

