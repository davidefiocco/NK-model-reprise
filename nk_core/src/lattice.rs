use rand::Rng;
use rand_xoshiro::Xoshiro256PlusPlus;

/// NK lattice with N sites, K neighbors per site
#[derive(Clone)]
pub struct NKLattice {
    pub n: usize,
    #[allow(dead_code)]  // Used by Python bindings
    pub k: usize,
    /// Coupling matrix: couplings[i] contains indices of site i and its K neighbors
    /// First element is always i itself, followed by K neighbor indices (sorted)
    pub couplings: Vec<Vec<usize>>,
    /// Random parameters a[conf] in range [0, 1]
    pub a: Vec<f64>,
    /// Random parameters b[conf] in range [-1, 1]  
    pub b: Vec<f64>,
}

impl NKLattice {
    /// Create a new NK lattice with random couplings and parameters
    pub fn new(n: usize, k: usize, rng: &mut Xoshiro256PlusPlus) -> Self {
        let couplings = Self::generate_couplings(n, k, rng);
        let num_configs = 2_usize.pow((k + 1) as u32);
        
        let a: Vec<f64> = (0..num_configs)
            .map(|_| rng.gen::<f64>())
            .collect();
        
        let b: Vec<f64> = (0..num_configs)
            .map(|_| rng.gen::<f64>() * 2.0 - 1.0)
            .collect();
        
        Self { n, k, couplings, a, b }
    }
    
    /// Generate random couplings ensuring each site has exactly K neighbors
    /// Algorithm: Iteratively pick random pairs and add mutual connections
    /// until all sites have K neighbors
    fn generate_couplings(n: usize, k: usize, rng: &mut Xoshiro256PlusPlus) -> Vec<Vec<usize>> {
        const MAX_ATTEMPTS: usize = 100_000_000;
        
        loop {
            let mut couplings: Vec<Vec<usize>> = (0..n).map(|i| vec![i]).collect();
            let mut coupled = vec![vec![false; n]; n];
            
            // Mark diagonal as coupled (site coupled with itself)
            for i in 0..n {
                coupled[i][i] = true;
            }
            
            let mut available: Vec<usize> = (0..n * k).map(|i| i % n).collect();
            let mut success = false;
            
            for _ in 0..MAX_ATTEMPTS {
                // Pick two random indices from available pool
                if available.is_empty() {
                    break;
                }
                
                let idx1 = rng.gen_range(0..available.len());
                let idx2 = rng.gen_range(0..available.len());
                
                if idx1 == idx2 {
                    continue;
                }
                
                let site1 = available[idx1];
                let site2 = available[idx2];
                
                // Check if this pairing is valid
                if site1 != site2 && !coupled[site1][site2] {
                    let n1 = couplings[site1].len() - 1; // -1 because first element is self
                    let n2 = couplings[site2].len() - 1;
                    
                    if n1 < k && n2 < k {
                        // Add mutual coupling
                        couplings[site1].push(site2);
                        couplings[site2].push(site1);
                        coupled[site1][site2] = true;
                        coupled[site2][site1] = true;
                        
                        // Remove from available pool (swap with last element)
                        available.swap_remove(idx1.max(idx2));
                        available.swap_remove(idx1.min(idx2));
                        
                        // Check if we're done
                        if available.is_empty() {
                            success = true;
                            break;
                        }
                    }
                }
            }
            
            if success {
                // Sort each coupling list for consistency with original C code
                for coupling in &mut couplings {
                    coupling.sort_unstable();
                }
                return couplings;
            }
            // If failed, retry with new random attempt
        }
    }
    
    /// Get configuration value for site i given spin configuration
    /// conf_value = sum_{j=0..K} 2^j * spins[couplings[i][j]]
    pub fn get_conf_value(&self, i: usize, spins: &[u8]) -> usize {
        let mut conf_value = 0;
        for (j, &neighbor) in self.couplings[i].iter().enumerate() {
            if spins[neighbor] != 0 {
                conf_value += (1 << j) * spins[neighbor] as usize;
            }
        }
        conf_value
    }
}

/// Spin configuration for NK model
#[derive(Clone, Debug)]
pub struct SpinConfig {
    pub spins: Vec<u8>,
}

impl SpinConfig {
    /// Create initial spin configuration with N/2 up spins (1) and N/2 down spins (0)
    pub fn new(n: usize, rng: &mut Xoshiro256PlusPlus) -> Self {
        let mut spins = vec![0u8; n];
        let mut indices: Vec<usize> = (0..n).collect();
        
        // Fisher-Yates shuffle to randomly select N/2 sites for up spins
        for i in 0..n / 2 {
            let j = rng.gen_range(i..n);
            indices.swap(i, j);
            spins[indices[i]] = 1;
        }
        
        Self { spins }
    }
    
    /// Create configuration from existing spin vector
    pub fn from_vec(spins: Vec<u8>) -> Self {
        Self { spins }
    }
    
    /// Flip a spin at position i
    pub fn flip(&mut self, i: usize) {
        self.spins[i] = 1 - self.spins[i];
    }
    
    /// Compute Hamming distance to another configuration
    pub fn hamming_distance(&self, other: &SpinConfig) -> usize {
        self.spins.iter()
            .zip(other.spins.iter())
            .filter(|(a, b)| a != b)
            .count()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    
    #[test]
    fn test_lattice_generation() {
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(12345);
        let lattice = NKLattice::new(20, 10, &mut rng);
        
        assert_eq!(lattice.n, 20);
        assert_eq!(lattice.k, 10);
        assert_eq!(lattice.couplings.len(), 20);
        
        // Each site should have exactly K+1 elements (self + K neighbors)
        for coupling in &lattice.couplings {
            assert_eq!(coupling.len(), 11);
        }
        
        // Check parameters have correct size
        assert_eq!(lattice.a.len(), 2048); // 2^(K+1) = 2^11
        assert_eq!(lattice.b.len(), 2048);
    }
    
    #[test]
    fn test_spin_config() {
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(42);
        let config = SpinConfig::new(20, &mut rng);
        
        // Should have exactly N/2 up spins
        let up_count: usize = config.spins.iter().map(|&s| s as usize).sum();
        assert_eq!(up_count, 10);
    }
    
    #[test]
    fn test_hamming_distance() {
        let config1 = SpinConfig::from_vec(vec![0, 1, 0, 1, 0]);
        let config2 = SpinConfig::from_vec(vec![0, 0, 0, 1, 1]);
        
        assert_eq!(config1.hamming_distance(&config2), 2);
    }
}

