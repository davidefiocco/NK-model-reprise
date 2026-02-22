mod lattice;
mod energy;
mod dynamics;
mod deformation;

use pyo3::prelude::*;
use pyo3::types::PyDict;
use numpy::{PyArray1, PyArray2};
use rand::SeedableRng;
use rand_xoshiro::Xoshiro256PlusPlus;

use lattice::{NKLattice, SpinConfig};
use dynamics::{equilibrate, minimize_to_local_minimum};
use deformation::{oscillatory_deformation, DeformationObservables, compute_hysteresis_area};
use energy::compute_energy;

/// Generate a new NK lattice with random couplings
#[pyfunction]
fn create_lattice(n: usize, k: usize, seed: u64) -> PyResult<PyObject> {
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);
    let lattice = NKLattice::new(n, k, &mut rng);
    
    Python::with_gil(|py| {
        let dict = PyDict::new_bound(py);
        dict.set_item("n", n)?;
        dict.set_item("k", k)?;
        dict.set_item("seed", seed)?;
        
        // Convert couplings to 2D array
        let couplings_array = PyArray2::from_vec2_bound(
            py,
            &lattice.couplings.iter()
                .map(|row| row.iter().map(|&x| x as i32).collect::<Vec<_>>())
                .collect::<Vec<_>>()
        )?;
        dict.set_item("couplings", couplings_array)?;
        
        // Convert a and b parameters
        let a_array = PyArray1::from_vec_bound(py, lattice.a.clone());
        let b_array = PyArray1::from_vec_bound(py, lattice.b.clone());
        dict.set_item("a", a_array)?;
        dict.set_item("b", b_array)?;
        
        Ok(dict.into())
    })
}

/// Create initial spin configuration
#[pyfunction]
fn create_spin_config(n: usize, seed: u64) -> PyResult<Py<PyArray1<u8>>> {
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);
    let config = SpinConfig::new(n, &mut rng);
    
    Python::with_gil(|py| {
        Ok(PyArray1::from_vec_bound(py, config.spins).into())
    })
}

/// Run equilibration at temperature T
#[pyfunction]
fn run_equilibration(
    n: usize,
    k: usize,
    couplings: Vec<Vec<usize>>,
    a: Vec<f64>,
    b: Vec<f64>,
    initial_spins: Vec<u8>,
    temperature: f64,
    t_steps: usize,
    seed: u64,
) -> PyResult<PyObject> {
    let lattice = NKLattice { n, k, couplings, a, b };
    let initial_config = SpinConfig::from_vec(initial_spins);
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);
    
    let trajectory = equilibrate(&lattice, &initial_config, temperature, t_steps, &mut rng);
    
    Python::with_gil(|py| {
        let dict = PyDict::new_bound(py);
        
        let energies: Vec<f64> = trajectory.iter().map(|(e, _)| *e).collect();
        let energies_array = PyArray1::from_vec_bound(py, energies);
        dict.set_item("energies", energies_array)?;
        
        // Return last configuration
        let last_config = &trajectory.last().unwrap().1;
        let config_array = PyArray1::from_vec_bound(py, last_config.spins.clone());
        dict.set_item("final_config", config_array)?;
        
        // Return all configurations
        let all_configs: Vec<Vec<u8>> = trajectory.iter()
            .map(|(_, c)| c.spins.clone())
            .collect();
        let configs_array = PyArray2::from_vec2_bound(py, &all_configs)?;
        dict.set_item("all_configs", configs_array)?;
        
        Ok(dict.into())
    })
}

/// Run oscillatory deformation
#[pyfunction]
fn run_deformation(
    n: usize,
    k: usize,
    couplings: Vec<Vec<usize>>,
    a: Vec<f64>,
    b: Vec<f64>,
    initial_spins: Vec<u8>,
    gamma_max: f64,
    dgamma: f64,
    total_steps: usize,
    dump_zero_strain_only: bool,
) -> PyResult<PyObject> {
    let lattice = NKLattice { n, k, couplings, a, b };
    let initial_config = SpinConfig::from_vec(initial_spins);
    
    let result = oscillatory_deformation(
        &lattice,
        &initial_config,
        gamma_max,
        dgamma,
        total_steps,
        dump_zero_strain_only,
    );
    
    // Use the new from_result method for correct order parameter
    let observables = DeformationObservables::from_result(&result);
    let hysteresis_area = compute_hysteresis_area(&result.trajectory, dgamma);
    
    Python::with_gil(|py| {
        let dict = PyDict::new_bound(py);
        
        dict.set_item("gamma", PyArray1::from_vec_bound(py, observables.gamma_values))?;
        dict.set_item("energy", PyArray1::from_vec_bound(py, observables.energy_values))?;
        dict.set_item("msd", PyArray1::from_vec_bound(py, observables.msd_from_initial))?;
        dict.set_item("hamming", PyArray1::from_vec_bound(py, observables.hamming_from_initial))?;
        // Full-cycle displacement (correct order parameter for absorbing transition)
        dict.set_item("hamming_cycle", PyArray1::from_vec_bound(py, observables.full_cycle_displacement))?;
        dict.set_item("hysteresis_area", hysteresis_area)?;
        
        // Return gamma_acc values from trajectory
        let gamma_acc: Vec<f64> = result.trajectory.iter().map(|s| s.gamma_acc).collect();
        dict.set_item("gamma_acc", PyArray1::from_vec_bound(py, gamma_acc))?;
        
        // Return final configuration
        if let Some(last) = result.trajectory.last() {
            let final_config = PyArray1::from_vec_bound(py, last.config.spins.clone());
            dict.set_item("final_config", final_config)?;
        }
        
        // Return number of zero-strain configs (for diagnostics)
        dict.set_item("n_zero_strain_configs", result.zero_strain_configs.len())?;
        
        Ok(dict.into())
    })
}

/// Compute energy of a configuration
#[pyfunction]
fn compute_energy_py(
    n: usize,
    k: usize,
    couplings: Vec<Vec<usize>>,
    a: Vec<f64>,
    b: Vec<f64>,
    spins: Vec<u8>,
    gamma: f64,
) -> PyResult<f64> {
    let lattice = NKLattice { n, k, couplings, a, b };
    let config = SpinConfig::from_vec(spins);
    Ok(compute_energy(&lattice, &config, gamma))
}

/// Minimize a configuration to local minimum
#[pyfunction]
fn minimize_config(
    n: usize,
    k: usize,
    couplings: Vec<Vec<usize>>,
    a: Vec<f64>,
    b: Vec<f64>,
    spins: Vec<u8>,
    gamma: f64,
) -> PyResult<Py<PyArray1<u8>>> {
    let lattice = NKLattice { n, k, couplings, a, b };
    let mut config = SpinConfig::from_vec(spins);
    
    minimize_to_local_minimum(&lattice, &mut config, gamma);
    
    Python::with_gil(|py| {
        Ok(PyArray1::from_vec_bound(py, config.spins).into())
    })
}

/// NK model core library
#[pymodule]
fn nk_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(create_lattice, m)?)?;
    m.add_function(wrap_pyfunction!(create_spin_config, m)?)?;
    m.add_function(wrap_pyfunction!(run_equilibration, m)?)?;
    m.add_function(wrap_pyfunction!(run_deformation, m)?)?;
    m.add_function(wrap_pyfunction!(compute_energy_py, m)?)?;
    m.add_function(wrap_pyfunction!(minimize_config, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_rust_only() {
        // Test that doesn't require Python
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(42);
        let lattice = NKLattice::new(10, 5, &mut rng);
        let config = SpinConfig::new(10, &mut rng);
        let energy = compute_energy(&lattice, &config, 0.0);
        assert!(energy.is_finite());
    }
}
