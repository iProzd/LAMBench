from lambench.models.ase_models import ASEModel
from ase import Atoms
from ase.calculators.calculator import Calculator
from ase.md.verlet import VelocityVerlet
from ase.md.velocitydistribution import (
    MaxwellBoltzmannDistribution,
    Stationary,
    ZeroRotation,
)
from ase.units import fs
import numpy as np
import time
from typing import Optional
from lambench.tasks.calculator.nve_md_data import TEST_DATA
import logging


def run_md_nve_simulation(
    model: ASEModel,
    num_steps: int,
    timestep: float,
    temperature_K: int,
    test_data: Optional[list[Atoms]] = TEST_DATA,
) -> dict[str, float]:
    """
    This function runs NVE simulations for a list of test systems using the given model.
    """
    results = {}
    for atoms in test_data:
        result = nve_simulation_single(
            atoms,
            model.calc,
            num_steps=num_steps,
            timestep=timestep,
            temperature_K=temperature_K,
        )
        results[str(atoms.symbols)] = result
        logging.info(f"Simulation completed for system {str(atoms.symbols)}: {result}")
    return results


def nve_simulation_single(
    atoms: Atoms,
    calculator: Calculator,
    num_steps: int,
    timestep: float,
    temperature_K: int,
):
    """
    Run an NVE simulation using VelocityVerlet and return performance metrics.

    Parameters:
        atoms: ASE Atoms objects for simulation.
        calculator: ASE calculator to use for the simulation.
        num_steps (int): Number of steps to run.
        timestep (float): Time step in fs.
        temperature_K (int): Temperature in Kelvin.

    Returns:
        dict: A dictionary containing:
            - 'simulation_time': Time taken for the simulation (s).
            - 'steps': Total steps completed (int).
            - 'slope': Energy drift per step (eV/atom/ps).
    """

    atoms.calc = calculator
    MaxwellBoltzmannDistribution(atoms, temperature_K=temperature_K)
    Stationary(atoms)
    ZeroRotation(atoms)
    dyn = VelocityVerlet(atoms, timestep * fs)

    # Track energies and steps
    energies = []

    def log_energy(a=atoms):
        energies.append(a.get_total_energy())
        if energies[-1] > 1e10:
            # To allow for early stopping in case of divergence
            raise RuntimeError

    dyn.attach(log_energy, interval=1)

    # Measure performance
    start_time = time.time()
    try:
        dyn.run(num_steps)
    except Exception as e:
        print(f"Simulation crashed after {dyn.nsteps} steps: {e}")
    end_time = time.time()

    # Compute metrics
    simulation_time = end_time - start_time

    # Perform linear fit on energies using np.linalg.lstsq
    if len(energies) > 1:
        times = np.arange(dyn.nsteps + 1) * timestep * fs
        A = np.vstack([times, np.ones(len(times))]).T
        slope, _ = np.linalg.lstsq(A, energies, rcond=None)[0]
    else:
        slope = np.nan

    try:
        momenta_diff = np.linalg.norm(atoms.get_momenta().sum(axis=0))
    except Exception:
        momenta_diff = np.nan
    return {
        "simulation_time": simulation_time,  # Simulation efficiency, s
        "steps": dyn.nsteps,  # Simulation stability
        "momenta_diff": momenta_diff,  # Momentum conservation, amu · Å/fs
        "slope": np.abs(1000 * slope / len(atoms)),  # Energy drift, eV/atom/ps
    }
