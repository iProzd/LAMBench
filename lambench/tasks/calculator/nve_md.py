from lambench.models.ase_models import ASEModel
from ase import Atoms
from ase.calculators.calculator import Calculator
from ase.md.verlet import VelocityVerlet
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.units import fs
import numpy as np
import time
from lambench.metrics.utils import log_average
from typing import Optional
from lambench.tasks.calculator.nve_md_data import TEST_DATA


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
    results = []
    for atoms in test_data:
        result = nve_simulation_single(
            atoms,
            model.calc,
            num_steps=num_steps,
            timestep=timestep,
            temperature_K=temperature_K,
        )
        results.append(result)
    return aggregated_results(results)


def aggregated_results(results: list[dict[str, float]]) -> dict[str, float]:
    # Aggregate results
    aggregated_result = {
        "simulation_time": np.mean(
            [
                result["simulation_time"]
                if result["simulation_time"] is not None
                else np.nan
                for result in results
            ]
        ),
        "energy_std": log_average(
            [
                result["energy_std"] if result["energy_std"] is not None else np.nan
                for result in results
            ]
        ),
        "steps": np.mean(
            [result["steps"] if result["steps"] != 0 else np.nan for result in results]
        ),
        "slope": log_average([result["slope"] for result in results]),
    }
    return aggregated_result


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
            - 'energy_std': Standard deviation of total energy (eV).
            - 'steps': Total steps completed (int).
            - 'slope': Energy drift per step (eV/fs).
    """

    atoms.calc = calculator
    MaxwellBoltzmannDistribution(atoms, temperature_K=temperature_K)
    dyn = VelocityVerlet(atoms, timestep * fs)

    # Track energies and steps
    energies = []
    steps_done = 0

    def log_energy(a=atoms):
        nonlocal steps_done
        energies.append(a.get_total_energy())
        steps_done += 1

    dyn.attach(log_energy, interval=1)

    # Measure performance
    start_time = time.time()
    try:
        dyn.run(num_steps)
    except Exception as e:
        print(f"Simulation crashed after {steps_done} steps: {e}")
    end_time = time.time()

    # Compute metrics
    simulation_time = end_time - start_time
    energy_std = np.std(energies) if len(energies) > 1 else None

    # Perform linear fit on energies using np.linalg.lstsq
    if len(energies) > 1:
        times = np.arange(steps_done) * timestep * fs
        A = np.vstack([times, np.ones(len(times))]).T
        slope, _ = np.linalg.lstsq(A, energies, rcond=None)[0]
    else:
        slope = np.nan

    return {
        "simulation_time": simulation_time,  # Simulation efficiency
        "energy_std": energy_std,  # Energy stability
        "steps": steps_done,  # Simulation stability
        "slope": np.abs(slope),  # Energy drift
    }
