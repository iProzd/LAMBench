from lambench.models.ase_models import ASEModel
from ase import Atoms
from ase.calculators.calculator import Calculator
from ase.md.verlet import VelocityVerlet
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.units import fs
import numpy as np
import time
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
            timestep=timestep,
            num_steps=num_steps,
            temperature_K=temperature_K,
        )
        results.append(result)

    # Aggregate results
    aggregated_result = {
        "simulation_time": np.mean(
            [result["simulation_time"] for result in results]
            if result["simulation_time"] is not None
            else np.nan
        ),
        "energy_std": np.mean(
            [result["energy_std"] for result in results]
            if result["energy_std"] is not None
            else np.nan
        ),
        "steps": np.mean(
            [result["steps"] for result in results] if result["steps"] != 0 else np.nan
        ),
        "slope": np.mean(
            [result["slope"] for result in results]
            if result["slope"] is not None
            else np.nan
        ),
    }

    return calculate_final_result(aggregated_result)


def nve_simulation_single(
    atoms: Atoms,
    calculator: Calculator,
    timestep: float,
    num_steps: int,
    temperature_K: int,
):
    """
    Run an NVE simulation using VelocityVerlet and return performance metrics.

    Parameters:
        atoms: ASE Atoms objects for simulation.
        calculator: ASE calculator to use for the simulation.
        timestep (float): Time step in fs.
        num_steps (int): Number of steps to run.
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
        slope = None

    return {
        "simulation_time": simulation_time,  # Simulation efficiency
        "energy_std": energy_std,  # Energy stability
        "steps": steps_done,  # Simulation stability
        "slope": slope,  # Energy drift
    }


def calculate_final_result(
    aggregated_result, division_protection: float = 1e-6
) -> dict[str, float]:
    """
    This function aggreate the results across all four metrics and return the final result.
    """
    final_result = np.log(
        aggregated_result["steps"]
        / (
            aggregated_result["simulation_time"]
            * (aggregated_result["energy_std"] + division_protection)
            * (np.abs(aggregated_result["slope"]) + division_protection)
        )
    )
    return {"NVE Score": final_result}
