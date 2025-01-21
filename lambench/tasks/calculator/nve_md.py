from lambench.models.ase_models import ASEModel
from ase import Atoms
from ase.calculators.calculator import Calculator
from ase.md.verlet import VelocityVerlet
from ase.units import fs
import numpy as np
import time

TEST_DATA = [
    Atoms(
        "H2O", positions=[(0.0, 0.0, 0.0), (0.0, 0.757, 0.587), (0.0, -0.757, 0.587)]
    )  # Example system
]


def run_md_nve_simulation(test_systems: list[Atoms], model: ASEModel) -> None:
    pass


def nve_simulation_single(
    atoms: Atoms, calculator: Calculator, timestep=1.0, num_steps=1000
):
    """
    Run an NVE simulation using VelocityVerlet and return performance metrics.

    Parameters:
        atoms: ASE Atoms objects for simulation.
        calculator: ASE calculator to use for the simulation.
        timestep (float): Time step in fs.
        num_steps (int): Number of steps to run.

    Returns:
        dict: A dictionary containing:
            - 'speed': Average simulation speed in steps/second.
            - 'energy_std': Standard deviation of total energy (eV).
            - 'steps': Total steps completed (int).
    """

    atoms.calc = calculator
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
    speed = steps_done / simulation_time if simulation_time > 0 else 0
    energy_std = np.std(energies) if len(energies) > 1 else None

    return {
        "speed": speed,
        "energy_std": energy_std,
        "steps": steps_done,
    }
