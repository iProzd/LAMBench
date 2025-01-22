from lambench.tasks.calculator.nve_md import nve_simulation_single
import pytest
from ase import Atoms
from ase.calculators.emt import EMT


@pytest.fixture
def setup_testing_data():
    """Fixture to provide testing data for NVE simulation."""
    return Atoms(
        "H2O", positions=[(0.0, 0.0, 0.0), (0.0, 0.757, 0.587), (0.0, -0.757, 0.587)]
    )  # Example system


@pytest.fixture
def setup_calculator():
    """Fixture to provide an ASE calculator (EMT)."""
    return EMT()


def test_nve_simulation_metrics(setup_testing_data, setup_calculator):
    """Test NVE simulation metrics for speed, std, and steps."""
    result = nve_simulation_single(
        setup_testing_data, setup_calculator, timestep=1.0, num_steps=100
    )

    assert result["steps"] > 0, "Steps should be greater than zero."
    assert result["speed"] > 0, "Speed should be greater than zero."
    if result["energy_std"] is not None:
        assert (
            result["energy_std"] >= 0
        ), "Energy standard deviation should be non-negative."


def test_nve_simulation_crash_handling(setup_testing_data, setup_calculator):
    """Test crash handling by simulating an intentional crash."""
    atoms = setup_testing_data

    def faulty_calculator(a):
        """A faulty calculator to simulate a crash."""
        raise RuntimeError("Intentional crash for testing.")

    res = nve_simulation_single(atoms, faulty_calculator, num_steps=100)
    assert res["steps"] == 0, "Simulation should crash."
