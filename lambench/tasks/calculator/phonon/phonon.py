from lambench.models.ase_models import ASEModel
from ase import Atoms
import phonopy
from lambench.tasks.calculator.phonon.phonon_utils import ase_to_phonopy_atoms


def run_phonon_simulation_single(
    atoms: Atoms,
    model: ASEModel,
    distance: float = 0.01,
):
    """
    Run a phonon simulation using the given model and return performance metrics.
    """

    # Step 1: Run relaxation
    atoms: Atoms = model.run_ase_relaxation(atoms, model.calc)
    if atoms is None:
        return None

    # Step 2: Convert ASE Atoms object to PhonopyAtoms object
    phonon_atoms = ase_to_phonopy_atoms(atoms)
    phonon = phonopy.Phonopy(
        phonon_atoms, supercell_matrix=[[2, 0, 0], [0, 2, 0], [0, 0, 2]]
    )

    # Step 3: Generate displacements
    phonon.generate_displacements(distance=distance, is_diagonal=False)

    # Step 4: Calculate force constants
    forcesets = []

    for frame in phonon.supercells_with_displacements:
        frame_atom = Atoms(
            cell=frame.cell,
            symbols=frame.symbols,
            scaled_positions=frame.scaled_positions,
            pbc=True,
        )
        frame_atom.calc = model.calc
        forces = frame_atom.get_forces()
        forcesets.append(forces)

    phonon.forces = forcesets
    phonon.produce_force_constants()
    phonon.symmetrize_force_constants()

    # phonon.auto_band_structure()
    # phonon.plot_band_structure().show()
