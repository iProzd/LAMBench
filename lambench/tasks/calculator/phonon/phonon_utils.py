from ase import Atoms
from phonopy.structure.atoms import PhonopyAtoms


def ase_to_phonopy_atoms(atoms: Atoms) -> PhonopyAtoms:
    """
    Convert ASE Atoms object to PhonopyAtoms object.
    """
    # Extract atomic symbols and positions
    symbols = atoms.get_chemical_symbols()
    positions = atoms.get_positions()
    cell = atoms.get_cell()
    masses = atoms.get_masses()

    return PhonopyAtoms(symbols=symbols, positions=positions, cell=cell, masses=masses)
