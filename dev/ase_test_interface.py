from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor
import numpy as np
import dpdata
import glob
from pathlib import Path
from lamstare.infra.ood_database import OODRecord


def run_ase_dptest(
    calc,
    testpath: str,
):
    """
    Given a ASE calculator and a test path, run ASE dptest and return the results.
    """
    adptor = AseAtomsAdaptor()

    energy_err = []
    energy_pre = []
    energy_lab = []
    atom_num = []
    energy_err_per_atom = []
    force_err = []
    virial_err = []
    virial_err_per_atom = []
    max_ele_num = 120

    systems = [i.parent for i in testpath.rglob("type_map.raw")]
    assert systems, f"No systems found in the test data {testpath}."
    mix_type = any(systems[0].rglob("real_atom_types.npy"))

    for filepth in systems:
        if mix_type:
            sys = dpdata.MultiSystems()
            sys.load_systems_from_file(filepth, fmt="deepmd/npy/mixed")
        else:
            sys = dpdata.LabeledSystem(filepth, fmt="deepmd/npy")

        for ls in sys:
            for frame in ls:
                atoms = frame.to_ase_structure()[0]
                atoms.calc = calc
                ff = atoms.get_forces()

                energy_predict = np.array(atoms.get_potential_energy())
                if not np.isnan(energy_predict):
                    atomic_numbers = atoms.get_atomic_numbers()
                    atom_num.append(np.bincount(atomic_numbers, minlength=max_ele_num))

                    energy_pre.append(energy_predict)
                    energy_lab.append(frame.data["energies"])
                    energy_err.append(energy_predict - frame.data["energies"])
                    force_err.append(frame.data["forces"].squeeze(0) - np.array(ff))
                    energy_err_per_atom.append(energy_err[-1] / force_err[-1].shape[0])
                    try:
                        stress = atoms.get_stress()
                        stress_tensor = (
                            -np.array(
                                [
                                    [stress[0], stress[5], stress[4]],
                                    [stress[5], stress[1], stress[3]],
                                    [stress[4], stress[3], stress[2]],
                                ]
                            )
                            * atoms.get_volume()
                        )
                        virial_err.append(frame.data["virials"] - stress_tensor)
                        virial_err_per_atom.append(
                            virial_err[-1] / force_err[-1].shape[0]
                        )
                    except:
                        pass
                else:
                    pass

    atom_num = np.array(atom_num)
    energy_err = np.array(energy_err)
    energy_pre = np.array(energy_pre)
    energy_lab = np.array(energy_lab)
    shift_bias, _, _, _ = np.linalg.lstsq(atom_num, energy_err, rcond=1e-10)
    unbiased_energy = (
        energy_pre
        - (atom_num @ shift_bias.reshape(max_ele_num, -1)).reshape(-1)
        - energy_lab.squeeze()
    )
    unbiased_energy_err_per_a = unbiased_energy / atom_num.sum(-1)

    res = {
        "Energy MAE": [np.mean(np.abs(np.stack(unbiased_energy)))],
        "Energy RMSE": [np.sqrt(np.mean(np.square(unbiased_energy)))],
        "Energy MAE/Natoms": [np.mean(np.abs(np.stack(unbiased_energy_err_per_a)))],
        "Energy RMSE/Natoms": [np.sqrt(np.mean(np.square(unbiased_energy_err_per_a)))],
        "Force MAE": [np.mean(np.abs(np.concatenate(force_err)))],
        "Force RMSE": [np.sqrt(np.mean(np.square(np.concatenate(force_err))))],
    }
    if virial_err_per_atom != []:
        res.update(
            {
                "Virial MAE": [np.mean(np.abs(np.stack(virial_err)))],
                "Virial RMSE": [np.sqrt(np.mean(np.square(np.stack(virial_err))))],
                "Virial MAE/Natoms": [np.mean(np.abs(np.stack(virial_err_per_atom)))],
                "Virial RMSE/Natoms": [
                    np.sqrt(np.mean(np.square(np.stack(virial_err_per_atom))))
                ],
            }
        )
    return res


def ase_test(model_name, testpath_mapping):
    """
    To test your model, you need to implement an ASE calculator for your model.
    """
    if model_name == "DP":
        from deepmd.calculator import DP

        CALC = DP("/path/to/your/model.pth")
    elif model_name == "MACE":
        from mace.calculators import mace_mp

        CALC = mace_mp(model="medium", device="cuda", default_dtype="float64")
    else:
        raise ValueError(f"Model {model_name} not supported.")

    for ood_name, testpath in mapping.items():
        print(f"Processing {ood_name}, {testpath}")
        head_dptest_res = run_ase_dptest(CALC, testpath)
        with open(f"{ood_name}_dptest.json", "w") as f:
            json.dump(head_dptest_res, f, indent=4)


if __name__ == "__main__":
    mapping = {
        "head_A": ["/path/to/your/test_data"],
        "head_B": ["/path/to/your/test_data"],
    }
    ase_test("DP", mapping)
