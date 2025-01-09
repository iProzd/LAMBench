from pathlib import Path
from lambench.models.basemodel import BaseLargeAtomModel
import logging
from pymatgen.io.ase import AseAtomsAdaptor
from ase.calculators.calculator import Calculator
import numpy as np
import dpdata
from typing import Optional

from lambench.tasks.direct.direct_predict import DirectPredictTask


class ASEModel(BaseLargeAtomModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.model_type != "ASE":
            raise ValueError(
                f"Model type {self.model_type} is not supported by ASEModel"
            )

    def evaluate(self, task: DirectPredictTask) -> Optional[dict[str, float]]:
        if task.target_name != "standard":
            logging.error(f"ASEModel does not support target_name {task.target_name}")
            return
        else:
            if self.model_id.lower().startswith("mace"):
                from mace.calculators import mace_mp

                CALC = mace_mp(model="medium", device="cuda", default_dtype="float64")
            elif self.model_id.lower().startswith("orb"):
                from orb_models.forcefield import pretrained
                from orb_models.forcefield.calculator import ORBCalculator

                orbff = pretrained.orb_v2(device="cuda")  # orb-v2-20241011.ckpt
                CALC = ORBCalculator(orbff, device="cuda")
            elif self.model_id.lower().startswith("7net"):
                from sevenn.sevennet_calculator import SevenNetCalculator

                CALC = SevenNetCalculator("7net-0_11July2024", device="cuda")
            elif self.model_id.lower().startswith("eqv2"):
                from fairchem.core import OCPCalculator

                CALC = OCPCalculator(
                    checkpoint_path="eqV2_153M_omat_mp_salex.pt", cpu=False
                )
            elif self.model_id.lower().startswith("mattersim"):
                from mattersim.forcefield import MatterSimCalculator

                CALC = MatterSimCalculator(
                    load_path="MatterSim-v1.0.0-5M.pth", device="cuda"
                )
            elif self.model_id.lower().startswith("dp"):
                logging.error("Please use DPModel for DP models.")
                return
            else:
                logging.error(f"Model {self.model_id} is not supported by ASEModel")
                return
            return self.run_ase_dptest(CALC, task.test_data)


    @staticmethod
    def run_ase_dptest(calc: Calculator, test_data: Path) -> dict:
        adaptor = AseAtomsAdaptor() # FIXME: this is not used
        energy_err = []
        energy_pre = []
        energy_lab = []
        atom_num = []
        energy_err_per_atom = []
        force_err = []
        virial_err = []
        virial_err_per_atom = []
        max_ele_num = 120

        systems=[i.parent for i in test_data.rglob("type_map.raw")]
        assert systems, f"No systems found in the test data {test_data}."
        mix_type = any(systems[0].rglob('real_atom_types.npy'))

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
                    force_pred = atoms.get_forces()

                    energy_predict = np.array(atoms.get_potential_energy())
                    if not np.isnan(energy_predict):
                        atomic_numbers = atoms.get_atomic_numbers()
                        atom_num.append(
                            np.bincount(atomic_numbers, minlength=max_ele_num)
                        )

                        energy_pre.append(energy_predict)
                        energy_lab.append(frame.data["energies"])
                        energy_err.append(energy_predict - frame.data["energies"])
                        # TODO: handle the datasets without force labels
                        force_err.append(
                            frame.data["forces"].squeeze(0) - np.array(force_pred)
                        )
                        energy_err_per_atom.append(
                            energy_err[-1] / force_err[-1].shape[0]
                        )
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
                            pass # no virial in the data

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
            "Energy RMSE/Natoms": [
                np.sqrt(np.mean(np.square(unbiased_energy_err_per_a)))
            ],
            "Force MAE": [np.mean(np.abs(np.concatenate(force_err)))],
            "Force RMSE": [np.sqrt(np.mean(np.square(np.concatenate(force_err))))],
        }
        if virial_err_per_atom != []:
            res.update(
                {
                    "Virial MAE": [np.mean(np.abs(np.stack(virial_err)))],
                    "Virial RMSE": [np.sqrt(np.mean(np.square(np.stack(virial_err))))],
                    "Virial MAE/Natoms": [
                        np.mean(np.abs(np.stack(virial_err_per_atom)))
                    ],
                    "Virial RMSE/Natoms": [
                        np.sqrt(np.mean(np.square(np.stack(virial_err_per_atom))))
                    ],
                }
            )
        return res
