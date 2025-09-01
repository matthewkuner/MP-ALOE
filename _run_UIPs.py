from mace.calculators import MACECalculator
from pymatgen.io.ase import AseAtomsAdaptor
import pandas as pd
import json

import torch
torch.set_default_dtype(torch.float32)


mace_calculator_1 = MACECalculator(
    model_paths = "models/mace_iter_2_model_1-cpu.model", # insert path to model 1
    default_dtype="float32",
    device="cpu"
)

mace_calculator_2 = MACECalculator(
    model_paths = "models/mace_iter_2_model_2-cpu.model", # insert path to model 2
    default_dtype="float32",
    device="cpu"
)

mace_calculator_3 = MACECalculator(
    model_paths = "models/mace_iter_2_model_3-cpu.model", # insert path to model 3
    default_dtype="float32",
    device="cpu"
)

calculators = [mace_calculator_1, mace_calculator_2, mace_calculator_3]

mace_elemental_energy_filepath = "formation_energies_mace_avg_iter_2.json"
mace_avg_elemental_energies = json.load(open(mace_elemental_energy_filepath))


def calculate_formation_energy(total_energy, structure, elemental_energies):
    for site in structure.sites:
        symbol = site.specie.symbol
        total_energy -= elemental_energies[symbol]

    return total_energy


def evaluate_structures_using_UIPs(structure, convert_to_formation_energies = True):
    results = []
    model_names = []
    adaptor = AseAtomsAdaptor()
    atoms = adaptor.get_atoms(structure)

    for i in range(0, len(calculators)):
        calculator = calculators[i]
        atoms.calc = calculator

        cur_energy = atoms.get_potential_energy()
        if convert_to_formation_energies:
            cur_energy = calculate_formation_energy(
                cur_energy, 
                structure, 
                mace_avg_elemental_energies
            ) # convert to formation energy
        cur_energy /= len(structure)

        cur_forces = atoms.get_forces()
        cur_stress = atoms.get_stress()

        results.append({"energy": cur_energy, "forces": cur_forces, "stress": cur_stress})
        model_names.append(f"mace_{i+1}")

    df = pd.DataFrame(results, index = model_names,)

    return df