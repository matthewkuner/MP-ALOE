#%%
import numpy as np
import pandas as pd
import copy
import itertools
import os
from pathlib import Path
from collections import Counter
from itertools import permutations
import random

import datetime

from pymatgen.core.structure import Structure
from pymatgen.core.periodic_table import Element
from pymatgen.io.vasp.sets import MPRelaxSet

import warnings
warnings.filterwarnings("ignore")

from _scramble_structure import scramble_structure
from _guess_lattice_param import get_structure_with_guessed_lattice_params
from _run_UIPs import evaluate_structures_using_UIPs
import shutil

energy_diff_threshold = 0.1 #eV/atom
force_diff_threshold = 0.1 #eV/Angstrom
stress_diff_threshold = 0.1 #eV/Angstrom^3
max_allowed_median_energy = 0.25 # eV/atom
max_allowed_median_force_mag = 50 # eV/Angstrom
max_allowed_median_stress_mag = 0.5 # 1 eV/Angstrom^3 = ~160 GPa

he_factor = 3
ne_factor = 2
noble_gas_factor = 5
actinide_factor = 4
lanthanide_factor = 3
Tc_factor = 4

base_folder_to_save = "../candidate_structures_for_dft"

arbitrary_elements = ["H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne", "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar", "K", "Ca"]

def query_by_committee(structure):

    def get_min_energy_diff(list_of_energy_vals):
        return np.std(list_of_energy_vals)  

    def get_min_force_diff(list_of_force_arrays):
        diffs = []
        force_combos = itertools.combinations(list_of_force_arrays, 2)
        for force_combo in force_combos:
            f1 = np.array(force_combo[0])
            f2 = np.array(force_combo[1])
            f12_list_of_mag_diff = np.linalg.norm(np.abs(f1 - f2), axis=1)
            f12_rmse_of_mag_diffs = np.sqrt(np.mean(np.square(f12_list_of_mag_diff)))
            diffs.append(f12_rmse_of_mag_diffs)

        return np.mean(diffs)

    def get_min_stress_diff(list_of_stress_arrays):
        diffs = []
        stress_combos = itertools.combinations(list_of_stress_arrays, 2)
        for stress_combo in stress_combos:
            s1 = np.array(stress_combo[0])
            s2 = np.array(stress_combo[1])
            s12_abs_stress_diff = np.abs(s1 - s2)
            s12_mag_of_stress_diff = np.linalg.norm(s12_abs_stress_diff)
            diffs.append(s12_mag_of_stress_diff)

        return np.mean(diffs)
    
    df = evaluate_structures_using_UIPs(structure)
    energy_diff = get_min_energy_diff(df["energy"].to_list())
    force_diff = get_min_force_diff(df["forces"].to_list())
    stress_diff = get_min_stress_diff(df["stress"].to_list())

    # Calculate energies, max force magnitudes, and stress magnitudes as a
    # proxy for structure reasonableness. These are later used to
    # throw out unreasonable structures.
    list_of_energies = df["energy"]
    list_of_max_force_mags = []
    for i in range(0, len(df["forces"])):
        cur_forces = df["forces"].iloc[i]
        force_mags = np.linalg.norm(cur_forces, axis=1)
        max_force_mag = max(force_mags)
        list_of_max_force_mags.append(max_force_mag)
    list_of_stress_mags = []
    for i in range(0, len(df["stress"])):
        cur_stresses = df["stress"].iloc[i]
        stress_mag = abs(np.mean(cur_stresses[0:3]))
        list_of_stress_mags.append(stress_mag)

    return energy_diff, force_diff, stress_diff, list_of_energies, list_of_max_force_mags, list_of_stress_mags, df


# get all species to use in structure generation
def get_all_ele_combos(num_components: int):
    elements_to_exclude = ["Po", "At", "Rn", "Fr", "Ra"]
    species_list = []
    for i in range(1,95):
        symbol = Element.from_Z(i).symbol
        ele = Element.from_Z(i)
        if symbol not in elements_to_exclude:
            species_list.append(symbol)
    print(species_list)
    print(f"length of species list: {len(species_list)}")
    all_ele_combos = list(itertools.combinations(species_list, num_components))

    return all_ele_combos


def find_unique_ele_permutations(prototype_structure):
    """
    Say you have a prototype structure with elements H and He, and you want
    to substitute in the species Fe and Co. This function checks whether
    (H --> Fe) + (He --> Co) is equivalent to (H --> Co) + (He --> Fe).
    If they are equivalent, only one structure would be returned. If they are
    different, two structures would be returned (one for each substitution pairing).
    
    This function is generalized to perform this action for any number of
    species substitutions (e.g. [H, He, Li] being changed to [Fe, Co, Ni])
    """
    # get elements in prototype_structure
    elements = [ele.symbol for ele in prototype_structure.elements]

    permuts = permutations(elements, len(elements))
    list_of_uniquely_permuted_structs = []
    for permut in permuts:
        new_struct = copy.deepcopy(prototype_structure)

        # substitute prototype structure species to current permutation of species
        specMap = dict(zip(elements, permut))
        new_struct.replace_species(specMap)

        is_unique = True
        for unique_struct in list_of_uniquely_permuted_structs:
            if new_struct.matches(unique_struct, primitive_cell = False, ltol = 0.01, stol = 0.01,):
                is_unique = False
                break
        if is_unique:
            list_of_uniquely_permuted_structs.append(new_struct)

    return list_of_uniquely_permuted_structs


def run_QBC_for_prototype_struct(prototype_filepath, test_mode = False):

    if test_mode:
        energy_diff_threshold = 0.01 
        force_diff_threshold = 0.01 
        stress_diff_threshold = 0.01
        max_allowed_median_energy = 100
        max_allowed_median_force_mag = 100
        max_allowed_median_stress_mag = 100

    prototype_info = Path(prototype_filepath).name.split("_", 2)[2]
    weight = 1
    if "weight" in prototype_info:
        prototype_info, weight = prototype_info.rsplit("_weight")
        weight = np.sqrt(float(weight)) / 3
    
    # load in prototype struct
    prototype_structure = Structure.from_file(prototype_filepath)


    errors = []
    structs_with_big_PES_gap = []
    num_structs_due_to_energy_disagreement = 0
    num_structs_due_to_force_disagreement = 0
    num_structs_due_to_stress_disagreement = 0

    num_too_high_energy = 0
    num_too_high_force = 0
    num_too_high_stress = 0

    # get all unique ways to occupy the prototype structure
    unique_prototype_site_occupations = find_unique_ele_permutations(prototype_structure)
    print(len(unique_prototype_site_occupations))

    cur_1st_ele = None
    all_ele_combos = get_all_ele_combos(prototype_structure.n_elems)
    print(len(all_ele_combos))
    for ele_combo in all_ele_combos:

        # below is to print out info for debugging and monitoring progress, nothing more
        if cur_1st_ele != ele_combo[0]:
            cur_1st_ele = ele_combo[0]
            print(f"number of structs added due to energy, force, and stress disagreements: {num_structs_due_to_energy_disagreement, num_structs_due_to_force_disagreement, num_structs_due_to_stress_disagreement}")
            print(f"num of structs thrown out due to having too high energy, force, stress: {num_too_high_energy, num_too_high_force, num_too_high_stress}")
            print("total number of structs evaluated:", len(errors), "\n")
            print("starting on structs with:", cur_1st_ele)  # track progress visually


        if len(ele_combo) != prototype_structure.n_elems:
            raise ValueError("number of prototype_structure eles and number of desired eles to substitute into struct do not match!")
        
        for struct_throwaway in unique_prototype_site_occupations:
            # Randomly skip ternary structures due to combinatorial explosion of 
            # possible compositions.
            if random.random() > weight:
                # print('Skipping this structure randomly...')
                continue


            structure = copy.deepcopy(struct_throwaway)
            specMap = dict(zip(arbitrary_elements, ele_combo))
            structure.replace_species(specMap)
            

            # guess the lattice param based on pymatgen Element.atomic_radius
            structure = get_structure_with_guessed_lattice_params(structure)
            # apply random perturbations to atom positions and lattice
            structure = scramble_structure(structure)


            energy_diff, force_diff, stress_diff, energies_list, max_force_mag_list, stress_mag_list, df_results = query_by_committee(structure)

            # if predicted energies are too high, do not include the structure
            # if predicted forces are too high, do not include the structure
            # if predicted stresses are too high, do not include the structure
            struct_too_extreme = False
            if np.median(energies_list) > max_allowed_median_energy:
                struct_too_extreme = True
                num_too_high_energy += 1
            if np.median(max_force_mag_list) > max_allowed_median_force_mag:
                struct_too_extreme = True
                num_too_high_force += 1
            if np.median(stress_mag_list) > max_allowed_median_stress_mag:
                struct_too_extreme = True
                num_too_high_stress += 1

            if struct_too_extreme:
                continue


            # get multiplication factors for thresholds, used to reduce number of
            # undesirable structures selected.
            mult_factor = 1
            # reduce likelihood of including structures with noble gases
            has_noble_ele = False
            # has_ne_he = False
            has_ne = False
            has_he = False
            has_Tc = False
            has_lanthanoid = False
            has_actinoid = False
            for ele in ele_combo:
                if Element(ele).is_noble_gas:
                    has_noble_ele = True
                if ele in ["He"]:
                    has_he = True
                if ele in ["Ne"]:
                    has_ne = True
                if ele in ["Tc"]:
                    has_Tc = True
                if Element(ele).is_actinoid:
                    has_actinoid = True
                if Element(ele).is_lanthanoid:
                    has_lanthanoid = True
            if has_noble_ele:
                mult_factor *= noble_gas_factor
            if has_he:
                mult_factor *= he_factor
            if has_ne:
                mult_factor *= ne_factor
            if has_Tc:
                mult_factor *= Tc_factor
            if has_actinoid:
                mult_factor *= actinide_factor
            if has_lanthanoid:
                mult_factor *= lanthanide_factor

            # weight structures according to how many sites they have --> we want smaller structures
            mult_factor *= np.sqrt(len(structure))/np.sqrt(2)

            cur_energy_diff_threshold = energy_diff_threshold * mult_factor
            cur_force_diff_threshold = force_diff_threshold * mult_factor
            cur_stress_diff_threshold = stress_diff_threshold * mult_factor
            if energy_diff > cur_energy_diff_threshold:
                num_structs_due_to_energy_disagreement += 1
            if force_diff > cur_force_diff_threshold:
                num_structs_due_to_force_disagreement += 1
            if stress_diff > cur_stress_diff_threshold:
                num_structs_due_to_stress_disagreement += 1

            if (energy_diff > cur_energy_diff_threshold) or (force_diff > cur_force_diff_threshold) or (stress_diff > cur_stress_diff_threshold):
                structs_with_big_PES_gap.append(structure)


            # log information for analysis in post
            energy_diff = round(energy_diff, 8)
            force_diff = round(force_diff, 8)
            stress_diff = round(stress_diff, 8)
            errors.append([energy_diff, force_diff, stress_diff])

        if test_mode: # only run a few structures for testing purposes
            if len(errors) > 0:
                break


    # write structures of interest to POSCAR files
    # folder_to_save_structs_in = base_folder_to_save + "/" + datetime.datetime.utcnow().strftime("%Y-%m-%d_%H-%M-%S.%f")
    folder_to_save_structs_in = os.path.join(base_folder_to_save, prototype_info)
    try:
        os.mkdir(base_folder_to_save)
    except:
        pass
    try:
        if os.path.exists(folder_to_save_structs_in):
            shutil.rmtree(folder_to_save_structs_in)
        os.mkdir(folder_to_save_structs_in)
    except:
        pass


    # write candidate structures to file AND calculate their CHGNet descriptor
    for struct in structs_with_big_PES_gap:
        formula = struct.formula
        formula = "_".join(sorted(formula.split(" ")))
        
        # ensure a unique filename is made for each permuted occupation of a prototype
        permut_number = 1
        struct_identifier = f"{formula}_permut{permut_number}_{prototype_info}.POSCAR"

        new_file_path = f"{folder_to_save_structs_in}/{struct_identifier}"
        while os.path.isfile(new_file_path):
            permut_number += 1
            new_file_path = f"{folder_to_save_structs_in}/{formula}_permut{permut_number}_{prototype_info}.POSCAR"

        # write new file
        with open(new_file_path, 'w') as f:  
            print(MPRelaxSet(structure=struct).poscar, file=f) 


    # write ensemble metrics to csv
    column_names = ["energy_disagreement", "force_disagreement", "stress_disagreement"]
    df = pd.DataFrame(errors, columns = column_names)
    # round data to make files smaller
    df = df.round(4)
    metrics_filename = f"{folder_to_save_structs_in}/_ensemble_metrics"
    df.to_csv(f"{metrics_filename}.csv")


    print('outputting summary stats')
    print(f"number of candidate structures / total structs: {len(structs_with_big_PES_gap)} / {len(errors)}")
    errors = np.array(errors)

    for percentile in [5,25,50,75,95]:
        print(f"{percentile}th percentile error: {np.percentile(errors, percentile, axis=0)} [energy, force, stress]")
    ele_counts = []
    for struct in structs_with_big_PES_gap:
        ele_counts += [specie.symbol for specie in struct.species]
    print("counts of each element and the total number of elements represented")
    print(Counter(ele_counts))
    print(len(Counter(ele_counts)))