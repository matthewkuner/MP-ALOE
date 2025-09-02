import pandas as pd
import itertools
import numpy as np
from pymatgen.core.periodic_table import Element
import copy

# accounts for slight overprediction of these lattice guesses vs. MP structures
general_multiplication_factor = 0.96  # set to 0.92 in 1st iter

noble_gas_multiplication_factor = 2
He_Ne_factor = 1.4
Ar_factor = 1.25
noble_gas_radii = {
    "He": 0.31 * noble_gas_multiplication_factor * He_Ne_factor,
    "Ne": 0.38 * noble_gas_multiplication_factor * He_Ne_factor,
    "Ar": 0.71 * noble_gas_multiplication_factor * Ar_factor,
    "Kr": 0.88 * noble_gas_multiplication_factor,
    "Xe": 1.08 * noble_gas_multiplication_factor,
}
hydrogen_radius_multiplication_factor = 3
medium_alkali_multiplication_factor = 0.9
cs_factor = 0.9


def get_radius_of_specie(specie_symbol):
    specie = Element(specie_symbol)
    if specie.symbol in noble_gas_radii.keys():
        radius = noble_gas_radii[specie.symbol]
    elif specie.symbol == "H":
        radius = specie.atomic_radius.real * hydrogen_radius_multiplication_factor
    elif specie.symbol in [
        "K",
        "Rb",
    ]:
        radius = specie.atomic_radius.real * medium_alkali_multiplication_factor
    elif specie.symbol == "Cs":
        radius = specie.atomic_radius.real * cs_factor
    else:
        radius = specie.atomic_radius.real
    return radius


def get_structure_with_guessed_lattice_params(structure):
    nn_info_df = pd.DataFrame(
        structure.get_neighbor_list(9),
        index=["center_indices", "points_indices", "offset_vectors", "distances"],
    ).T
    site_inds = [ind for ind in range(0, len(structure))]
    combos = itertools.product(site_inds, repeat=2)
    combos = sorted(set([tuple(sorted(combo)) for combo in combos]))
    min_dist_list = []
    for combo in combos:
        cur_nns = nn_info_df[
            (nn_info_df["center_indices"] == combo[0])
            & (nn_info_df["points_indices"] == combo[1])
        ]
        # print(cur_nns)
        min_dist_for_site_pair = min(cur_nns["distances"])
        specie_list = [structure.sites[i].specie for i in combo]
        symbol_list = [structure.sites[i].specie.symbol for i in combo]
        radii = [get_radius_of_specie(specie) for specie in specie_list]
        min_dist_predicted_for_site_pair = sum(radii)
        scale_ratio_implied = min_dist_predicted_for_site_pair / min_dist_for_site_pair
        min_dist_list.append(
            symbol_list
            + [min_dist_for_site_pair]
            + [min_dist_predicted_for_site_pair]
            + [scale_ratio_implied]
        )
    scale_ratio = (
        np.max(pd.DataFrame(min_dist_list).iloc[:, -1]) * general_multiplication_factor
    )
    structure_with_guessed_lattice_param = copy.deepcopy(structure)
    structure_with_guessed_lattice_param.scale_lattice(
        scale_ratio**3 * structure.volume
    )
    return structure_with_guessed_lattice_param


def get_packing_factor(structure):
    vol_taken_by_species = 0
    for site in structure.sites:
        radius = get_radius_of_specie(site.specie)
        vol_taken_by_species += (4 / 3) * np.pi * (radius**3)
    packing_factor = vol_taken_by_species / structure.volume
    return packing_factor
