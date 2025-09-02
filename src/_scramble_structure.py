import math
import numpy as np
from scipy.stats import truncnorm

from pymatgen.alchemy.materials import TransformedStructure
from pymatgen.analysis.elasticity import Strain
from pymatgen.transformations.standard_transformations import DeformStructureTransformation

from atomate2.common.analysis.elastic import get_default_strain_states



def generate_elastic_deformations(
    stdev_of_normal_distribution: float = 0.02,
    order: int = 2,
):
    """
    THIS CODE WAS TAKEN (and modified) FROM THE ATOMATE2 PACKAGE!!!

    Generate elastic deformations.

    Parameters
    ----------
    order : int    Order of the tensor expansion to be determined. Can be either 2 or 3.

    Returns
    -------
    List[Deformation]    A list of deformations.
    """
    
    strain_states = get_default_strain_states(order)
    # random strain magnitudes, according to a normal distribution centered at 0 with some stdev
    min_z_val = -2
    max_z_val = 2
    strain_magnitudes = truncnorm.rvs(min_z_val, max_z_val, size=len(strain_states), scale=stdev_of_normal_distribution)

    strains = []
    for i in range(0,len(strain_states)):
        strains.extend([Strain.from_voigt(strain_magnitudes[i] * np.array(strain_states[i]))])

    deformations = [s.get_deformation_matrix() for s in strains]

    return deformations



def normalize_mag(abc, current_norm, desired_norm):
    ratio = current_norm / desired_norm
    abc_new = np.sqrt(abc**2 / ratio **2)
    abc_new = math.copysign(abc_new, abc) # ensure same sign is kept
    return abc_new



def scramble_structure(structure):
    # apply random perturbations to each atom, relative to the lattice constants
    stdev = 0.020 / len(structure) # set to 0.005/len(structure) for 1st active learning iteration
    min_z_val = -3
    max_z_val = 3
    for i in range(0, len(structure)):
        random_translation_mag = truncnorm.rvs(min_z_val, max_z_val, size=3, scale = stdev)
        # print(np.linalg.norm(random_translation_mag))
        structure.translate_sites(i, random_translation_mag)

     # deform the structure across 6 strain modes (e_11 ... e_66)
    deformations = generate_elastic_deformations(stdev_of_normal_distribution = 0.02) #set to 0.02 for 1st active learning iteration
    for i, deformation in enumerate(deformations):
        dst = DeformStructureTransformation(deformation=deformation)
        ts = TransformedStructure(structure, transformations=[dst])
        structure = ts.final_structure
    
    return structure
