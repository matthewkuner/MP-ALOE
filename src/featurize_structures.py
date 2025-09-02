import matgl
import json
import glob
from pathlib import Path
import numpy as np
from pymatgen.core.structure import Structure

base_folder_to_save = "../candidate_structures_for_dft"

def featurize_structures_in_folder(folder_path):
    struct_filepaths = glob.glob(folder_path + "/*.POSCAR")

    list_of_descriptor_dicts = []
    for struct_filepath in struct_filepaths:
        struct = Structure.from_file(struct_filepath)

        m3gnet_matgl = matgl.load_model("M3GNet-MP-2018.6.1-Eform")
        cur_descriptor = np.array(m3gnet_matgl.model.featurize_structure(
            struct, output_layers=['readout'])['readout'].detach()
        )[0]
        eform = m3gnet_matgl.model.predict_structure(struct).item() # the .item() gets the value from the torch.tensor object

        list_of_descriptor_dicts.append(
            {
                "descriptor": cur_descriptor.tolist(),
                "energies": eform,
                "file_name": Path(struct_filepath).name,
            } 
        )

    with open(f"{base_folder_to_save}/{Path(folder_path).name}_descriptors.json", 'w') as fout:
        json.dump(list_of_descriptor_dicts , fout)
