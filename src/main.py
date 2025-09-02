#%%
# from QBC import run_QBC_for_prototype_struct
# from featurize_structures import featurize_structures_in_folder
# from downsample import run_DIRECT_sampling
import qbc
import featurize_structures
import downsample
import os

test_mode = True # set to False to run full workflow!

# # step 1: run QBC on all generated structures
prototype_struct_folder = "../data/prototype_structures"
list_of_prototype_structs = [f for f in os.listdir(prototype_struct_folder) if "POSCAR" in f]
if test_mode: list_of_prototype_structs = list_of_prototype_structs[:20] 

for struct_filename in list_of_prototype_structs:
    print(struct_filename)
    struct_filepath = os.path.join(prototype_struct_folder, struct_filename)
    qbc.run_QBC_for_prototype_struct(
        struct_filepath,
        test_mode = test_mode
    )

# step 2: prepare M3GNet descriptors for all QBC-identified candidate structures (used in next step)
candidate_structures_folder = "../candidate_structures_for_dft"
for folder in os.listdir(candidate_structures_folder):
    folder_path = os.path.join(candidate_structures_folder, folder)

    if not os.path.isdir(folder_path):
        continue

    if os.path.isdir(folder_path):
        featurize_structures.featurize_structures_in_folder(folder_path)

# step 3: downsample the QBC-identified candidate structures using DIRECT sampling
downsample.run_DIRECT_sampling(candidate_structures_folder, threshold = 0.1)

# %%
