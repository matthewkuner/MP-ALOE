import numpy as np
import pandas as pd
import os
import glob
import json
from maml.sampling.direct import DIRECTSampler, BirchClustering, SelectKFromClusters


data_path = "candidate_structures_for_dft"

def run_DIRECT_sampling(data_path, threshold = 0.1):
    descriptor_files = sorted(glob.glob(data_path + "/*descriptors.json"))
    master_list = []
    for descriptor_file in descriptor_files:
        f = open(descriptor_file) 
        data = json.load(f)

        for index in range(0, len(data)):
            data[index]["file_path"] = os.path.join(
                descriptor_file.split("_descriptors.json")[0], 
                data[index]["file_name"]
            )
        master_list += data
        print(len(master_list))
    with open(f"{data_path}/all_descriptors_concatenated_together.json", 'w') as fout:
        json.dump(master_list , fout)

    df = pd.DataFrame(master_list)
    print(f"number of starting structures = {len(df)}")

    num_clusters = None
    select_k_from_clusters = 5
    weighting_PCs = True

    DIRECT_sampler = DIRECTSampler(
        structure_encoder=None,
        clustering=BirchClustering(n=num_clusters, threshold_init=threshold),
        select_k_from_clusters=SelectKFromClusters(k=select_k_from_clusters),
        weighting_PCs = weighting_PCs,
    )

    DIRECT_selection = DIRECT_sampler.fit_transform(pd.DataFrame(df["descriptor"].values.tolist()))
    
    print(
        f"DIRECT selected {len(DIRECT_selection['selected_indexes'])} structures from {len(DIRECT_selection['PCAfeatures'])} total structures."
    )

    selected_structures_df = df.iloc[DIRECT_selection['selected_indexes']]
    selected_filepaths = selected_structures_df["file_path"].tolist()
    with open(f"{data_path}/final_selected_structures_for_dft.txt", "w") as f:
        for filepath in selected_filepaths:
            f.write(f"{filepath}\n")
