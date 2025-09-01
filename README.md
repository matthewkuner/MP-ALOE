# MP-ALOE
Accompanying code used for the generation of the MP-ALOE dataset.

This code was executed using Python version 3.9.16. It is likely that other python versions and package versions may work, but we have not tested this.

The `main.py` file can, in theory, be ran to reproduce the workflows used in the publication. However, we strongly suggest modifying the code to include parallelization across many cores/nodes. We did not include such code here. Set `test_mode` to False to run the full workflow.

`QBC.py` takes prototype structures, occupies them with all possible combinations of elements, guesses each structure's lattice parameter, and assesses whether each structure is well-covered by the existing committee of MLIPs. 

`featurize_structures.py` uses the pretrained M3GNet formation energy model from the matgl repo to featurize the candidate structures identified by `QBC.py`.

`downsample.py` uses DIRECT sampling to downsample the candidate structures identified previously. This creates a file `final_selected_structures_for_dft.txt`, which contains the (relative) path to the final list of selected structures. The user should then perform DFT calculations on the structures.

