## Utilities
* Aws-jupyter.py

    Organize processes on aws. See details in [Cell_extractor_aws.md](../Cell_extractor_aws.md).
* README.md
* lib
    * permute.py
    * shape_utils.py
    * utils.py
## Main code
* extractPatches.py

    Extracts all cells from a given file. See details in [Cell_extractor_local.md](../Cell_extractor_local.md).
* Cells_extractor.py

    Extracts all cells from all images of a given directory. See details in [Cell_extractor_local.md](../Cell_extractor_local.md).
* SortFile_v2.py

    Permute cell files in a random order. See details in [Cell_extractor_local.md](../Cell_extractor_local.md).
* CreateVQs_v2.py

    Use vector quantization to conclude permuted cells into representative samples. See details in [Diffusionmap.md](../Diffusionmap.md)
* diffusionmap.py

    Train diffusion mappings based on representative cells.
* Sqlite.py

    Extract cell shape features of all cells from a given image and store them into a sqlite database.
* DM_visualization.py

    Visualize cell shape features in original images.
* Patch_features_extractor.py

    Extract features of given positive and negative samples of all landmarks of a brain.
* label_patch.py
* patch_normalizer.py
* old

    A directory for old version scripts.
* Cell_datajoint

    A directory for scripts setting up tasks on aws.

## Configuration
* shape_params.yaml

    Parameters for cell extracting.