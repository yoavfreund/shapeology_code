# Guidelines for Shapeology
## About 
This project is meant to analyze brain images based on cell shape features. The functions are:
* Extracting cells from brain images
* Learning cell shape features
* Training decision trees with cell shape features
* Generating detection score maps
* Finding statistically significant regions

## Installation
Please read the [Installation.md](Installation.md) for the details of the installation process.

## Usage
### Step 1: Cell Extraction
This step is to extract cells from brain section images and permute the generated cell patches in a random order. 

[Cell_extractor_local.md](Cell_extractor_local.md) offers a guided example showing how to complete this step on your computer locally.

To speed up the process, you can refer to ([Cell_extractor_aws.md](Cell_extractor_aws.md)) for the method to run on multiple AWS instances.
Essential credential files of AWS and datajoint (**VaultBrain**) are required.

### Step 2: Diffusion Map Training
This step is to find representative cell patches via K-means++ and then use them to calculate diffusion map.

[Diffusionmap.md](Diffusionmap.md) provides a guided example showing how to complete this step on your computer.