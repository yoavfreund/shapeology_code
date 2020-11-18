# Diffusion map
This documentation describes how to finish the diffusion map training step on your computer locally. 
This process can only be implemented on one computer.

## Usage
#### Step 1. Find representative samples via K-means
```
python CreateVQs_v2.py padded_size
```
```
positional arguments:
  --samples_size  Number of samples taken for K-means, default=500000
  --src_root      Path to directory containing permuted cell files, default=$tmp_dir/permute/
  --save_dir      Path to directory saving VQ files, default=$tmp_dir/vq/
  padded_size    One of the three padded size, 15, 51 or 201
```
This script is to use vector quantization conclude cells into representative samples.
* Find representative cells of a specific number first.
* Divide cells into clusters based on representative cells and take the 
the mean of each cluster as representative samples.