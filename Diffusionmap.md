# Diffusion map

### 1. Find representative samples via K-means
```
python CreateVQs_v2.py padded_size  samples_size  vq_dir
```
```
positional arguments:
  padded_size    One of the three padded size, 15, 51 or 201
  samples_size   Number of samples taken for K-means
  vq_dir         directory for storing VQs
```
This script is to conclude cells into representative samples with 
a limited number.
* Find representative cells of a specific number first.
* Divide cells into clusters based on representative cells and take the 
the mean of each cluster as representative samples.