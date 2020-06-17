## Guidelines for Cell Shape Analysis


## Cell Shape Analysis Pipeline

#### Running extractPatches.py on all sections of one brain
This script uses datajoint.populate and Cells_extractor.py to excute extractPatches.py on all sections of a given brain.
```
python Cell_datajoint/Cells_extract_datajoint.py yaml stack
```
```
optional arguments:
  yaml        Path to Yaml file with parameters
  stack       The name of the brain, type=str, default='MD594'
```





