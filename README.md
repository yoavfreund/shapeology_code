# Generating patches
#### process_file.py
A script for processing a single section.
#### run_job.py
Wrapper around extract Patches
#### extractPatches.py
Extract patches from a single tile
#### patch_normalizer.py
Normalize the patches - called by extractPatches.py
## Managing multiple ec2 instances
#### Controller.py
The controller runs a given script on a set of files on S3. The application is intended to run on a set of ec2 instances in parallel.
Lock files are used to insure that each file is processed exactly once.

This python file will be rewritten in a simpler way using datajoin

#### watchdxog.sh
Command put in cron job to run watchdog.py

#### watchdog.py
script to check whether controller is running and, if not, restart it.

# Analysis of patches
#### CreateVQs.py
#### diffusion_maps.py
#### label_patch.py
