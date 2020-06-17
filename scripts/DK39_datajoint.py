import argparse
parser = argparse.ArgumentParser()
parser.add_argument("Environment", type=str, help="Local or AWS or Muralis")
args = parser.parse_args()

import datajoint as dj
import numpy as np
import json
from subprocess import call
import yaml
import sys, os
import shutil
import pandas as pd
import ray

sys.path.append('./Cell_datajoint/lib')
from utilities import *
sys.path.append('./lib')
from utils import run

if args.Environment == 'AWS':
    credFiles= '/home/ubuntu/data/Github/VaultBrain/credFiles_aws.yaml'
    yaml_file = 'shape_params-aws.yaml'
elif args.Environment == 'Muralis':
    credFiles = '/home/k1qian/data/Github/VaultBrain/credFiles_aws.yaml'
    yaml_file = 'shape_params-aws.yaml'
else:
    credFiles= '/Users/kuiqian/Github/VaultBrain/credFiles.yaml'
    yaml_file = 'shape_params.yaml'
dj.config['database.host'] = get_dj_creds(credFiles)['database.host']
dj.config['database.user'] = get_dj_creds(credFiles)['database.user']
dj.config['database.port'] = get_dj_creds(credFiles)['database.port']
dj.config['database.password'] = get_dj_creds(credFiles)['database.password']
dj.conn()

schema = dj.schema('kui_diffusionmap')
schema.spawn_missing_classes()

@schema
class Brain(dj.Manual):
    definition = """
    brain : char(10)    # name of the Brain
    """

stack = 'DK39'
print("\nAdding "+ stack+ " to the database")
Brain.insert1(dict(brain=stack),skip_duplicates=True)

@schema
class FileInfo(dj.Computed):
    definition="""
    -> Brain
    -----
    path_to_permute: char(200)       # path to permutation files
    size_of_permute: int             # size of permutation files
    path_to_vq: char(200)            # path to VQ files
    path_to_diffusionmap: char(200)  # path to diffusion map files
    """

    def make(self, key):
        stack = (Brain & key).fetch1('Brain')
        print('populating for ', stack, end='\n')
        for item in ['permute','VQ','diffusionMap']:
            if item=='diffusionMap':
                fp = os.environ['REPO_DIR'] + item +'/'
                if os.path.exists(fp):
                    key['path_to_'+item] = fp
            else:
                fp = os.environ['ROOT_DIR'] + item + '/'
                if os.path.exists(fp):
                    key['path_to_' + item] = fp
            if item=='permute':
                fp = os.environ['ROOT_DIR'] + item + '/'
                size = sum([os.path.getsize(os.path.join(r,file)) for r, d, files in os.walk(fp) for file in files])
                key['size_of_'+item] = size/1e9
        self.insert1(key)

FileInfo.populate()

@schema
class RawfilesDK39(dj.Computed):
    definition="""
    -> SectionDK39
    -----
    path_to_raw_img: char(200)    # path to the corresponding raw image file
    """
    stack = 'DK39'
    raw_images_root = os.path.join('CSHL_data_processed/', stack, 'neuroglancer_input/')
    bucket = "mousebrainatlas-data"
    client = get_s3_client(credFiles)
    def make(self, key):
        section = (SectionDK39 & key).fetch1('section_id')
        print('populating for ', section, end='\n')
        key['path_to_raw_img'] = self.raw_images_root + str(section) + '.tif'
        self.insert1(key)




