import datajoint as dj
import numpy as np
import json
from subprocess import call
import yaml
import sys, os
import shutil
import pandas as pd
import ray

sys.path.append('./lib')
from utilities import *
sys.path.append('../lib')
from utils import run

credFiles= '/home/ubuntu/data/Github/VaultBrain/credFiles_aws.yaml'
#credFiles= '/Users/kuiqian/Github/VaultBrain/credFiles.yaml'
dj.config['database.host'] = get_dj_creds(credFiles)['database.host']
dj.config['database.user'] = get_dj_creds(credFiles)['database.user']
dj.config['database.port'] = get_dj_creds(credFiles)['database.port']
dj.config['database.password'] = get_dj_creds(credFiles)['database.password']
dj.conn()

schema = dj.schema('kui_diffusionmap')
schema.spawn_missing_classes()

stack = 'MD594'
yaml_file = 'shape_params-aws.yaml'
#yaml_file = 'shape_params.yaml'
img_file = '/CSHL_hsv/'+stack+'/'
img_fp = os.environ['ROOT_DIR']+img_file
scripts_dir = os.environ['REPO_DIR']

def setup_download_from_s3(rel_fp, recursive=True):
    s3_fp = 's3://mousebrainatlas-data/' + rel_fp
    local_fp = os.environ['ROOT_DIR'] + rel_fp

    if os.path.exists(local_fp):
        print('ALREADY DOWNLOADED FILE')
        return

    if recursive:
        run('aws s3 cp --recursive {0} {1}'.format(s3_fp, local_fp))
    else:
        run('aws s3 cp {0} {1}'.format(s3_fp, local_fp))

def setup_upload_from_s3(rel_fp, recursive=True):
    s3_fp = 's3://mousebrainatlas-data/' + rel_fp
    local_fp = os.environ['ROOT_DIR'] + rel_fp

    if recursive:
        run('aws s3 cp --recursive {0} {1}'.format(local_fp, s3_fp))
    else:
        run('aws s3 cp {0} {1}'.format(local_fp, s3_fp))

setup_download_from_s3('CSHL_patches_features/')

@schema
class ImageMap(dj.Computed):
    definition="""
    -> Section
    -----
    structure_number : int   #number of structures
    """

    bucket = "mousebrainatlas-data"
    client = get_s3_client(credFiles)
    def make(self, key):
        section = (Section & key).fetch1('section_id')
        print('populating for ', section, end='\n')
        key_item = 'structure_number'
        try:
            objects = os.listdir(img_fp)
            cpt = sum([len(files) for r, d, files in os.walk(img_fp)])
            key[key_item] = cpt
        except:
            run('python3 {0}/HSV_v2.py {1} {2} {3}'.format(scripts_dir, stack, section, yaml_file))
            setup_upload_from_s3(img_file)
            cpt = sum([len(files) for r, d, files in os.walk(img_fp)])
            key[key_item] = cpt
            shutil.rmtree(img_fp)
        self.insert1(key)

ImageMap.populate(reserve_jobs=True)

