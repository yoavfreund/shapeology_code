import argparse
parser = argparse.ArgumentParser()
parser.add_argument("Environment", type=str, help="Local or AWS")
parser.add_argument("stack", type=str, help="The name of the stack")
args = parser.parse_args()

import datajoint as dj
import numpy as np
import json
import yaml
import sys, os
import shutil
import pandas as pd

sys.path.append('./lib')
from utilities import *
sys.path.append('../lib')
from utils import run

if args.Environment == 'AWS':
    credFiles= '/home/ubuntu/data/Github/VaultBrain/credFiles_aws.yaml'
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

stack = args.stack

img_file = 'CSHL_scoremaps_new/'+stack+'/'
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

setup_download_from_s3('CSHL_patch_samples_features_V2/MD589/')
setup_download_from_s3('CSHL_patch_samples_features_V2/MD585/')


@schema
class ScoreMap(dj.Computed):
    definition="""
    -> Section
    -----
    structure_number : int   #number of structures
    """

    bucket = "mousebrainatlas-data"
    #client = get_s3_client(credFiles)
    def make(self, key):
        section = (Section & key).fetch1('section_id')
        print('populating for ', section, end='\n')
        key_item = 'structure_number'
        try:
            objects = os.listdir(img_fp)
            cpt = sum([len(files) for r, d, files in os.walk(img_fp)])
            key[key_item] = cpt
        except:
            run('python3 {0}/Scoremap_v2.py {1} {2} {3}'.format(scripts_dir, stack, section, yaml_file))
            cpt = sum([len(files) for r, d, files in os.walk(img_fp)])
            key[key_item] = cpt
            setup_upload_from_s3(img_fp)
            setup_upload_from_s3('CSHL_grid_features/'+stack+'/'+ str(section) + '/')
            shutil.rmtree(img_fp)
        self.insert1(key)

ScoreMap.populate(reserve_jobs=True)
setup_upload_from_s3('CSHL_scoremaps_new/down32/')
