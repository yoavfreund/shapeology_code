import argparse
parser = argparse.ArgumentParser()
parser.add_argument("Environment", type=str, help="Local or AWS")
parser.add_argument("stack", type=str, help="The name of the stack")
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

patch_file = 'CSHL_patch_samples'
img_fp = patch_file+'/'+stack+'/'
feature_fp = patch_file+'_features/'+stack+'/'
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

@schema
class SampleFeatures(dj.Computed):
    definition="""
    -> Structure
    -----
    size_of_positive : int   #size of positive feature file
    size_of_negative : int   #size of negative feature file
    """

    bucket = "mousebrainatlas-data"
    client = get_s3_client(credFiles)
    def make(self, key):
        structure = (Structure & key).fetch1('structure')
        print('populating for ', structure, end='\n')
        for state in ['positive', 'negative']:
            s3_fp = feature_fp + structure + '/'+stack+'_'+structure+'_'+state+'.pkl'
            print(s3_fp)
            try:
                report = self.client.stat_object(self.bucket, s3_fp)
                key['size_of_'+state] = int(report.size / 1000)
            except:
                run('python3 {0}/Cell_generator.py {1} {2} {3} {4} {5}'.format(scripts_dir, stack, structure, state, yaml_file, patch_file))
                report = self.client.stat_object(self.bucket, s3_fp)
                key['size_of_'+state] = int(report.size / 1000)
        try:
            self.insert1(key)
        except:
            print('could not insert key=', key)

SampleFeatures.populate(reserve_jobs=True)

