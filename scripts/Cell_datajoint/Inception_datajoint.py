import argparse
parser = argparse.ArgumentParser()
parser.add_argument("Environment", type=str, help="Local or AWS")
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

MXNET_ROOTDIR = 'mxnet_models'
model_dir_name = 'inception-bn-blue-softmax'
model_name = 'inception-bn-blue-softmax'
scripts_dir = os.environ['REPO_DIR']


@schema
class CnnTraining(dj.Computed):
    definition="""
    -> Structure
    -----
    size_of_evalMetricHistory : int   #size of eval Metric History file
    """

    bucket = "mousebrainatlas-data"
    client = get_s3_client(credFiles)
    def make(self, key):
        structure = (Structure & key).fetch1('structure')
        print('populating for ', structure, end='\n')
        prefix = os.path.join(MXNET_ROOTDIR, model_dir_name + '_Kui', model_name + '_' + structure)
        s3_fp = prefix + '_evalMetricHistory.pkl'
        print(s3_fp)
        try:
            report = self.client.stat_object(self.bucket, s3_fp)
            key['size_of_evalMetricHistory'] = int(report.size)
        except:
            run('python3 {0}/Inception-bn_training.py {1}'.format(scripts_dir, structure))
            report = self.client.stat_object(self.bucket, s3_fp)
            key['size_of_evalMetricHistory'] = int(report.size)
        try:
            self.insert1(key)
        except:
            print('could not insert key=', key)

CnnTraining.populate(reserve_jobs=True)

