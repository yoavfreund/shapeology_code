import argparse
parser = argparse.ArgumentParser()
parser.add_argument("Environment", type=str, help="Local or AWS or Muralis")
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
from utils import run, configuration

if args.Environment == 'AWS':
    credFiles= '/home/ubuntu/data/Github/VaultBrain/credFiles_aws.yaml'
    yaml_file = 'shape_params-aws.yaml'
elif args.Environment == 'Muralis':
    credFiles = '/home/k1qian/data/Github/VaultBrain/credFiles.yaml'
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
scripts_dir = os.environ['REPO_DIR']
yamlfile = os.environ['REPO_DIR'] + yaml_file
params=configuration(yamlfile).getParams()
size_thresholds = params['normalization']['size_thresholds']


@schema
class Cells(dj.Computed):
    definition="""
    -> SectionDK39
    -----
    size_of_15 : int   #size of pkl file whose padded size is 15
    size_of_51 : int   #size of pkl file whose padded size is 51
    size_of_201 : int   #size of pkl file whose padded size is 201
    """

    bucket = "mousebrainatlas-data"
    client = get_s3_client(credFiles)
    def make(self, key):
        section = (SectionDK39 & key).fetch1('section_id')
        print('populating for ', section, end='\n')
        run('python {0}/Cells_extractor.py {1} {2}'.format(scripts_dir, stack, section))
        for size in size_thresholds:
            key_item = 'size_of_' + str(size)
            s3_fp = stack + '/cells/' + 'cells-' + str(size) + '/' + str(section) + '.bin'
            report = self.client.stat_object(self.bucket, s3_fp)
            key[key_item] = int(report.size / 1000)
        self.insert1(key)

Cells.populate(suppress_errors=True,reserve_jobs=True)


