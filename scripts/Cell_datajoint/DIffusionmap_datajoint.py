import datajoint as dj
import numpy as np
import json
from subprocess import call
import yaml
import sys, os
import pandas as pd

sys.path.append('./lib')
from utilities import *
sys.path.append('../lib')
from utils import run

# Connect to datajoint server
dj.conn()

# Define which schema you're using
stack = 'MD589'
yaml_file = 'shape_params-aws.yaml'
scripts_dir = os.environ['REPO_DIR']

schema = dj.schema('kui_diffusionmap')
schema.spawn_missing_classes()


@schema
class Process(dj.Computed):
    definition="""
    -> Structure
    -----
    size_positive : int   #size of file
    size_negative : int   #size of file
    """
    bucket = "mousebrainatlas-data"
    credFiles = './Cell_datajoint/VaultBrain/credFiles.yaml'
    client = get_s3_client(credFiles)
    def make(self, key):
        print('populating for ', key, end='')
        struc = (Structure & key).fetch1('structure')
        for state in ['positive', 'negative']:
            item_name = state+'_s3_fp'
            key_item = 'size_'+state
            s3_fp = (Structure & key).fetch1(item_name)
            try:
                report = self.client.stat_object(self.bucket, s3_fp)
                key[key_item] = int(report.size/1000000)
            except:
                run('python3 {0}/Cell_generator.py {1} {2} {3} {4}'.format(scripts_dir, stack, struc, state, yaml_file))
                report = self.client.stat_object(self.bucket, s3_fp)
                key[key_item] = int(report.size / 1000000)
        try:
            self.insert1(key)
        except:
            print('could not insert key=', key)

diffusion = Process()
diffusion.populate(reserve_jobs=True)
