import argparse
parser = argparse.ArgumentParser()
parser.add_argument("Environment", type=str, help="Local or AWS")
parser.add_argument("stack", type=str, help="The name of the stack")
args = parser.parse_args()

import datajoint as dj
import numpy as np
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

pkl_fp = 'CSHL_cells_images/'+stack+'/'
db_fp = 'CSHL_databases/'+stack+'/'
scripts_dir = os.environ['REPO_DIR']

@schema
class Sqlite(dj.Computed):
    definition="""
    -> Section
    -----
    size_of_db : int   #size of database file of each section
    size_of_pkl : int  #size of pkl file of each section
    """

    bucket = "mousebrainatlas-data"
    client = get_s3_client(credFiles)
    def make(self, key):
        section = (Section & key).fetch1('section_id')
        print('populating for ', section, end='\n')
        s3_pkl = pkl_fp + str(section) + '.pkl'
        s3_db = db_fp + str(section) + '.db'
        try:
            report_pkl = self.client.stat_object(self.bucket, s3_pkl)
            report_db = self.client.stat_object(self.bucket, s3_db)
        except:
            run('python {0}/Sqlite.py {1} {2} {3}'.format(scripts_dir, stack, section, yaml_file))
            report_pkl = self.client.stat_object(self.bucket, s3_pkl)
            report_db = self.client.stat_object(self.bucket, s3_db)
        key['size_of_db'] = int(report_db.size / 1000)
        key['size_of_pkl'] = int(report_pkl.size / 1000)
        self.insert1(key)

Sqlite.populate(suppress_errors=True,reserve_jobs=True)

