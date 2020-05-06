import argparse
parser = argparse.ArgumentParser()
parser.add_argument("yaml", type=str, help="path to configure file")
parser.add_argument("stack", type=str, help="The name of the stack")
args = parser.parse_args()

import datajoint as dj
import numpy as np
import sys, os
import pandas as pd

sys.path.append('./lib')
from utilities import *
sys.path.append(os.environ['REPO_DIR'])
from Cells_extractor import compute
from lib.utils import run, configuration


dj.config['database.host'] = get_dj_creds(credFiles)['database.host']
dj.config['database.user'] = get_dj_creds(credFiles)['database.user']
dj.config['database.port'] = get_dj_creds(credFiles)['database.port']
dj.config['database.password'] = get_dj_creds(credFiles)['database.password']
dj.conn()

schema = dj.schema('kui_diffusionmap')
schema.spawn_missing_classes()

stack = args.stack
yamlfile = os.environ['REPO_DIR'] + args.yaml
params = configuration(yamlfile).getParams()
credFiles = params['paths']['credFiles']

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
        file = (RawfilesDK39 & {'section_id': section}).fetch1('path_to_raw_img')
        print('populating for ', section, end='\n')
        key = compute(file, yamlfile, stack, section, self.client, self.bucket, key)
        self.insert1(key)

Cells.populate(suppress_errors=True,reserve_jobs=True)


