import argparse
parser = argparse.ArgumentParser()
parser.add_argument("stack", type=str, help="The name of the stack")
args = parser.parse_args()

import datajoint as dj
import numpy as np
import sys, os
import pandas as pd

sys.path.append('./lib')
from utilities import *
sys.path.append(os.environ['REPO_DIR'])
from lib.utils import run, configuration

def setup_upload_from_s3(rel_fp, recursive=True):
    s3_fp = 's3://mousebrainatlas-data/' + rel_fp
    local_fp = os.environ['ROOT_DIR'] + rel_fp

    if recursive:
        run('aws s3 cp --recursive {0} {1}'.format(local_fp, s3_fp))
    else:
        run('aws s3 cp {0} {1}'.format(local_fp, s3_fp))


stack = args.stack

credFiles = os.path.join(os.environ['VAULT'], 'credFiles.yaml')

dj.config['database.host'] = get_dj_creds(credFiles)['database.host']
dj.config['database.user'] = get_dj_creds(credFiles)['database.user']
dj.config['database.password'] = get_dj_creds(credFiles)['database.password']
dj.conn()

schema = dj.schema(get_dj_creds(credFiles)['schema'])
schema.spawn_missing_classes()

# pkl_fp = 'CSHL_shift_scores/'+stack+'_search/'
pkl_fp = 'CSHL_shift_scores/'+stack+'_correct/'
scripts_dir = os.environ['REPO_DIR']

@schema
class Shift3D(dj.Computed):
    definition="""
    -> SectionDK52
    -----
    size_of_file : int   #size of pkl file of each section
    """

    bucket = "mousebrainatlas-data"
    client = get_s3_client(credFiles)
    def make(self, key):
        section = (SectionDK52 & key).fetch1('section_id')
        print('populating for ', section, end='\n')
        key_item = 'size_of_file'
        s3_fp = pkl_fp + str(section) + '.pkl'
        try:
            report = self.client.stat_object(self.bucket, s3_fp)
        except:
            run('python {0}/Shape_shift_3D.py {1} {2}'.format(scripts_dir, stack, section))
            setup_upload_from_s3(s3_fp, recursive=False)
            report = self.client.stat_object(self.bucket, s3_fp)
        key[key_item] = int(report.size / 1000)
        self.insert1(key)


Shift3D.populate(suppress_errors=True,reserve_jobs=True)


