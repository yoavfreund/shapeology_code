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
import pickle
import cv2
from time import time
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

def CDF(x):
    x=np.sort(x)
    size=x.shape[0]
    y=np.arange(0,size)/size
    return x,y

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

pkl_file = 'CSHL_cells_threshold/'
pkl_dir = os.environ['ROOT_DIR']+pkl_file

cell_fp = 'CSHL_cells_features/MD589/Properties/'
cell_dir = os.environ['ROOT_DIR'] + 'CSHL_cells_features/MD589/Properties/'

@schema
class Thresholds(dj.Computed):
    definition="""
    -> StructureLeft
    -----
    size : int   #size of file
    """

    bucket = "mousebrainatlas-data"
    client = get_s3_client(credFiles)
    def make(self, key):
        struc = (StructureLeft & key).fetch1('structure')
        print('populating for ', struc, end='\n')
        setup_download_from_s3(cell_fp+struc)
        thresholds = {}
        fp = []
        fp.append(cell_dir + struc + '/MD589_' + struc + '_negative.pkl')
        fp.append(cell_dir + struc + '/MD589_' + struc + '_positive.pkl')
        features = []
        labels = []
        tens = []
        for state in range(2):
            cells = pickle.load(open(fp[state], 'rb'))
            cells = cells.drop(['left', 'top'], 1)
            cells = np.asarray(cells)
            for k in range(len(cells)):
                cells[k][0] = cells[k][0][:10]
            origin = np.concatenate((np.array(list(cells[:, 0])), cells[:, 1:]), axis=1)
            features.extend(origin)
            labels.extend([state] * len(origin))
        features = np.array(features)
        labels = np.array(labels)
        for k in range(features.shape[1]):
            print(k)
            x1, y1 = CDF(features[labels == 1, k])
            x2, y2 = CDF(features[labels == 0, k])
            if len(x1) < len(x2):
                x, y = x1, y1
                xc, yc = x2, y2
            else:
                x, y = x2, y2
                xc, yc = x1, y1
            if len(x) > 50000:
                times = int(len(x)/50000)
                x = x[0::times]
                y = y[0::times]
            for i in range(len(x)):
                if x[i] in xc:
                    index = np.where(xc == x[i])[0][0]
                    y[i] = (y[i] + yc[index]) / 2
                else:
                    y[i] = (y[i] + yc[np.argmin(np.absolute(xc - x[i]))]) / 2
            ten = [x[np.argmin(np.absolute(y - 0.01 * (j + 1)))] for j in range(99)]
            tens.append(ten)
        thresholds[struc] = tens
        if not os.path.exists(pkl_dir):
            os.mkdir(pkl_dir)
        filename = pkl_file + struc + '.pkl'
        pickle.dump(thresholds, open(os.environ['ROOT_DIR']+filename, 'wb'))
        setup_upload_from_s3(filename, recursive=False)
        report = self.client.stat_object(self.bucket, filename)
        key['size'] = int(report.size / 1000)
        self.insert1(key)

Thresholds.populate(reserve_jobs=True)

