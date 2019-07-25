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

fp = os.path.join('CSHL_data_processed', stack, stack + '_sorted_filenames.txt')
setup_download_from_s3(fp, recursive=False)
with open(os.environ['ROOT_DIR']+fp, 'r') as f:
    fn_idx_tuples = [line.strip().split() for line in f.readlines()]
    section_to_filename = {int(idx): fn for fn, idx in fn_idx_tuples}

fname = os.path.join('CSHL_data_processed', stack, 'All_patch_locations_structures.pkl')
setup_download_from_s3(fname, recursive=False)
all_patch_locations = pickle.load(open(os.environ['ROOT_DIR']+fname, 'rb'), encoding='latin1')

raw_images_root = 'CSHL_data_processed/'+stack+'/'+stack+'_prep2_lossless_gray/'
img_file = 'CSHL_patch_samples/'
if not os.path.exists(os.environ['ROOT_DIR']+img_file):
    os.mkdir(os.environ['ROOT_DIR']+img_file)
img_file = img_file+stack+'/'
if not os.path.exists(os.environ['ROOT_DIR']+img_file):
    os.mkdir(os.environ['ROOT_DIR']+img_file)
img_dir = os.environ['ROOT_DIR']+img_file

patch_size = 224

@schema
class RandomPatches(dj.Computed):
    definition="""
    -> Structure
    -----
    Positive_patches_number : int   #number of positive patches
    Negative_patches_number : int   #number of negative patches
    """

    bucket = "mousebrainatlas-data"
    def make(self, key):
        struc = (Structure & key).fetch1('structure')
        print('populating for ', struc, end='\n')
        for state in ['Positive', 'Negative']:
            t1 = time()
            if state=='Negative':
                struc = struc + '_surround_500um_noclass'
            img_fp = img_dir + struc + '/'
            try:
                cpt = sum([len(files) for r, d, files in os.walk(img_fp)])
                key[state+'_patches_number'] = cpt
            except:
                os.mkdir(img_fp)
                for section in all_patch_locations[struc].keys():
                    section_fn = raw_images_root + section_to_filename[section] + '_prep2_lossless_gray.tif'
                    setup_download_from_s3(section_fn)
                    img = cv2.imread(os.environ['ROOT_DIR']+section_fn, 2)
                    n_choose = min(len(all_patch_locations[struc][section]), 10)
                    indices_choose = np.random.choice(range(len(all_patch_locations[struc][section])), n_choose,
                                                      replace=False)
                    patches_choose = all_patch_locations[struc][section][indices_choose, :]
                    for index in range(n_choose):
                        x = int(float(patches_choose[index][0]))
                        y = int(float(patches_choose[index][1]))
                        patch = img[y-patch_size/2:y+patch_size/2,x-patch_size/2:x+patch_size/2]
                        filename = img_fp + str(section) + '_' + str(index) + '.tif'
                        cv2.imwrite(filename, patch)
                    os.remove(os.environ['ROOT_DIR'] + section_fn)
                cpt = sum([len(files) for r, d, files in os.walk(img_fp)])
                key[state+'_patches_number'] = cpt
                setup_upload_from_s3(img_file + struc)
                shutil.rmtree(img_fp)
                print(struc + ' finished in %5.1f seconds' % (time() - t1))
        self.insert1(key)

RandomPatches.populate(reserve_jobs=True)

