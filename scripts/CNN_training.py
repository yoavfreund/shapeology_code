import argparse

parser = argparse.ArgumentParser()
parser.add_argument("structure", type=str, help="The nth group of structures")
args = parser.parse_args()
stack = args.stack
struc = args.structure
state = args.state

import cv2
# from cv2 import moments,HuMoments
import pickle
import numpy as np
import pandas as pd

import os
import sys
from time import time
from glob import glob
from extractPatches import patch_extractor
# from label_patch import diffusionMap
# from patch_normalizer import normalizer
from lib.utils import mark_contours, configuration, run


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

patch_dir = 'CSHL_patch_samples/'
cell_dir = os.environ['ROOT_DIR'] + args.filename + '_features/'
if not os.path.exists(cell_dir):
    os.mkdir(cell_dir)
cell_dir = cell_dir + stack + '/'
if not os.path.exists(cell_dir):
    os.mkdir(cell_dir)
    # os.mkdir(cell_dir+'Images/')
    # os.mkdir(cell_dir+'Properties/')

cell_dir = args.filename + '_features/' + stack + '/'
# t0=time()

# assert structure

for state in [state]:
    t1 = time()
    # savepath = cell_dir + 'Properties/' + structure + '/'
    # img_path = cell_dir + 'Images/' + structure + '/'
    savepath = cell_dir + structure + '/'
    pkl_out_file = savepath + stack + '_' + structure + '_' + state + '.pkl'
    # img_out_file = img_path+stack+'_'+structure+'_'+state+'_images.pkl'
    if os.path.exists(os.environ['ROOT_DIR'] + pkl_out_file):
        print(structure + '_' + state + ' ALREADY EXIST')
        continue
    else:
        if not os.path.exists(os.environ['ROOT_DIR'] + savepath):
            os.mkdir(os.environ['ROOT_DIR'] + savepath)
            # os.mkdir(img_path)

    if structure == '7nn':
        structure = '7n'

    if state == 'positive':
        setup_download_from_s3(patch_dir + structure)
        patches = [dir for dir in glob(os.environ['ROOT_DIR'] + patch_dir + structure + '/*')]
    else:
        setup_download_from_s3(patch_dir + structure + '_surround_500um_noclass')
        patches = [dir for dir in
                   glob(os.environ['ROOT_DIR'] + patch_dir + structure + '_surround_500um_noclass/*')]

    features = []

    n_choose = min(len(patches), 1000)
    indices_choose = np.random.choice(range(len(patches)), n_choose, replace=False)
    patches = np.array(patches)
    patches = patches[indices_choose]

    count = len(features)
    print(structure + '_' + state, count)
    #         cells.to_pickle(pkl_out_file)
    #         images.to_pickle(img_out_file)
    pickle.dump(features, open(os.environ['ROOT_DIR'] + pkl_out_file, 'wb'))
    setup_upload_from_s3(pkl_out_file, recursive=False)
    # s3_directory = 's3://mousebrainatlas-data/CSHL_cells_dm/'+stack+'/'+structure+'/'
    # run('aws s3 cp {0} {1}'.format(pkl_out_file,s3_directory))
    print(structure + '_' + state + ' finished in %5.1f seconds' % (time() - t1))

