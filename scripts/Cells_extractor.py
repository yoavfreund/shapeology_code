import argparse

parser = argparse.ArgumentParser()
parser.add_argument("stack", type=str, default='MD594', help="The name of the brain")
parser.add_argument("section", type=int, default=256, help="The section number")
args = parser.parse_args()
stack = args.stack
section = args.section

import cv2
from skimage import io
import numpy as np
import pickle
import os
import sys
import shutil
from scipy import stats
from time import time
sys.path.append(os.environ['REPO_DIR'])
from extractPatches import patch_extractor
from lib.utils import configuration, run
from lib.shape_utils import *


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

yamlfile=os.environ['REPO_DIR'] + 'shape_params-aws.yaml'
params=configuration(yamlfile).getParams()
extractor = patch_extractor(params)
min_std = params['preprocessing']['min_std']
size_thresholds = params['normalization']['size_thresholds']
raw_images_root = os.path.join('CSHL_data_processed/', stack, 'neuroglancer_input/')

t0 = time()
img_fn = raw_images_root + str(section) + '.tif'
setup_download_from_s3(img_fn, recursive=False)
img = io.imread(os.environ['ROOT_DIR']+img_fn)
tile = 255-img.copy()
_std = np.std(tile.flatten())
if _std < min_std:
    print('image', section,'std=',_std, 'too blank, skipping')
else:
    Stats = extractor.segment_cells(tile)
    extracted = extractor.extract_blobs(Stats, tile, dm=False)
    patchesBySize={size:[] for size in size_thresholds} # storage for normalized patches
    patchIndex={size:[] for size in size_thresholds}

    #collect patches by size
    for i in range(len(extracted)):
        properties=extracted[i]
        padded_size=properties['padded_size']
        patch = properties['padded_patch']
        if patch is None:
            continue
        patchesBySize[padded_size].append(patch)

    for size in size_thresholds:
        pics=pack_pics(patchesBySize[size])
        pics=pics.astype(np.float16)
        order = np.random.permutation(pics.shape[0])
        pics = pics[order, :, :]

        fn = stack+'/cells/'+'cells-'+str(size)+'/'
        if not os.path.exists(os.environ['ROOT_DIR']+fn):
            os.makedirs(os.environ['ROOT_DIR']+fn)
        fn += str(section)+'.bin'
        pics.tofile(os.environ['ROOT_DIR']+fn)
        print(os.path.getsize(os.environ['ROOT_DIR']+fn))
        # setup_upload_from_s3(fn, recursive=False)
        # os.remove(os.environ['ROOT_DIR'] + fn)
    save_fp = os.environ['ROOT_DIR'] + stack+'/cells/'
    setup_upload_from_s3(save_fp)
    shutil.rmtree(save_fp)

    del patchesBySize
    print(section, 'finished in', time()-t0, 'seconds')
os.remove(os.environ['ROOT_DIR']+img_fn)

