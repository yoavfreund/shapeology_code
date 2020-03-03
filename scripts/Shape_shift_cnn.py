import argparse

parser = argparse.ArgumentParser()
parser.add_argument("stack", type=str, default='MD594', help="The name of the stack")
parser.add_argument("section", type=int, default=110, help="The section number")
args = parser.parse_args()
stack = args.stack
section = args.section

import cv2
import numpy as np
import pandas as pd
import pickle
import skimage
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import sys
import shutil
from scipy import stats
from time import time
from glob import glob
sys.path.append(os.environ['REPO_DIR'])
from lib.utils import configuration, run
from matplotlib.path import Path
from shapely.geometry import Polygon
import mxnet as mx

from collections import namedtuple
Batch = namedtuple('Batch', ['data'])

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


fname = os.path.join('CSHL_data_processed', stack, 'Annotation.npy')
setup_download_from_s3(fname, recursive=False)
annotation = np.load(os.environ['ROOT_DIR']+fname, allow_pickle = True, encoding='latin1')
contours = pd.DataFrame(annotation)
contours = contours.rename(columns={0:"name", 1:"section", 2:"vertices"})
contours_grouped = contours.groupby('section')
contours_struc = contours.groupby('name')
valid_sections = np.sort(contours['section'].unique())

# features_fn = 'CSHL_region_features/'
# if not os.path.exists(os.environ['ROOT_DIR']+features_fn):
#     os.mkdir(os.environ['ROOT_DIR']+features_fn)
# features_fn = features_fn+stack+'/'
# if not os.path.exists(os.environ['ROOT_DIR']+features_fn):
#     os.mkdir(os.environ['ROOT_DIR']+features_fn)

savepath = 'CSHL_shift_cnn/'
if not os.path.exists(os.environ['ROOT_DIR']+savepath):
    os.mkdir(os.environ['ROOT_DIR']+savepath)
savepath = savepath+stack+'/'
if not os.path.exists(os.environ['ROOT_DIR']+savepath):
    os.mkdir(os.environ['ROOT_DIR']+savepath)

resol = 0.46
step_size = int(20/resol)
window_size = 224
half = 20

paired_structures = ['5N', '6N', '7N', '7n', 'Amb', 'LC', 'LRt', 'Pn', 'Tz', 'VLL', 'RMC', \
                     'SNC', 'SNR', '3N', '4N', 'Sp5I', 'Sp5O', 'Sp5C', 'PBG', '10N', 'VCA', 'VCP', 'DC']
singular_structures = ['AP', '12N', 'RtTg', 'SC', 'IC']

all_structures = paired_structures + singular_structures

MXNET_ROOTDIR = 'mxnet_models'
model_dir_name = 'inception-bn-blue-softmax'
model_name = 'inception-bn-blue-softmax'
setup_download_from_s3(os.path.join(MXNET_ROOTDIR, model_dir_name, 'mean_224.npy'), recursive=False)
mean_img = np.load(os.path.join(os.environ['ROOT_DIR'], MXNET_ROOTDIR, model_dir_name, 'mean_224.npy'))
model_prefix = os.path.join(MXNET_ROOTDIR, model_dir_name, model_name)

raw_images_root = 'CSHL_data_processed/'+stack+'/'+stack+'_prep2_lossless_gray/'
img_fn = raw_images_root + section_to_filename[section] + '_prep2_lossless_gray.tif'
setup_download_from_s3(img_fn, recursive=False)
img = cv2.imread(os.environ['ROOT_DIR']+img_fn, 2)
m, n = img.shape
margin = 200/0.46

polygons = [(contour['name'], contour['vertices']) \
                for contour_id, contour in contours_grouped.get_group(section).iterrows()]

count = 0
t1 = time()
Scores = {}
for contour_id, contour in polygons:
    structure = contour_id
    if structure not in all_structures:
        continue
    polygon = contour.copy()
    Scores[structure] = {}
    while os.path.exists(os.path.join(os.environ['ROOT_DIR'], model_prefix + '_' + structure + '-symbol.json'))==0:
        setup_download_from_s3(model_prefix + '_' + structure + '-symbol.json', recursive=False)
        setup_download_from_s3(model_prefix + '_' + structure + '-0045.params', recursive=False)

    model, arg_params, aux_params = mx.model.load_checkpoint(os.path.join(os.environ['ROOT_DIR'], model_prefix + '_' + structure), 45)


    [left, right, up, down] = [int(max(min(polygon[:, 0]) - margin - half * step_size, 0)),
                               int(min(np.ceil(max(polygon[:, 0]) + margin + half * step_size),n-1)),
                               int(max(min(polygon[:, 1]) - margin - half * step_size, 0)),
                               int(min(np.ceil(max(polygon[:, 1]) + margin + half * step_size),m-1))]
    xs, ys = np.meshgrid(np.arange(left, right-window_size, window_size//2), np.arange(up, down-window_size, window_size//2), indexing='xy')
    windows = np.c_[xs.flat, ys.flat] + window_size//2

    patches = np.array([img[wy-window_size//2:wy+window_size//2, wx-window_size//2:wx+window_size//2] for wx,wy in windows])
    batch_size = patches.shape[0]
    mod = mx.mod.Module(symbol=model, label_names=None, context=mx.cpu())
    mod.bind(for_training=False,
             data_shapes=[('data', (batch_size, 1, 224, 224))])
    mod.set_params(arg_params, aux_params, allow_missing=True)
    test = (patches - mean_img)[:, None, :, :]
    mod.forward(Batch([mx.nd.array(test)]))
    scores = mod.get_outputs()[0].asnumpy()[:,1]


    Scores[structure][str(section)+'_positive'] = {}
    Scores[structure][str(section) + '_negative'] = {}
    x_shift_positive = []
    x_shift_negative = []
    y_shift_positive = []
    y_shift_negative = []


    for i in range(-half, half+1):
        region = polygon.copy()
        region[:, 0] += i * step_size
        path = Path(region)
        indices_inside = np.where(path.contains_points(windows))[0]
        score = scores[indices_inside].mean()
        x_shift_positive.append(score)

        surround = Polygon(region).buffer(margin, resolution=2)
        path = Path(list(surround.exterior.coords))
        indices_sur = np.where(path.contains_points(windows))[0]
        indices_outside = np.setdiff1d(indices_sur, indices_inside)
        score = scores[indices_outside].mean()
        x_shift_negative.append(score)

        region = polygon.copy()
        region[:, 1] += i * step_size
        path = Path(region)
        indices_inside = np.where(path.contains_points(windows))[0]
        score = scores[indices_inside].mean()
        y_shift_positive.append(score)

        surround = Polygon(region).buffer(margin, resolution=2)
        path = Path(list(surround.exterior.coords))
        indices_sur = np.where(path.contains_points(windows))[0]
        indices_outside = np.setdiff1d(indices_sur, indices_inside)
        score = scores[indices_outside].mean()
        y_shift_negative.append(score)

    Scores[structure][str(section) + '_positive']['x'] = x_shift_positive
    Scores[structure][str(section) + '_positive']['y'] = y_shift_positive
    Scores[structure][str(section) + '_negative']['x'] = x_shift_negative
    Scores[structure][str(section) + '_negative']['y'] = y_shift_negative

    z_shift_positive = []
    z_shift_negative = []
    [left, right, up, down] = [int(max(min(polygon[:, 0]) - margin, 0)),
                               int(min(np.ceil(max(polygon[:, 0]) + margin), n - 1)),
                               int(max(min(polygon[:, 1]) - margin, 0)),
                               int(min(np.ceil(max(polygon[:, 1]) + margin), m - 1))]
    xs, ys = np.meshgrid(np.arange(left, right - window_size, window_size//2), np.arange(up, down - window_size, window_size//2),
                         indexing='xy')
    windows = np.c_[xs.flat, ys.flat] + window_size // 2

    for i in range(-half, half + 1):
        loc_z = section + i
        if loc_z in valid_sections:
            shutil.rmtree(os.environ['ROOT_DIR'] + raw_images_root)
            sec_fn = raw_images_root + section_to_filename[loc_z] + '_prep2_lossless_gray.tif'
            setup_download_from_s3(sec_fn, recursive=False)
            sec = cv2.imread(os.environ['ROOT_DIR'] + sec_fn, 2)
            patches = np.array([sec[wy - window_size // 2:wy + window_size // 2, wx - window_size // 2:wx + window_size // 2] for
                 wx, wy in windows])
            batch_size = patches.shape[0]
            mod = mx.mod.Module(symbol=model, label_names=None, context=mx.cpu())
            mod.bind(for_training=False,
                     data_shapes=[('data', (batch_size, 1, 224, 224))])
            mod.set_params(arg_params, aux_params, allow_missing=True)
            test = (patches - mean_img)[:, None, :, :]
            mod.forward(Batch([mx.nd.array(test)]))
            scores = mod.get_outputs()[0].asnumpy()[:, 1]

            region = polygon.copy()
            path = Path(region)
            indices_inside = np.where(path.contains_points(windows))[0]
            score = scores[indices_inside].mean()
            z_shift_positive.append(score)

            surround = Polygon(region).buffer(margin, resolution=2)
            path = Path(list(surround.exterior.coords))
            indices_sur = np.where(path.contains_points(windows))[0]
            indices_outside = np.setdiff1d(indices_sur, indices_inside)
            score = scores[indices_outside].mean()
            z_shift_negative.append(score)

        else:
            z_shift_positive.append(0)
            z_shift_negative.append(0)


    Scores[structure][str(section) + '_positive']['z'] = z_shift_positive
    Scores[structure][str(section) + '_negative']['z'] = z_shift_negative


    count += 1
    print(section, structure, count, '/', len(polygons))

# if NotUpload:
#     pickle.dump(grid_features, open(os.environ['ROOT_DIR'] + grid_fn, 'wb'))
#     setup_upload_from_s3(grid_fn, recursive=False)
filename = savepath + str(section) + '.pkl'
pickle.dump(Scores, open(os.environ['ROOT_DIR'] + filename, 'wb'))
setup_upload_from_s3(filename, recursive=False)
shutil.rmtree(os.environ['ROOT_DIR']+raw_images_root)
# os.remove(os.environ['ROOT_DIR']+img_fn)
print(str(section) + ' finished in %5.1f seconds' % (time() - t1))




