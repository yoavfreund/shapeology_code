import argparse

parser = argparse.ArgumentParser()
parser.add_argument("stack", type=str, help="The name of the stack")
parser.add_argument("section", type=int, help="The section number")
parser.add_argument("yaml", type=str, help="Path to Yaml file with parameters")
args = parser.parse_args()
stack = args.stack
section = args.section

import cv2
import numpy as np
import pandas as pd
import pickle
import xgboost as xgb
#from matplotlib import pyplot as plt
import skimage
from skimage.transform import rescale
import os
import sys
from time import time
from extractPatches import patch_extractor
from lib.utils import configuration, run


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

def features_extractor(tile, params, extractor, threshold):
    if params['preprocessing']['polarity']==-1:
        tile = 255-tile
    min_std=params['preprocessing']['min_std']
    _std = np.std(tile.flatten())

    extracted = []
    if _std < min_std:
        extracted.append([0] * 1981)
    else:
        Stats = extractor.segment_cells(tile)
        cells = extractor.extract_blobs(Stats,tile)
        cells = pd.DataFrame(cells)
        cells = cells[cells['padded_patch'].notnull()]
        cells = cells.drop(['padded_patch','left','top'],1)
        cells = np.asarray(cells)
        for k in range(len(cells)):
            cells[k][0] = cells[k][0][:10]
        origin = np.concatenate((np.array(list(cells[:,0])),cells[:,1:]),axis=1)
        for k in range(origin.shape[1]):
            x, y = CDF(origin[:,k])
            ten = [y[np.argmin(np.absolute(x-threshold[k][j]))] for j in range(99)]
            extracted.extend(ten)
        extracted.extend([cells.shape[0]])
    return extracted


fp = os.path.join('CSHL_data_processed', stack, stack + '_sorted_filenames.txt')
setup_download_from_s3(fp, recursive=False)
with open(os.environ['ROOT_DIR']+fp, 'r') as f:
    fn_idx_tuples = [line.strip().split() for line in f.readlines()]
    section_to_filename = {int(idx): fn for fn, idx in fn_idx_tuples}

fn = 'CSHL_data_processed/MD589/ThresholdsV2.pkl'
setup_download_from_s3(fn, recursive=False)
thresholds = pickle.load(open(os.environ['ROOT_DIR']+fn,'rb'))

fname = os.path.join('CSHL_data_processed', stack, 'Annotation.npy')
setup_download_from_s3(fname, recursive=False)
annotation = np.load(os.environ['ROOT_DIR']+fname, allow_pickle = True, encoding='latin1')
contours = pd.DataFrame(annotation)
contours = contours.rename(columns={0:"name", 1:"section", 2:"vertices"})
contours_grouped = contours.groupby('section')

#Parameters
param = {}
param['max_depth']= 3   # depth of tree
param['eta'] = 0.2      # shrinkage parameter
param['silent'] = 1     # not silent
param['objective'] = 'binary:logistic' #'multi:softmax'
param['nthread'] = 7 # Number of threads used
param['num_class']=1
num_round = 100

yamlfile=os.environ['REPO_DIR']+args.yaml
params=configuration(yamlfile).getParams()

cell_dir = os.environ['ROOT_DIR'] + 'CSHL_patch_samples_features_V2/MD589/'
cell2_dir = os.environ['ROOT_DIR'] + 'CSHL_patch_samples_features_V2/MD585/'
raw_images_root = 'CSHL_data_processed/'+stack+'/'+stack+'_prep2_lossless_gray/'
features_fn = 'CSHL_grid_features/'
if not os.path.exists(os.environ['ROOT_DIR']+features_fn):
    os.mkdir(os.environ['ROOT_DIR']+features_fn)
features_fn = features_fn+stack+'/'
if not os.path.exists(os.environ['ROOT_DIR']+features_fn):
    os.mkdir(os.environ['ROOT_DIR']+features_fn)

savepath = 'CSHL_scoremaps_new/'
if not os.path.exists(os.environ['ROOT_DIR']+savepath):
    os.mkdir(os.environ['ROOT_DIR']+savepath)
downsample_fp = savepath+'down32/'
if not os.path.exists(os.environ['ROOT_DIR']+downsample_fp):
    os.mkdir(os.environ['ROOT_DIR']+downsample_fp)
downsample_fp = downsample_fp+stack+'/'
if not os.path.exists(os.environ['ROOT_DIR']+downsample_fp):
    os.mkdir(os.environ['ROOT_DIR']+downsample_fp)
savepath = savepath+stack+'/'
if not os.path.exists(os.environ['ROOT_DIR']+savepath):
    os.mkdir(os.environ['ROOT_DIR']+savepath)

paired_structures = ['5N', '6N', '7N', '7nn', 'Amb', 'LC', 'LRt', 'Pn', 'Tz', 'VLL', 'RMC', \
                     'SNC', 'SNR', '3N', '4N', 'Sp5I', 'Sp5O', 'Sp5C', 'PBG', '10N', 'VCA', 'VCP', 'DC']
singular_structures = ['AP', '12N', 'RtTg', 'SC', 'IC']

all_structures = paired_structures + singular_structures
window_size = 224
stride = 112 #int(30/0.46)

img_fn = raw_images_root + section_to_filename[section] + '_prep2_lossless_gray.tif'
setup_download_from_s3(img_fn, recursive=False)
img = cv2.imread(os.environ['ROOT_DIR']+img_fn, 2)
m, n = img.shape
extractor = patch_extractor(params)

polygons = [(contour['name'], contour['vertices']) \
                for contour_id, contour in contours_grouped.get_group(section).iterrows()]

xs, ys = np.meshgrid(np.arange(0, n, stride), np.arange(0, m, stride), indexing='xy')
locations = np.c_[xs.flat, ys.flat]
valid_structure = {}
for contour_id, contour in polygons:
    valid_structure[contour_id] = contour

grid_fn = features_fn + str(section) + '.pkl'
try:
    setup_download_from_s3(grid_fn, recursive=False)
    grid_features = pickle.load(open(os.environ['ROOT_DIR']+grid_fn,'rb'))
    NotUpload = False
except:
    grid_features = {}
    NotUpload = True

# grid_features = {}
for i in range(len(locations)):
    left = locations[i][0]
    right = int(min(left + window_size, n))
    up = locations[i][1]
    down = int(min(up + window_size, m))
    tile = img[up:down, left:right]
    grid_index = str(section)+'_'+str(left)+'_'+str(up)
    try:
        if grid_index in grid_features.keys():
            extracted = grid_features[grid_index]
        else:
            extracted = features_extractor(tile, params, extractor, thresholds)
            grid_features[grid_index] = extracted
    except:
        continue
    if i % 1000 == 0:
        print(i, len(locations))

if NotUpload:
    pickle.dump(grid_features, open(os.environ['ROOT_DIR'] + grid_fn, 'wb'))
    setup_upload_from_s3(grid_fn, recursive=False)

for j in range(len(all_structures)):
    structure = all_structures[j]
    subpath = savepath + structure + '/'
    if not os.path.exists(os.environ['ROOT_DIR'] + subpath):
        os.mkdir(os.environ['ROOT_DIR'] + subpath)
    downsubpath = downsample_fp + structure + '/'
    if not os.path.exists(os.environ['ROOT_DIR'] + downsubpath):
        os.mkdir(os.environ['ROOT_DIR'] + downsubpath)

    fp = []
    fp.append(cell_dir + structure + '/MD589_' + structure + '_positive.pkl')
    fp.append(cell_dir + structure + '/MD589_' + structure + '_negative.pkl')
    X_train = []
    y_train = []
    for state in range(2):
        clouds = pickle.load(open(fp[state], 'rb'))
        X_train.extend(np.array(clouds))
        y_train.extend([1 - state] * len(clouds))

    fp = []
    fp.append(cell2_dir + structure + '/MD585_' + structure + '_positive.pkl')
    fp.append(cell2_dir + structure + '/MD585_' + structure + '_negative.pkl')
    for state in range(2):
        clouds = pickle.load(open(fp[state], 'rb'))
        X_train.extend(np.array(clouds))
        y_train.extend([1 - state] * len(clouds))
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    dtrain = xgb.DMatrix(X_train, label=y_train)
    bst = xgb.train(param, dtrain, num_round, verbose_eval=False)

    if structure == '7nn':
        structure = '7n'

    scoremap = np.zeros([m, n], dtype=np.float16)
    for x, y in locations:
        try:
            grid_index = str(section) + '_' + str(x) + '_' + str(y)
            feature_vector = grid_features[grid_index]
            xtest = xgb.DMatrix(feature_vector)
            score = bst.predict(xtest, output_margin=True, ntree_limit=bst.best_ntree_limit)
            origin = scoremap[y:y+stride, x:x+stride]
            comp = np.absolute(origin) - np.absolute(score)
            scoremap[y:y+stride, x:x+stride] = origin * (comp > 0) + score * (comp < 0)
        except:
            continue

    scoremap = 1 / (1 + np.exp(-scoremap))
    # scoremap = (scoremap - scoremap.min()) / (scoremap.max() - scoremap.min())
    gray = scoremap * 255
    gray = gray.astype(np.uint8)

    gray32 = rescale(gray, 1. / 32, anti_aliasing=True)
    gray32 = gray32 * 255
    gray32 = gray32.astype(np.uint8)

    if structure in valid_structure.keys():
        polygon = valid_structure[structure]
        rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
        com = cv2.polylines(rgb.copy(), [polygon.astype(np.int32)], True, [0, 255, 0], 15, lineType=8)

        polygon = polygon / 32
        rgb32 = cv2.cvtColor(gray32, cv2.COLOR_GRAY2RGB)
        com32 = cv2.polylines(rgb32.copy(), [polygon.astype(np.int32)], True, [0, 255, 0], 2, lineType=8)
        filename = subpath + structure + '_' + str(section) + '_contour.tif'
        filename32 = downsubpath + structure + '_' + str(section) + '_contour.tif'
    else:
        com = gray
        com32 = gray32
        filename = subpath + structure + '_' + str(section) + '.tif'
        filename32 = downsubpath + structure + '_' + str(section) + '.tif'

    cv2.imwrite(os.environ['ROOT_DIR'] + filename, com)
    setup_upload_from_s3(filename, recursive=False)

    cv2.imwrite(os.environ['ROOT_DIR'] + filename32, com32)
    setup_upload_from_s3(filename32, recursive=False)
    print(section, structure, j, '/', len(all_structures))

os.remove(os.environ['ROOT_DIR']+img_fn)
os.remove(os.environ['ROOT_DIR']+grid_fn)
