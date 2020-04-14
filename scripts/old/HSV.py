import argparse

parser = argparse.ArgumentParser()
parser.add_argument("stack", type=str, help="The name of the stack")
parser.add_argument("structure", type=str, help="The nth group of structures")
parser.add_argument("yaml", type=str, help="Path to Yaml file with parameters")
args = parser.parse_args()
stack = args.stack
struc = args.structure

import cv2
import numpy as np
import pandas as pd
import pickle
import xgboost as xgb
from matplotlib import pyplot as plt
import skimage
import os
import sys
from time import time
from extractPatches import patch_extractor
from lib.utils import configuration


def CDF(x):
    x=np.sort(x)
    size=x.shape[0]
    y=np.arange(0,size)/size
    return x,y

def features_extractor(patch,params):
    extractor=patch_extractor(patch,params)
    tile=patch #cv2.imread(patch,0)
    if params['preprocessing']['polarity']==-1:
        tile = 255-tile
    min_std=params['preprocessing']['min_std']
    _std = np.std(tile.flatten())

    extracted = []
    if _std < min_std:
        extracted.append([0] * 201)
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
            ten = [x[np.argmin(np.absolute(y-0.1*(j+1)))] for j in range(10)]
            extracted.extend(ten)
        extracted.extend([cells.shape[0]/100])
    return extracted

def image_generator(structure, savepath, cell_dir, param, params, num_round, half_size,\
                    contours_grouped, raw_images_root, section_to_filename, all_patch_locations):
    t1 = time()
    subpath = savepath + structure + '/'
    if not os.path.exists(subpath):
        os.mkdir(subpath)

    fp = []
    fp.append(cell_dir + structure + '/MD589_' + structure + '_positive.pkl')
    fp.append(cell_dir + structure + '/MD589_' + structure + '_negative.pkl')
    features = []
    labels = []
    for state in range(2):
        clouds = pickle.load(open(fp[state], 'rb'))
        features.extend(np.array(clouds))
        labels.extend([1 - state] * len(clouds))
    features = np.array(features)
    labels = np.array(labels)
    X_train = features
    y_train = labels
    dtrain = xgb.DMatrix(X_train, label=y_train)
    bst = xgb.train(param, dtrain, num_round, verbose_eval=False)

    if structure == '7nn':
        structure = '7n'

    negative = structure + '_surround_500um_noclass'

    polygons = [(contour['section'], contour['vertices']) \
                for contour_id, contour in contours_grouped.get_group(structure).iterrows()]
    count = 0
    for contour_id, contour in polygons:
        section = contour_id
        if section not in all_patch_locations[structure].keys():
            continue
        polygon = contour.copy()
        img = cv2.imread(raw_images_root + section_to_filename[section] + '_prep2_lossless_gray.tif', 2)
        m, n = img.shape
        [left, right, up, down] = [int(max(min(all_patch_locations[negative][section][:, 0]) - half_size, 0)),
                                   int(min(np.ceil(max(all_patch_locations[negative][section][:, 0]) + half_size),
                                           n - 1)),
                                   int(max(min(all_patch_locations[negative][section][:, 1]) - half_size, 0)),
                                   int(min(np.ceil(max(all_patch_locations[negative][section][:, 1]) + half_size),
                                           m - 1))]

        xs, ys = np.meshgrid(np.arange(left + half_size, right - half_size + 1, half_size * 2),
                             np.arange(up + half_size, down - half_size + 1, half_size * 2), indexing='xy')
        locations = np.c_[xs.flat, ys.flat]
        inside = all_patch_locations[structure][section]
        all_rows = locations.view([('', locations.dtype)] * locations.shape[1])
        inside_rows = inside.view([('', inside.dtype)] * inside.shape[1])
        outside = np.setdiff1d(all_rows, inside_rows).view(locations.dtype).reshape(-1, locations.shape[1])
        windows = []
        windows.append(inside)
        windows.append(outside)
        polygon[:, 0] = polygon[:, 0] - left
        polygon[:, 1] = polygon[:, 1] - up

        hsv = np.zeros([down - up + 1, right - left + 1, 3])
        hsv[:, :, 2] = 1
        for state in range(2):
            for index in range(len(windows[state])):
                try:
                    x = int(float(windows[state][index][0]))
                    y = int(float(windows[state][index][1]))
                    patch = img[y - half_size:y + half_size, x - half_size:x + half_size]
                    extracted = features_extractor(patch, params)
                    xtest = xgb.DMatrix(extracted)
                    score = bst.predict(xtest, output_margin=True, ntree_limit=bst.best_ntree_limit)
                    value_img = patch / 255
                    hsv[y - half_size - up:y + half_size - up, x - half_size - left:x + half_size - left, 2] = value_img
                    satua_img = np.zeros_like(value_img) + score
                    origin = hsv[y - half_size - up:y + half_size - up, x - half_size - left:x + half_size - left, 1]
                    comp = np.absolute(origin) - np.absolute(satua_img)
                    hsv[y - half_size - up:y + half_size - up, x - half_size - left:x + half_size - left,
                    1] = origin * (comp > 0) + satua_img * (comp < 0)
                except:
                    continue
        hsv[:, :, 0] = (hsv[:, :, 1] < 0) * 0.66 + (hsv[:, :, 1] > 0) * 1.0
        hsv[:, :, 1] = np.absolute(hsv[:, :, 1])
        hsv[:, :, 1] = (hsv[:, :, 1] - hsv[:, :, 1].min()) / (hsv[:, :, 1].max() - hsv[:, :, 1].min()) * 0.8 + 0.2
        rgb = skimage.color.hsv2rgb(hsv)
        rgb = rgb * 255
        rgb = rgb.astype(np.uint8)
        com = cv2.polylines(rgb.copy(), [polygon.astype(np.int32)], True, [0, 255, 0], 15, lineType=8)
        filename = subpath + structure + '_' + str(section) + '.tif'
        cv2.imwrite(filename, com)
        if count%5==0:
            print(structure + '_' + str(count), '/', len(polygons))
        count += 1

    print(structure + ' finished in %5.1f seconds' % (time() - t1))

fp = os.path.join(os.environ['ROOT_DIR'], 'CSHL_data_processed', stack, stack + '_sorted_filenames.txt')
with open(fp, 'r') as f:
    fn_idx_tuples = [line.strip().split() for line in f.readlines()]
    section_to_filename = {int(idx): fn for fn, idx in fn_idx_tuples}

fname = os.path.join(os.environ['ROOT_DIR'], 'CSHL_data_processed', stack, 'All_patch_locations.pkl')
all_patch_locations = pickle.load(open(fname, 'rb'), encoding='latin1')

fname = os.path.join(os.environ['ROOT_DIR'], 'CSHL_data_processed', stack, 'Annotation.npy')
annotation = np.load(fname, allow_pickle = True, encoding='latin1')
contours = pd.DataFrame(annotation)
contours = contours.rename(columns={0:"name", 1:"section", 2:"vertices"})
contours_grouped = contours.groupby('name')

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

cell_dir = os.environ['ROOT_DIR'] + 'CSHL_patches_features/MD589/'
raw_images_root = os.environ['ROOT_DIR']+'/CSHL_data_processed/'+stack+'/'+stack+'_prep2_lossless_gray/'
savepath = os.environ['ROOT_DIR']+'/CSHL_hsv/'
if not os.path.exists(savepath):
    os.mkdir(savepath)
savepath = savepath+stack+'/'
if not os.path.exists(savepath):
    os.mkdir(savepath)

resol = 0.46
half_size = 112

image_generator(struc, savepath, cell_dir, param, params, num_round, half_size,\
                    contours_grouped, raw_images_root, section_to_filename, all_patch_locations)
