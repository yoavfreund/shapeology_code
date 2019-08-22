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
import os
import sys
import shutil
from time import time
from extractPatches import patch_extractor
from lib.utils import configuration, run
from matplotlib.path import Path
from shapely.geometry import Polygon


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

def features_extractor(tile,state,params,extractor, threshold):
    contours, _ = cv2.findContours(tile.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if state=='positive':
        if len(contours)==1:
            object_area = cv2.contourArea(contours[0])
        else:
            areas=[]
            for j in range(len(contours)):
                areas.extend([cv2.contourArea(contours[j])])
            object_area = max(areas)
    else:
        if len(contours)==2:
            object_area = cv2.contourArea(contours[0])-cv2.contourArea(contours[1])
        else:
            areas=[]
            for j in range(len(contours)):
                areas.extend([cv2.contourArea(contours[j])])
            areas=np.sort(areas)
            object_area = areas[-1]-areas[-2]

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
            ten = [y[np.argmin(np.absolute(x - threshold[k][j]))] for j in range(99)]
            extracted.extend(ten)
        extracted.extend([cells.shape[0]/object_area*224*224])
    return extracted

def image_generator(section, savepath, features_fn, cell_dir, cell2_dir, param, params, num_round, step_size,\
                    contours_grouped, raw_images_root, section_to_filename, all_structures, thresholds, valid_sections):
    t1 = time()
    img_fn = raw_images_root + section_to_filename[section] + '_prep2_lossless_gray.tif'
    setup_download_from_s3(img_fn, recursive=False)
    img = cv2.imread(os.environ['ROOT_DIR']+img_fn, 2)
    m, n = img.shape
    margin = 200/0.46
    extractor = patch_extractor(params)

    polygons = [(contour['name'], contour['vertices']) \
                for contour_id, contour in contours_grouped.get_group(section).iterrows()]

    # grid_fn = features_fn + str(section) + '.pkl'
    # try:
    #     setup_download_from_s3(grid_fn, recursive=False)
    #     grid_features = pickle.load(open(os.environ['ROOT_DIR']+grid_fn,'rb'))
    #     NotUpload = False
    # except:
    #     grid_features = {}
    #     NotUpload = True

    count = 0
    Scores = {}
    for contour_id, contour in polygons:
        structure = contour_id
        if structure not in all_structures:
            continue
        polygon = contour.copy()
        Scores[structure] = {}

        if structure == '7n':
            structure = '7nn'

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

        [left, right, up, down] = [int(max(min(polygon[:, 0]) - margin, 0)),
                                   int(min(np.ceil(max(polygon[:, 0]) + margin), n - 1)),
                                   int(max(min(polygon[:, 1]) - margin, 0)),
                                   int(min(np.ceil(max(polygon[:, 1]) + margin), m - 1))]
        xs, ys = np.meshgrid(np.arange(left, right + 1), np.arange(up, down + 1), indexing='xy')
        locations = np.c_[xs.flat, ys.flat]

        path = Path(polygon)
        indices_inside = np.where(path.contains_points(locations))[0]
        indices_in = locations[indices_inside]
        x_raw = indices_in[:, 0] - left
        y_raw = indices_in[:, 1] - up
        mask = np.zeros((down - up + 1, right - left + 1))
        for i in range(len(indices_in)):
            mask[y_raw[i], x_raw[i]] = 1
        mask = mask.astype(np.uint8)

        Scores[structure][str(section)+'_positive'] = {}
        x_shift = []
        y_shift = []
        z_shift = []
        for i in range(-10,11):
            try:
                nleft = int(max(left+i*step_size, 0))
                nright = int(min(right+i*step_size, n-1))
                patch = img[up:down + 1, nleft:nright + 1] * mask[:,int(nleft -left-i*step_size):\
                                                             int(nleft -left-i*step_size)+nright - nleft + 1]
                # grid_index = str(section) + '_' + structure + '_' + 'postive_x_'+str(i)
                # if grid_index in grid_features.keys():
                #     extracted = grid_features[grid_index]
                # else:
                extracted = features_extractor(patch,'positive',params,extractor, thresholds)
                    # grid_features[grid_index] = extracted

                xtest = xgb.DMatrix(extracted)
                score = bst.predict(xtest, output_margin=True, ntree_limit=bst.best_ntree_limit)
                x_shift.append(score)
            except:
                x_shift.append(0)

            try:
                nup = int(max(up+i*step_size, 0))
                ndown = int(min(down+i*step_size, n-1))
                patch = img[nup:ndown + 1, left:right + 1] * mask[int(nup -up-i*step_size):\
                                                             int(nup -up-i*step_size)+ndown - nup + 1, :]
                # grid_index = str(section) + '_' + structure + '_' + 'postive_y_' + str(i)
                # if grid_index in grid_features.keys():
                #     extracted = grid_features[grid_index]
                # else:
                extracted = features_extractor(patch, 'positive', params, extractor, thresholds)
                    # grid_features[grid_index] = extracted

                xtest = xgb.DMatrix(extracted)
                score = bst.predict(xtest, output_margin=True, ntree_limit=bst.best_ntree_limit)
                y_shift.append(score)
            except:
                y_shift.append(0)

            loc_z = section + i*2
            if loc_z in valid_sections:
                sec_fn = raw_images_root + section_to_filename[loc_z] + '_prep2_lossless_gray.tif'
                setup_download_from_s3(sec_fn, recursive=False)
                sec = cv2.imread(os.environ['ROOT_DIR'] + sec_fn, 2)
                try:
                    patch = sec[up:down + 1, left:right + 1] * mask
                    extracted = features_extractor(patch, 'positive', params, extractor, thresholds)
                    xtest = xgb.DMatrix(extracted)
                    score = bst.predict(xtest, output_margin=True, ntree_limit=bst.best_ntree_limit)
                    z_shift.append(score)
                except:
                    z_shift.append(0)
            else:
                z_shift.append(0)

        Scores[structure][str(section) + '_positive']['x'] = x_shift
        Scores[structure][str(section) + '_positive']['y'] = y_shift
        Scores[structure][str(section) + '_positive']['z'] = z_shift

        surround = Polygon(polygon).buffer(margin, resolution=2)
        path = Path(list(surround.exterior.coords))

        indices_sur = np.where(path.contains_points(locations))[0]
        indices_outside = np.setdiff1d(indices_sur, indices_inside)
        indices_out = locations[indices_outside]

        x_raw = indices_out[:, 0] - left
        y_raw = indices_out[:, 1] - up
        mask = np.zeros((down - up + 1, right - left + 1))
        for i in range(len(indices_out)):
            mask[y_raw[i], x_raw[i]] = 1
        mask = mask.astype(np.uint8)

        Scores[structure][str(section) + '_negative'] = {}
        x_shift = []
        y_shift = []
        z_shift = []
        for i in range(-10, 11):
            try:
                nleft = int(max(left + i * step_size, 0))
                nright = int(min(right + i * step_size, n - 1))
                patch = img[up:down + 1, nleft:nright + 1] * mask[:, int(nleft - left - i * step_size): \
                                                                int(nleft - left - i * step_size) + nright - nleft + 1]
                # grid_index = str(section) + '_' + structure + '_' + 'negative_x_' + str(i)
                # if grid_index in grid_features.keys():
                #     extracted = grid_features[grid_index]
                # else:
                extracted = features_extractor(patch, 'negative', params, extractor, thresholds)
                    # grid_features[grid_index] = extracted

                xtest = xgb.DMatrix(extracted)
                score = bst.predict(xtest, output_margin=True, ntree_limit=bst.best_ntree_limit)
                x_shift.append(score)
            except:
                x_shift.append(0)

            try:
                nup = int(max(up + i * step_size, 0))
                ndown = int(min(down + i * step_size, n - 1))
                patch = img[nup:ndown + 1, left:right + 1] * mask[int(nup - up - i * step_size): \
                                                                  int(nup - up - i * step_size) + ndown - nup + 1, :]
                # grid_index = str(section) + '_' + structure + '_' + 'negative_y_' + str(i)
                # if grid_index in grid_features.keys():
                #     extracted = grid_features[grid_index]
                # else:
                extracted = features_extractor(patch, 'negative', params, extractor, thresholds)
                #     grid_features[grid_index] = extracted

                xtest = xgb.DMatrix(extracted)
                score = bst.predict(xtest, output_margin=True, ntree_limit=bst.best_ntree_limit)
                y_shift.append(score)
            except:
                y_shift.append(0)

            loc_z = section + i * 2
            if loc_z in valid_sections:
                sec_fn = raw_images_root + section_to_filename[loc_z] + '_prep2_lossless_gray.tif'
                # setup_download_from_s3(sec_fn, recursive=False)
                sec = cv2.imread(os.environ['ROOT_DIR'] + sec_fn, 2)
                # os.remove(os.environ['ROOT_DIR'] + sec_fn)
                try:
                    patch = sec[up:down + 1, left:right + 1] * mask
                    extracted = features_extractor(patch, 'positive', params, extractor, thresholds)
                    xtest = xgb.DMatrix(extracted)
                    score = bst.predict(xtest, output_margin=True, ntree_limit=bst.best_ntree_limit)
                    z_shift.append(score)
                except:
                    z_shift.append(0)
            else:
                z_shift.append(0)

        Scores[structure][str(section) + '_negative']['x'] = x_shift
        Scores[structure][str(section) + '_negative']['y'] = y_shift
        Scores[structure][str(section) + '_negative']['z'] = z_shift


        count += 1
        print(section, structure, count, '/', len(polygons))

    # if NotUpload:
    #     pickle.dump(grid_features, open(os.environ['ROOT_DIR'] + grid_fn, 'wb'))
    #     setup_upload_from_s3(grid_fn, recursive=False)
    filename = savepath + str(section) + '.pkl'
    pickle.dump(Scores, open(os.environ['ROOT_DIR'] + filename, 'wb'))
    setup_upload_from_s3(filename, recursive=False)
    shutil.rmtree(raw_images_root)
    # os.remove(os.environ['ROOT_DIR']+img_fn)
    print(str(section) + ' finished in %5.1f seconds' % (time() - t1))


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
valid_sections = np.sort(contours['section'].unique())

fn = 'CSHL_data_processed/MD589/ThresholdsV2.pkl'
setup_download_from_s3(fn, recursive=False)
thresholds = pickle.load(open(os.environ['ROOT_DIR']+fn,'rb'))
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
features_fn = 'CSHL_region_features/'
if not os.path.exists(os.environ['ROOT_DIR']+features_fn):
    os.mkdir(os.environ['ROOT_DIR']+features_fn)
features_fn = features_fn+stack+'/'
if not os.path.exists(os.environ['ROOT_DIR']+features_fn):
    os.mkdir(os.environ['ROOT_DIR']+features_fn)

savepath = 'CSHL_shift/'
if not os.path.exists(os.environ['ROOT_DIR']+savepath):
    os.mkdir(os.environ['ROOT_DIR']+savepath)
savepath = savepath+stack+'/'
if not os.path.exists(os.environ['ROOT_DIR']+savepath):
    os.mkdir(os.environ['ROOT_DIR']+savepath)

resol = 0.46
step_size = int(30/resol)

paired_structures = ['5N', '6N', '7N', '7n', 'Amb', 'LC', 'LRt', 'Pn', 'Tz', 'VLL', 'RMC', \
                     'SNC', 'SNR', '3N', '4N', 'Sp5I', 'Sp5O', 'Sp5C', 'PBG', '10N', 'VCA', 'VCP', 'DC']
singular_structures = ['AP', '12N', 'RtTg', 'SC', 'IC']

all_structures = paired_structures + singular_structures

image_generator(section, savepath, features_fn, cell_dir, cell2_dir, param, params, num_round, step_size,\
                    contours_grouped, raw_images_root, section_to_filename, all_structures, thresholds, valid_sections)
