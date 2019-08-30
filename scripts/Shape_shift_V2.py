import argparse

parser = argparse.ArgumentParser()
parser.add_argument("stack", type=str, help="The name of the stack")
parser.add_argument("section", type=int, help="The section number")
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
import sqlite3
import shutil
from time import time
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

def features_to_score(features, thresholds, bst, object_area):
    extracted = []
    for k in range(features.shape[1]):
        x, y = CDF(features[:,k])
        ten = [y[np.argmin(np.absolute(x - thresholds[k][j]))] for j in range(99)]
        extracted.extend(ten)
    extracted.extend([features.shape[0]/object_area*224*224])

    xtest = xgb.DMatrix(extracted)
    score = bst.predict(xtest, output_margin=True, ntree_limit=bst.best_ntree_limit)
    return score


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

cell_dir = os.environ['ROOT_DIR'] + 'CSHL_patch_samples_features_V2/MD589/'
cell2_dir = os.environ['ROOT_DIR'] + 'CSHL_patch_samples_features_V2/MD585/'

savepath = 'CSHL_shift_20um/'
if not os.path.exists(os.environ['ROOT_DIR']+savepath):
    os.mkdir(os.environ['ROOT_DIR']+savepath)
savepath = savepath+stack+'/'
if not os.path.exists(os.environ['ROOT_DIR']+savepath):
    os.mkdir(os.environ['ROOT_DIR']+savepath)

resol = 0.46
step_size = int(20/resol)

paired_structures = ['5N', '6N', '7N', '7n', 'Amb', 'LC', 'LRt', 'Pn', 'Tz', 'VLL', 'RMC', \
                     'SNC', 'SNR', '3N', '4N', 'Sp5I', 'Sp5O', 'Sp5C', 'PBG', '10N', 'VCA', 'VCP', 'DC']
singular_structures = ['AP', '12N', 'RtTg', 'SC', 'IC']

all_structures = paired_structures + singular_structures

t1 = time()
margin = 200/0.46

polygons = [(contour['name'], contour['vertices']) \
            for contour_id, contour in contours_grouped.get_group(section).iterrows()]

db_dir = 'CSHL_databases/' + stack + '/'
db_fp = db_dir + str(section) + '.db'
setup_download_from_s3(db_fp, recursive=False)

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

    [left, right, up, down] = [int(max(min(polygon[:, 0]) - margin - 10 * step_size, 0)),
                               int(np.ceil(max(polygon[:, 0]) + margin + 10 * step_size)),
                               int(max(min(polygon[:, 1]) - margin - 10 * step_size, 0)),
                               int(np.ceil(max(polygon[:, 1]) + margin + 10 * step_size))]
    conn = sqlite3.connect(os.environ['ROOT_DIR'] + db_fp)
    cur = conn.cursor()
    raws = cur.execute('SELECT * FROM features WHERE x>=? AND x<=? AND y>=? AND y<=?', (left, right, up, down))
    info = np.array(list(raws))
    locations = info[:, 1:3]
    features = info[:, 3:]

    Scores[structure][str(section) + '_positive'] = {}
    Scores[structure][str(section) + '_negative'] = {}
    inside_area = Polygon(polygon).area
    outside_area = Polygon(polygon).buffer(margin, resolution=2).area - inside_area
    x_shift_positive = []
    x_shift_negative = []
    y_shift_positive = []
    y_shift_negative = []
    z_shift_positive = []
    z_shift_negative = []
    for i in range(-20, 21):
        region = polygon.copy()
        region[:, 0] += i * step_size
        path = Path(region)
        indices_inside = np.where(path.contains_points(locations))[0]
        features_inside = features[indices_inside]
        if features_inside.shape[0]:
            score = features_to_score(features_inside, thresholds, bst, inside_area)
            x_shift_positive.append(score)
        else:
            x_shift_positive.append(0)

        surround = Polygon(region).buffer(margin, resolution=2)
        path = Path(list(surround.exterior.coords))
        indices_sur = np.where(path.contains_points(locations))[0]
        indices_outside = np.setdiff1d(indices_sur, indices_inside)
        features_outside = features[indices_outside]
        if features_outside.shape[0]:
            score = features_to_score(features_outside, thresholds, bst, outside_area)
            x_shift_negative.append(score)
        else:
            x_shift_negative.append(0)

        region = polygon.copy()
        region[:, 1] += i * step_size
        path = Path(region)
        indices_inside = np.where(path.contains_points(locations))[0]
        features_inside = features[indices_inside]
        if features_inside.shape[0]:
            score = features_to_score(features_inside, thresholds, bst, inside_area)
            y_shift_positive.append(score)
        else:
            y_shift_positive.append(0)

        surround = Polygon(region).buffer(margin, resolution=2)
        path = Path(list(surround.exterior.coords))
        indices_sur = np.where(path.contains_points(locations))[0]
        indices_outside = np.setdiff1d(indices_sur, indices_inside)
        features_outside = features[indices_outside]
        if features_outside.shape[0]:
            score = features_to_score(features_outside, thresholds, bst, outside_area)
            y_shift_negative.append(score)
        else:
            y_shift_negative.append(0)

    conn.close()
    [left, right, up, down] = [int(max(min(polygon[:, 0]) - margin, 0)),
                               int(np.ceil(max(polygon[:, 0]) + margin)),
                               int(max(min(polygon[:, 1]) - margin, 0)),
                               int(np.ceil(max(polygon[:, 1]) + margin))]
    for i in range(-20, 21):
        loc_z = section + i * 2
        if loc_z in valid_sections:
            sec_fp = db_dir + str(loc_z) + '.db'
            setup_download_from_s3(sec_fp, recursive=False)
            conn = sqlite3.connect(os.environ['ROOT_DIR'] + sec_fp)
            cur = conn.cursor()
            try:
                raws = cur.execute('SELECT * FROM features WHERE x>=? AND x<=? AND y>=? AND y<=?', (left, right, up, down))
                info = np.array(list(raws))
                locations = info[:, 1:3]
                features = info[:, 3:]

                path = Path(polygon)
                indices_inside = np.where(path.contains_points(locations))[0]
                features_inside = features[indices_inside]

                score = features_to_score(features_inside, thresholds, bst, inside_area)
                z_shift_positive.append(score)

                surround = Polygon(polygon).buffer(margin, resolution=2)
                path = Path(list(surround.exterior.coords))
                indices_sur = np.where(path.contains_points(locations))[0]
                indices_outside = np.setdiff1d(indices_sur, indices_inside)
                features_outside = features[indices_outside]

                score = features_to_score(features_outside, thresholds, bst, outside_area)
                z_shift_negative.append(score)
            except:
                z_shift_positive.append(0)
                z_shift_negative.append(0)
        else:
            z_shift_positive.append(0)
            z_shift_negative.append(0)
        conn.close()

    Scores[structure][str(section) + '_positive']['x'] = x_shift_positive
    Scores[structure][str(section) + '_positive']['y'] = y_shift_positive
    Scores[structure][str(section) + '_positive']['z'] = z_shift_positive

    Scores[structure][str(section) + '_negative']['x'] = x_shift_negative
    Scores[structure][str(section) + '_negative']['y'] = y_shift_negative
    Scores[structure][str(section) + '_negative']['z'] = z_shift_negative

    count += 1
    print(section, structure, count, '/', len(polygons))

shutil.rmtree(os.environ['ROOT_DIR']+db_dir)
filename = savepath + str(section) + '.pkl'
pickle.dump(Scores, open(os.environ['ROOT_DIR'] + filename, 'wb'))
setup_upload_from_s3(filename, recursive=False)
# os.remove(os.environ['ROOT_DIR']+img_fn)
print(str(section) + ' finished in %5.1f seconds' % (time() - t1))
