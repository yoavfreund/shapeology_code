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
from time import time
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

time_log=[]
def clock(message):
    print('%8.1f \t%s'%(time(),message))
    time_log.append((time(),message))

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
    t0 = time()
    extracted = []
    num = sum(features[:, -2])
    cdf = features[:, :-2] * features[:, -2].reshape([-1, 1])
    cdf = np.sum(cdf, axis=0) / num
    extracted.extend(cdf)
    extracted.extend([num/object_area*317 * 317])
    extracted.extend([features[:, -1].mean()])
    print('CDF',time()-t0)
    xtest = xgb.DMatrix(extracted)
    score = bst.predict(xtest, output_margin=True, ntree_limit=bst.best_ntree_limit)
    return score

import io
def adapt_array(arr):
    """
    http://stackoverflow.com/a/31312102/190597 (SoulNibbler)
    """
    out = io.BytesIO()
    np.save(out, arr)
    out.seek(0)
    return sqlite3.Binary(out.read())

def convert_array(text):
    out = io.BytesIO(text)
    out.seek(0)
    return np.load(out)

sqlite3.register_adapter(np.ndarray, adapt_array)
sqlite3.register_converter("array", convert_array)

contours = pickle.load(open('/data/Shapeology_Files/BstemAtlasDataBackup/ucsd_brain/masks/MD589/MD589_aligned_section_structure_vertices_down16.pickle','rb'))
polygons = contours[section]

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

cell_dir = os.environ['ROOT_DIR'] + 'CSHL_patch_samples_features_v1/MD589/'
cell2_dir = os.environ['ROOT_DIR'] + 'CSHL_patch_samples_features_v1/MD585/'

savepath = 'CSHL_shift_scores/'
if not os.path.exists(os.environ['ROOT_DIR']+savepath):
    os.mkdir(os.environ['ROOT_DIR']+savepath)
savepath = savepath+stack+'_cdf/'
if not os.path.exists(os.environ['ROOT_DIR']+savepath):
    os.mkdir(os.environ['ROOT_DIR']+savepath)

resol = 0.325
step_size = int(20/resol)
half = 20

paired_structures = ['5N', '6N', '7N', '7n', 'Amb', 'LC', 'LRt', 'Pn', 'Tz', 'VLL', 'RMC', \
                     'SNC', 'SNR', '3N', '4N', 'Sp5I', 'Sp5O', 'Sp5C', 'PBG', '10N', 'VCA', 'VCP', 'DC']
singular_structures = ['AP', '12N', 'RtTg', 'SC', 'IC']

all_structures = paired_structures + singular_structures

t1 = time()
margin = 200/resol


db_dir = 'CSHL_databases/' + stack + '_cdf/'
db_fp = db_dir + str(section) + '.db'
setup_download_from_s3(db_fp, recursive=False)

count = 0
Scores = {}
clock('Process Begin')
for structure in polygons.keys():
    if structure not in all_structures:
        continue

    len_max = 0
    for sec in contours.keys():
        if structure not in contours[sec].keys():
            continue
        polygon = contours[sec][structure].copy()*16*1.4154
        length = polygon[:,0].max()-polygon[:,0].min()
        width = polygon[:,1].max()-polygon[:,1].min()
        if max(length, width) > len_max:
            len_max = max(length, width)
    step_size = max(int(len_max/30), int(30/resol))

    polygon = polygons[structure].copy()*16*1.4154
    Scores[structure] = {}

    print(structure)
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

    [left, right, up, down] = [int(max(min(polygon[:, 0]) - margin - half * step_size, 0)),
                               int(np.ceil(max(polygon[:, 0]) + margin + half * step_size)),
                               int(max(min(polygon[:, 1]) - margin - half * step_size, 0)),
                               int(np.ceil(max(polygon[:, 1]) + margin + half * step_size))]
    clock('Connect database')
    t0 = time()
    conn = sqlite3.connect(os.environ['ROOT_DIR'] + db_fp, detect_types=sqlite3.PARSE_DECLTYPES)
    cur = conn.cursor()
    raws = cur.execute('SELECT * FROM features WHERE x>=? AND x<=? AND y>=? AND y<=?', (left, right, up, down))
    info = np.array(list(raws))
    locations = info[:, 1:3]
    locations = locations.astype(np.int32)
    features = info[:, 3]
    features = np.vstack(features)
    print(features.shape)
    print(structure, 'Database', time()-t0)
    clock('Finish database')

    Scores[structure][str(section) + '_positive'] = {}
    Scores[structure][str(section) + '_negative'] = {}
    inside_area = Polygon(polygon).area
    outside_area = Polygon(polygon).buffer(margin, resolution=2).area - inside_area

    xy_shift_positive = np.zeros([2*half+1, 2*half+1])
    xy_shift_negative = np.zeros([2*half+1, 2*half+1])
    z_shift_positive = []
    z_shift_negative = []

    for i in range(-half, half+1):
        for j in range(-half, half+1):
            region = polygon.copy()
            region[:, 0] += i * step_size
            region[:, 1] += j * step_size
            clock('Collect cells')
            t0 = time()
            path = Path(region)
            indices_inside = np.where(path.contains_points(locations))[0]
            features_inside = features[indices_inside]

            surround = Polygon(region).buffer(margin, resolution=2)
            path = Path(list(surround.exterior.coords))
            indices_sur = np.where(path.contains_points(locations))[0]
            indices_outside = np.setdiff1d(indices_sur, indices_inside)
            features_outside = features[indices_outside]
            print('Collect cells', time()-t0)
            clock('finish collection')

            clock('Boosting Scores')
            t0 = time()
            if features_inside.shape[0]:
                score = features_to_score(features_inside, thresholds, bst, inside_area)
                # x_shift_positive.append(score)
                xy_shift_positive[j+half, i+half] = score
            # else:
                # x_shift_positive.append(0)


            if features_outside.shape[0]:
                score = features_to_score(features_outside, thresholds, bst, outside_area)
                # x_shift_negative.append(score)
                xy_shift_negative[j + half, i + half] = score
            print('Boosting scores', time()-t0)
            clock('Finish scores')
            # else:
                # x_shift_negative.append(0)

            # region = polygon.copy()
            # region[:, 1] += i * step_size
            # path = Path(region)
            # indices_inside = np.where(path.contains_points(locations))[0]
            # features_inside = features[indices_inside]
            # if features_inside.shape[0]:
            #     score = features_to_score(features_inside, thresholds, bst, inside_area)
            #     y_shift_positive.append(score)
            # else:
            #     y_shift_positive.append(0)
            #
            # surround = Polygon(region).buffer(margin, resolution=2)
            # path = Path(list(surround.exterior.coords))
            # indices_sur = np.where(path.contains_points(locations))[0]
            # indices_outside = np.setdiff1d(indices_sur, indices_inside)
            # features_outside = features[indices_outside]
            # if features_outside.shape[0]:
            #     score = features_to_score(features_outside, thresholds, bst, outside_area)
            #     y_shift_negative.append(score)
            # else:
            #     y_shift_negative.append(0)

    conn.close()



    # Scores[structure][str(section) + '_positive']['x'] = x_shift_positive
    # Scores[structure][str(section) + '_positive']['y'] = y_shift_positive
    Scores[structure][str(section) + '_positive']['xy'] = xy_shift_positive

    # Scores[structure][str(section) + '_negative']['x'] = x_shift_negative
    # Scores[structure][str(section) + '_negative']['y'] = y_shift_negative
    Scores[structure][str(section) + '_negative']['xy'] = xy_shift_negative

    count += 1
    print(section, structure, count, '/', len(polygons))

# shutil.rmtree(os.environ['ROOT_DIR']+db_dir)
filename = savepath + str(section) + '.pkl'
pickle.dump(Scores, open(os.environ['ROOT_DIR'] + filename, 'wb'))
# setup_upload_from_s3(filename, recursive=False)
# os.remove(os.environ['ROOT_DIR']+img_fn)
print(str(section) + ' finished in %5.1f seconds' % (time() - t1))