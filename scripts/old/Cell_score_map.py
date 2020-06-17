import argparse

parser = argparse.ArgumentParser()
parser.add_argument("stack", type=str, help="The name of the stack", default='MD594')
parser.add_argument("section", type=int, default=256, help="The section number")
args = parser.parse_args()
stack = args.stack
section = args.section


import cv2
import numpy as np
import pandas as pd
import pickle
#from matplotlib import pyplot as plt
import skimage
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import sys
import sqlite3
import xgboost as xgb
import colorsys
import shutil
from time import time
from skimage.transform import rescale
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

fn = 'CSHL_data_processed/MD589/ThresholdsV2.pkl'
setup_download_from_s3(fn, recursive=False)
thresholds = pickle.load(open(os.environ['ROOT_DIR']+fn,'rb'))

fn = 'CSHL_data_processed/MD594/holds.pkl'
setup_download_from_s3(fn, recursive=False)
holds = pickle.load(open(os.environ['ROOT_DIR']+fn,'rb'))

#Parameters
param = {}
param['max_depth']= 3   # depth of tree
param['eta'] = 0.2      # shrinkage parameter
param['silent'] = 1     # not silent
param['objective'] = 'binary:logistic' #'multi:softmax'
param['nthread'] = 7 # Number of threads used
param['num_class']=1
num_round = 100

cell_dir = os.environ['ROOT_DIR'] + 'CSHL_patch_samples_features/MD589/'
cell2_dir = os.environ['ROOT_DIR'] + 'CSHL_patch_samples_features/MD585/'

savepath = 'CSHL_cells_score/'
if not os.path.exists(os.environ['ROOT_DIR']+savepath):
    os.mkdir(os.environ['ROOT_DIR']+savepath)
savepath = savepath+stack+'/'
if not os.path.exists(os.environ['ROOT_DIR']+savepath):
    os.mkdir(os.environ['ROOT_DIR']+savepath)

resol = 0.46

paired_structures = ['5N', '6N', '7N', '7n', 'Amb', 'LC', 'LRt', 'Pn', 'Tz', 'VLL', 'RMC', \
                     'SNC', 'SNR', '3N', '4N', 'Sp5I', 'Sp5O', 'Sp5C', 'PBG', '10N', 'VCA', 'VCP', 'DC']
singular_structures = ['AP', '12N', 'RtTg', 'SC', 'IC']

all_structures = paired_structures + singular_structures

# sets = {0: ['5N', 'Sp5C', 'Tz', 'SNC', 'LC', 'SC', '12N'],
#  1: ['6N', '3N', 'VLL', 'Sp5I', '7N', 'IC', 'AP'],
#  2: ['7n', 'RMC', 'Pn', 'Amb', '10N', 'DC', 'PBG'],
#  3: ['LRt', 'SNR', '4N', 'VCA'],
#  4: ['Sp5O', 'RtTg', 'VCP']}

margin = 200/0.46

polygons = [(contour['name'], contour['vertices']) \
            for contour_id, contour in contours_grouped.get_group(section).iterrows()]

db_dir = 'CSHL_databases/' + stack + '/'
db_fp = db_dir + str(section) + '.db'
setup_download_from_s3(db_fp, recursive=False)
conn = sqlite3.connect(os.environ['ROOT_DIR']+ db_fp)
cur = conn.cursor()

raw_images_root = 'CSHL_data_processed/'+stack+'/'+stack+'_prep2_lossless_gray/'
img_fn = raw_images_root + section_to_filename[section] + '_prep2_lossless_gray.tif'
setup_download_from_s3(img_fn, recursive=False)
img = cv2.imread(os.environ['ROOT_DIR']+img_fn, 2)
m, n = img.shape

origin= ['area', 'height', 'horiz_std', 'mean', 'padded_size',
       'rotation', 'rotation_confidence', 'std', 'vert_std', 'width', 'density', 'area_ratio']
columns = []
for i in range(10):
    name = 'DMVec'+str(i)
    for j in range(99):
        columns.append(name+'*'+str(j))
for i in range(10):
    name = origin[i]
    for j in range(99):
        columns.append(name+'*'+str(j))
columns.extend(origin[-2:])
columns = np.array(columns)

thresh = cv2.adaptiveThreshold(255 - img, 1, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 101, -20)
Stats = cv2.connectedComponentsWithStats(thresh)
mask = Stats[1]
window_size = 224
stride = 56
color_mark = [0.02, 0.33, 0.68, 270/360, 0.85]
# color_posi = [20/360, 40/360, 60/360, 80/360, 140/360, 164/360, 200/360]
count = 0

whole = np.zeros([m, n, 3], dtype=np.uint8)
whole = skimage.color.gray2rgb(img.copy())
hsv = np.zeros([m, n, 3])
hsv[:,:,2] = img.copy()/255
bboxs = []
cs = []
name_struc = []
center = []

for contour_id, contour in polygons:
    structure = contour_id
    if structure not in all_structures:
        continue
    polygon = contour.copy()
    center.append([polygon[:, 0].mean(), polygon[:, 1].mean()])

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
                               int(min(np.ceil(max(polygon[:, 0]) + margin), n)),
                               int(max(min(polygon[:, 1]) - margin, 0)),
                               int(min(np.ceil(max(polygon[:, 1]) + margin), m))]
    bboxs.append([left, right, up, down])
    cs.append(polygon.astype(np.int32))
    name_struc.append(structure)
    raws = cur.execute('SELECT * FROM features WHERE x>=? AND x<=? AND y>=? AND y<=?', (left, right, up, down))
    info = np.array(list(raws))
    locations = info[:, 1:3]
    features = info[:, 3:]

    xs, ys = np.meshgrid(np.arange(left, right, stride), np.arange(up, down, stride), indexing='xy')
    windows = np.c_[xs.flat, ys.flat]

    for index in range(len(windows)):
        extracted = []
        wx = int(windows[index][0])
        wy = int(windows[index][1])
        if hsv[wy, wx, 1] != 0:
            ignore = 0
            distance = np.sqrt((center[-1][0] - wx) ** 2 + (center[-1][1] - wy) ** 2)
            for i in range(len(center)-1):
                dis = np.sqrt((center[i][0]-wx)**2+(center[i][1]-wy)**2)
                if distance>dis:
                    ignore = 1
                    break
            if ignore:
                continue
        indices_window = np.where((locations[:, 0] > wx) & (locations[:, 0] < wx + window_size) \
                                  & (locations[:, 1] > wy) & (locations[:, 1] < wy + window_size))[0]
        if len(indices_window):
            cells = features[indices_window]
            for k in range(cells.shape[1]):
                x, y = CDF(cells[:, k])
                ten = [y[np.argmin(np.absolute(x - thresholds[k][j]))] for j in range(99)]
                extracted.extend(ten)
            extracted.extend([cells.shape[0]])
            extracted.extend([cells[:, 10].sum() / (224 * 224)])
        else:
            extracted.append([0] * 1982)
        xtest = xgb.DMatrix(extracted)
        score = bst.predict(xtest, output_margin=True, ntree_limit=bst.best_ntree_limit)
        origin = hsv[wy: wy + window_size, wx: wx + window_size, 1]
        satua_img = np.zeros_like(origin) + score
        comp = np.absolute(origin) - np.absolute(satua_img)
        hsv[wy: wy + window_size, wx: wx + window_size, 1] = origin * (comp > 0) + satua_img * (comp < 0)

    count += 1
    print(section, count, '/', len(polygons))

hsv2 = hsv.copy()
for i in range(len(bboxs)):
    left, right, up, down = bboxs[i]
    hsv2[up:down, left:right, 0] = (hsv[up:down, left:right, 1] <= 0) * 1 + (hsv[up:down, left:right, 1] > 0) * 0.66
    # color_group[structure]
    # hsv[up:down, left:right, 1] = 0.3
    # hsv[up:down, left:right, 0] = color_group[structure]
    # hsv[up:down, left:right, 1] = np.absolute(hsv[up:down, left:right, 1])
    # hsv[up:down, left:right, 1] = (hsv[up:down, left:right, 1] - hsv[up:down, left:right, 1].min()) \
    #                               / (hsv[up:down, left:right, 1].max() - hsv[up:down, left:right, 1].min()) * 0.3

for i in range(len(bboxs)):
    left, right, up, down = bboxs[i]
    hsv2[up:down, left:right, 1] = np.absolute(hsv[up:down, left:right, 1])
    hsv2[up:down, left:right, 1] = (hsv[up:down, left:right, 1] - hsv[up:down, left:right, 1].min()) \
                                  / (hsv[up:down, left:right, 1].max() - hsv[up:down, left:right, 1].min()) * 0.2+0.05
    rgb = skimage.color.hsv2rgb(hsv2[up:down, left:right, :])
    rgb = rgb * 255
    rgb = rgb.astype(np.uint8)
    whole[up:down, left:right, :] = rgb.copy()

for i in range(len(bboxs)):
    left, right, up, down = bboxs[i]
    polygon = cs[i]
    # polygon[:, 0] = polygon[:, 0] - left
    # polygon[:, 1] = polygon[:, 1] - up
    area = (right - left) * (down - up)
    thickness = int(min(right - left, down - up)/80)
    # com = cv2.GaussianBlur(rgb.copy(), (5, 5), 0)
    cv2.polylines(whole, [polygon.astype(np.int32)], True, [0, 255, 0], thickness, lineType=8)
    cv2.putText(whole, name_struc[i], (left, up + 300), cv2.FONT_HERSHEY_SIMPLEX, 15, [0, 0, 255],
                thickness=20)




filename = savepath + str(section) + '.jpg'
# whole = cv2.cvtColor(whole, cv2.COLOR_BGR2RGB)
whole8 = rescale(whole, 1.0/4, multichannel=True, anti_aliasing=True)
whole8 = whole8 * 255
whole8 = whole8.astype(np.uint8)
cv2.imwrite(os.environ['ROOT_DIR'] + filename, whole8)
setup_upload_from_s3(filename, recursive=False)



# os.remove(os.environ['ROOT_DIR']+img_fn)

