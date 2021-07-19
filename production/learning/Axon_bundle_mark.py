import os
import sqlite3
import numpy as np
import cv2
from skimage import io,color
import xgboost as xgb
import pickle

def middle_angle(cdfs,angles):
    alpha = 20
    for cdf in cdfs:
        max_diff = 0
        for idx in range(len(angles)):
            segment = angles-angles[idx]-alpha
            segment = segment[segment<=0]
            end = np.argmax(segment)
            if max_diff<cdf[end]-cdf[idx]:
                max_diff = cdf[end]-cdf[idx]
                middle_angle = angles[idx]+alpha/2
    return middle_angle

def color_angle(score):
    if score>1.65:
        return [255,0,0]
    elif score>1.22:
        return [0,255,0]
    else:
        return [0,0,255]

def max_difference(cdfs,angles):
    diff = []
    for cdf in cdfs:
        value = []
        for alpha in [20]:
            max_diff = 0
            for idx in range(len(angles)):
                segment = angles-angles[idx]-alpha
                segment = segment[segment<=0]
                end = np.argmax(segment)
                if max_diff<cdf[end]-cdf[idx]:
                    max_diff = cdf[end]-cdf[idx]
            value.extend([max_diff])
        diff.append(value)
    return np.array(diff)

param = {}
param['max_depth'] = 3  # depth of tree
param['eta'] = 0.2  # shrinkage parameter
param['silent'] = 1  # not silent
param['objective'] = 'binary:logistic'  # 'multi:softmax'
param['nthread'] = 7  # Number of threads used
param['num_class'] = 1
num_round = 100

resol = 0.325
window_size = int(100/resol)
stride = window_size #int(window_size/2)
object_area = window_size**2

fn = 'CSHL_data_processed/MD589/ThresholdsV2.pkl'
thresholds = pickle.load(open(os.environ['ROOT_DIR'] + fn, 'rb'))

cell_dir = os.environ['ROOT_DIR'] + 'CSHL_patch_samples_features_v1/'
fp = []
fp.append(cell_dir + 'bundles/bundles_positive.pkl')
fp.append(cell_dir + 'bundles/bundles_negative.pkl')
X_train = []
y_train = []
for state in range(2):
    clouds = np.array(pickle.load(open(fp[state], 'rb')))
    new_features = max_difference(clouds[:,14*99:15*99],np.array(thresholds[14]))
    clouds = np.concatenate((clouds, new_features),axis=1)
    X_train.extend(clouds)
    y_train.extend([1 - state] * len(clouds))
dtrain = xgb.DMatrix(X_train, label=y_train)
bst = xgb.train(param, dtrain, num_round, verbose_eval=False)


if __name__=="__main__":

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("stack", type=str, help="The name of the brain")
    parser.add_argument("file", type=str, help="The path to the image file")
    parser.add_argument("--save_dir", type=str, default=os.path.join(os.environ['ROOT_DIR'], 'Axon_bundle/'),
                        help="Path to directory saving images")
    args = parser.parse_args()
    stack = args.stack
    img_fn = args.file
    save_dir = os.path.join(args.save_dir, stack)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    dot = img_fn.rfind('.')
    slash = img_fn.rfind('/')
    section = img_fn[slash + 1:dot]
    save_fn = os.path.join(save_dir, section+'.tif')
    section = int(section)

    img = io.imread(img_fn)
    m, n = img.shape
    xs, ys = np.meshgrid(np.arange(0, n, stride), np.arange(0, m, stride), indexing='xy')
    windows = np.c_[xs.flat, ys.flat]

    locations_all = []
    features_all = []

    for i in range(-2,3):
        db_dir = 'CSHL_databases/' + stack + '/'
        db_fp = db_dir + str(section + i) + '.db'
        conn = sqlite3.connect(os.environ['ROOT_DIR'] + db_fp)
        cur = conn.cursor()

        raws = cur.execute('SELECT * FROM features')
        info = np.array(list(raws))
        locations = info[:, :3]
        features = info[:, 3:]
        locations_all.extend(locations)
        features_all.extend(features)

    features_all = np.array(features_all)
    locations_all = np.array(locations_all)

    rgb = color.gray2rgb(img)
    for index in range(len(windows)):
        extracted = []
        wx = int(windows[index][0])
        wy = int(windows[index][1])
        indices_window = np.where((locations_all[:, 1] > wx) & (locations_all[:, 1] < wx + window_size) \
                                  & (locations_all[:, 2] > wy) & (locations_all[:, 2] < wy + window_size))[0]
        if len(indices_window):
            cells = features_all[indices_window]
            n1 = cells.shape[0]
            for k in range(cells.shape[1]):
                data1 = np.sort(cells[:, k])
                ten = np.searchsorted(data1, thresholds[k], side='right') / n1
                if k == 14:
                    new_features = max_difference(np.array(ten).reshape(1, -1), np.array(thresholds[14]))
                    new_features = list(new_features.reshape(new_features.shape[1]))
                extracted.extend(ten)
            extracted.extend([n1 / object_area * 317 * 317])
            extracted.extend([cells[:, 12].sum() / object_area])
            extracted.extend(new_features)
        else:
            extracted.append([0] * 1983)
        xtest = xgb.DMatrix(extracted)
        score = bst.predict(xtest, output_margin=True, ntree_limit=bst.best_ntree_limit)
        if score > 0.93:
            bundle_angle = middle_angle(np.array(extracted[14 * 99:15 * 99]).reshape(1, -1), np.array(thresholds[14]))
            color = color_angle(score)
            coef = np.tan(bundle_angle / 180 * np.pi)
            # print(index, score, bundle_angle)
            cells_loc = locations_all[indices_window]
            indices_cells = np.where(((bundle_angle - 10) < cells[:, 14]) & (cells[:, 14] < (bundle_angle + 10)) \
                                     & (cells_loc[:, 0] == section))[0]
            for idx in indices_cells:
                left, top = int(cells_loc[idx, 1] - cells[idx, 10] / 2), int(cells_loc[idx, 2] - cells[idx, 11] / 2)
                right, bottom = int(cells_loc[idx, 1] + cells[idx, 10] / 2), int(cells_loc[idx, 2] + cells[idx, 11] / 2)
                if cells[idx, 14] > 0:
                    cv2.line(rgb, (left, top), (right, bottom), color, 2)
                else:
                    cv2.line(rgb, (left, bottom), (right, top), color, 2)

            if bundle_angle > 0:
                if coef > 1:
                    cv2.line(rgb, (wx, wy), (wx + window_size // 2, wy + int(window_size / 2 / coef)), color, 20)
                else:
                    cv2.line(rgb, (wx, wy), (wx + int(window_size / 2 * coef), wy + window_size // 2), color, 20)
            else:
                if coef < -1:
                    cv2.line(rgb, (wx, wy + window_size // 2),
                             (wx + window_size // 2, wy + window_size // 2 + int(window_size / 2 / coef)), color, 20)
                else:
                    cv2.line(rgb, (wx, wy + window_size // 2), (wx - int(window_size / 2 * coef), wy), color, 20)


    io.imsave(save_fn, rgb)