import numpy as np
import pickle
import xgboost as xgb
from time import time
import os
import sqlite3
from scipy import stats

time_log = {}
def time_count(message,duration):
    if message not in time_log.keys():
        time_log[message] = 0
    time_log[message] += duration

def collect_inside_cell_features(loc):
    coord = cell_shape_features[:, :3] - loc
    features_in_box = cell_shape_features[:, 3:]
    coord[:, 1:] /= ratio
    coord = np.int16(coord)

    indices = (coord[:, 0] >= 0) & (grid3D.shape[0] > coord[:, 0])
    coord = coord[indices]
    features_in_box = features_in_box[indices]
    indices = (coord[:, 1] >= 0) & (grid3D.shape[1] > coord[:, 1])
    coord = coord[indices]
    features_in_box = features_in_box[indices]
    indices = (coord[:, 2] >= 0) & (grid3D.shape[2] > coord[:, 2])
    coord = coord[indices]
    features_in_box = features_in_box[indices]

    values = grid3D[coord[:, 0], coord[:, 1], coord[:, 2]]
    indices = (values == 2)
    inside_shape_features = features_in_box[indices]
    inside_coord = coord[indices]
    indices = (values == 1)
    sur_shape_features = features_in_box[indices]
    sur_coord = coord[indices]
    return inside_shape_features,sur_shape_features,inside_coord,sur_coord

def CDF_comparison(features_inside,features_outside):
    eva = 0
    p_value = []
    for k in range(features_inside.shape[1]):
        x1 = np.sort(features_outside[:, k])
        x2 = np.sort(features_inside[:, k])
        value = stats.ks_2samp(x1, x2, mode='asymp')[1]
        eva += - np.log(value)
        p_value.append(- np.log(value))
    eva = sum(sorted(p_value, reverse=True)[:5])
    return eva

def features_to_vector(features, thresholds, object_area):
    extracted = []
    n1 = features.shape[0]
    np.sort(features,axis=0)
    for k in list(range(10)) + [14]:
    # for k in range(features.shape[1]): #iterate over features, one feature equal one cdf
        data1 = np.sort(features[:, k]) # sort feature k
        cdf = np.searchsorted(data1, thresholds[k], side='right') / n1
        extracted.extend(cdf)
    extracted.extend([features.shape[0] / object_area * 317 * 317]) # cell number normalized by region area
    extracted.extend([features[:, 12].sum() / object_area]) # cell areas in total normalized by region area
    return extracted


param = {}
param['max_depth'] = 3  # depth of tree
param['eta'] = 0.2  # shrinkage parameter
param['silent'] = 1  # not silent
param['objective'] = 'binary:logistic'  # 'multi:softmax'
param['nthread'] = 7  # Number of threads used
param['num_class'] = 1
num_round = 100

fn = 'CSHL_data_processed/MD589/ThresholdsV2.pkl'
thresholds = pickle.load(open(os.environ['ROOT_DIR'] + fn, 'rb'))


if __name__=='__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--structure", type=str, default='5N_L.pkl',
                        help="atlas information for one structure")
    parser.add_argument("--center", type=str, default='DK52.pkl',
                        help="annotation information for one brain")
    args = parser.parse_args()
    structure = args.structure
    stack = args.center

    savepath = 'CSHL_shift_scores/'+ stack + '_dm_2D/'
    t0 = time()
    # savepath = 'CSHL_shift_scores/cdf_fine_2D/' + stack + '/'
    if not os.path.exists(os.environ['ROOT_DIR'] + savepath):
        os.makedirs(os.environ['ROOT_DIR'] + savepath)

    resol = 0.325
    margin = 200 / resol
    fn = os.environ['ROOT_DIR'] + 'Detection_preparation_v2/' + structure+'.pkl'
    grid3D, total_shape_area, total_sur_area, min_x, min_y, len_max = pickle.load(open(fn,'rb'))
    step_size = max(int(len_max / 20), int(30 / resol))
    # step_size = max(int(len_max / 30), int(20 / resol))
    step_z = int(step_size * resol / 20)

    half = 15
    # fn = stack + '/' + stack + '_search_landmarks.pkl'
    fn = stack + '/' + stack + '_rough_landmarks.pkl'
    contours = pickle.load(open(os.environ['ROOT_DIR'] + fn, 'rb'))
    seq = sorted([section for section in contours.keys() if structure in contours[section].keys()])
    C = {i: contours[i][structure] for i in contours if structure in contours[i]}
    section_numbers = sorted(C.keys())
    Concat = np.concatenate([C[i] for i in C])
    center = np.mean(Concat, axis=0)

    polygon_set = []
    for sec in seq:
        polygon = contours[sec][structure].copy()
        polygon_set.append(polygon)
    polygon_set = np.concatenate(polygon_set)
    [left, right, up, down] = [int(max(min(polygon_set[:, 0]) -margin -half * step_size, 0)),
                               int(np.ceil(max(polygon_set[:, 0])  +margin + half * step_size)),
                               int(max(min(polygon_set[:, 1])  -margin - half * step_size, 0)),
                               int(np.ceil(max(polygon_set[:, 1]) +margin + half * step_size))]
    expend_seq = list(range(seq[0] - half * step_z, seq[0]))
    expend_seq.extend(seq)
    expend_seq.extend(list(range(seq[-1] + 1, seq[-1] + half * step_z + 1)))

    db_dir = 'CSHL_databases/' + stack + '/'
    cell_shape_features = []
    for section in expend_seq:
        try:
            sec_fp = db_dir + '%03d' % section + '.db'

            conn = sqlite3.connect(os.environ['ROOT_DIR'] + sec_fp)
            cur = conn.cursor()
            raws = cur.execute('SELECT * FROM features WHERE x>=? AND x<=? AND y>=? AND y<=?',
                               (left, right, up, down))
            info = np.array(list(raws))
            locations = info[:, 1:3]
            features = info[:, 3:]
            cell_shape_features.append(info)
        except:
            continue
    cell_shape_features = np.concatenate(cell_shape_features)

    ratio = int(20 / resol)
    bst = pickle.load(open(os.environ['ROOT_DIR'] + 'Detection_models/dm_only/' + structure + '.pkl', 'rb'))
    # bst = pickle.load(open(os.environ['ROOT_DIR'] + 'Detection_models/v5/' + structure + '.pkl', 'rb'))
    # vectors_as_input = []
    xyz_shift_map = np.zeros([2 * half + 1, 2 * half + 1, 2 * half + 1])
    section_map = {}
    cdf_collection = {}

    for k in range(-half, half + 1):
        print(k)
        for i in range(-half, half + 1):
            for j in range(-half, half + 1):
                # Collect cells in the shifted 3D shapes
                start_time = time()
                cdf_collection[(j, i, k)] = {}

                loc = np.array([seq[0] + k * step_z, center[0] + min_x-margin + i * step_size, center[1] + min_y-margin + j * step_size])
                inside_shape_features,sur_shape_features,inside_coord,sur_coord = collect_inside_cell_features(loc)
                time_count('Collect cells', time() - start_time)
                vectors_input = []
                for sec in seq:
                    if sec- seq[0] not in section_map.keys():
                        section_map[sec- seq[0]] = np.zeros([2 * half + 1, 2 * half + 1, 2 * half + 1])
                    start_time = time()
                    inside_features = inside_shape_features[inside_coord[:, 0] == sec - seq[0]]
                    outside_features = sur_shape_features[sur_coord[:, 0] == sec - seq[0]]
                    time_count('Collect cells by section', time() - start_time)
                    if inside_features.shape[0] and outside_features.shape[0]:
                        start_time = time()
                        inside_cdf = features_to_vector(inside_features, thresholds, total_shape_area[sec - seq[0]])
                        sur_cdf = features_to_vector(outside_features, thresholds, total_sur_area[sec - seq[0]])
                        feature_vector = np.array(inside_cdf) - np.array(sur_cdf)
                        time_count('Compute CDFs', time() - start_time)
                        vectors_input.append(feature_vector)
                        cdf_collection[(j, i, k)][sec - seq[0]] = feature_vector
                if len(vectors_input):
                    start_time = time()
                    xtest = xgb.DMatrix(vectors_input)
                    score = bst.predict(xtest, output_margin=True, ntree_limit=bst.best_ntree_limit)
                    xyz_shift_map[j + half, i + half, k + half] = sum(score)
                    time_count('Compute xgboost', time() - start_time)
                    for indice in range(len(vectors_input)):
                        sec = sorted(cdf_collection[(j, i, k)].keys())[indice]
                        section_map[sec][j + half, i + half, k + half] = score[indice]

    fn = savepath + structure + '.pkl'
    pickle.dump(xyz_shift_map, open(os.environ['ROOT_DIR'] + fn, 'wb'))
    time_count('Total', time() - t0)
    pickle.dump(time_log, open(os.environ['ROOT_DIR'] + savepath + structure + '_time.pkl', 'wb'))
    fn = savepath + structure + '_maps.pkl'
    pickle.dump(section_map, open(os.environ['ROOT_DIR'] + fn, 'wb'))
    # fn = savepath + structure + '_vectors.pkl'
    # pickle.dump(cdf_collection, open(os.environ['ROOT_DIR'] + fn, 'wb'))

