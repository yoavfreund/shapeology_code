import numpy as np
import pickle
import xgboost as xgb
from time import time
import os
import sqlite3
from joblib import Parallel, delayed

time_log = {}
def time_count(message,duration):
    if message not in time_log.keys():
        time_log[message] = 0
    time_log[message] += duration


def collect_bitmap_cell_features(loc, all_cells, mask, area):
    '''
    This function is to use pre-computed masks to collect cells and compute feature vectors for all regions
    divided by bit combinations (0 or 1 defining outside or inside the shifting 2D shape for each shift).
    :param loc: x,y coordinates of the first point in the 2D scanning region
    :param all_cells: a matrix containing coordinates, 20*99 feature values and shape area for each cell in the shifting section
    :param mask: the pre-computed mask matrix giving indices of bit combinations to all sampled points in the 2D scanning region
    :param area: the area of a 2D shape
    :return: a matrix containing feature vectors for all bit combinations and a vector containing indices of bit combinations
    '''
    # Get the coordinates of cells centered by the first point in the 2D scanning region
    coord = all_cells[:, 1:3] - loc
    features_in_box = all_cells[:, 3:]
    # Get the coordinates of cells into the mask space by sampling ratio
    coord /= ratio
    coord = np.int16(coord)

    # Collect cells inside the scanning region
    indices = (coord[:, 0] >= 0) & (mask.shape[0] > coord[:, 0])
    coord = coord[indices]
    features_in_box = features_in_box[indices]
    indices = (coord[:, 1] >= 0) & (mask.shape[1] > coord[:, 1])
    coord = coord[indices]
    features_in_box = features_in_box[indices]

    # Compute feature vectors for all bit combinations
    start_time = time()
    bitvalues = mask[coord[:, 0], coord[:, 1]]
    cell_index = np.zeros([bitvalues.max() + 1, len(bitvalues)])
    cell_index[bitvalues, np.arange(bitvalues.size)] = 1
    time_count('Compute CDFs: construct matrix', time() - start_time)
    start_time = time()
    cell_index = cell_index[~np.all(cell_index == 0, axis=1)]
    time_count('Compute CDFs: delete zero raw', time() - start_time)
    start_time = time()
    cell_number = cell_index.sum(axis=1)
    cdfs = np.dot(cell_index, features_in_box[:, :-1]) / cell_number.reshape([-1, 1])
    number_ratio = cell_number.reshape([-1, 1]) / area * 317 * 317
    area_ratio = np.dot(cell_index, features_in_box[:, -1]).reshape([-1, 1]) / area
    feature_vectors = np.concatenate((cdfs, number_ratio, area_ratio), axis=1)
    time_count('Compute CDFs: cdf', time() - start_time)

    return feature_vectors, np.sort(np.unique(bitvalues))

def combine_vectors(keys, bitmaps, unique_combinations, area):
    '''
    This function is to combine feature vectors of all regions inside a shifted 2D shape into
    one feature vector for each shift
    :param keys: indices for all bit combinations in the current section
    :param bitmaps: a matrix containing feature vectors for all bit combinations
    :param unique_combinations: all possible bit combinations in the mask
    :return: a matrix containing feature vectors for all shifts
    '''
    X = unique_combinations[keys].T
    coefficient = X*bitmaps[:,-2] * area / 317 / 317
    alpha = coefficient.sum(axis=1)
    cdfs = np.dot(coefficient,bitmaps[:,:-2])/alpha.reshape([-1,1])
    scalars = np.dot(X,bitmaps[:,-2:])
    feature_vectors = np.concatenate((cdfs,scalars),axis=1)
    return feature_vectors

def computation(section, mode='search', xyz_shift_map=0):
    '''
    This function is to compute the detection score map for one 2D shape
    :param section: the section number of the 2D shape
    :param xyz_shift_map: the sum of detection score maps of processed 2D shapes
    :return: updated sum of detection score maps of processed 2D shapes
    '''
    print(section - section_numbers[0], (time() - t0) / 60)
    mode = 'search' if mode=='search' else 'refine'
    fn = os.environ['ROOT_DIR'] + 'Detection_preparation_mask/' + mode + '/'+ structure + '/' + \
         str(section - section_numbers[0]) + '.pkl'
    unique_combinations_inner, indices_inner, unique_combinations_outer, indices_outer = \
        pickle.load(open(fn, 'rb'))
    inside_area = total_shape_area[section - section_numbers[0]]
    outside_area = total_sur_area[section - section_numbers[0]]
    vectors_input = []
    for k in range(-half, half + 1):
        loc = np.array([center[0] + min_x - margin - half * step_size,
                        center[1] + min_y - margin - half * step_size])
        features_section = cell_shape_features[cell_shape_features[:, 0] == int(section + k * step_z)]
        start_time = time()
        bitmaps_inner, keys_inner = collect_bitmap_cell_features(loc, features_section, indices_inner, inside_area)
        bitmaps_outer, keys_outer = collect_bitmap_cell_features(loc, features_section, indices_outer, outside_area)
        time_count('Compute CDFs', time() - start_time)

        start_time = time()
        cdf_inner = combine_vectors(keys_inner, bitmaps_inner, unique_combinations_inner, inside_area)
        cdf_outer = combine_vectors(keys_outer, bitmaps_outer, unique_combinations_outer, outside_area)
        time_count('Combine CDFs', time() - start_time)
        vectors_input.extend(list(cdf_inner - cdf_outer))

    start_time = time()
    xtest = xgb.DMatrix(vectors_input)
    score = bst.predict(xtest, output_margin=True, ntree_limit=bst.best_ntree_limit)
    xyz_shift_map += score.reshape([2 * half + 1, 2 * half + 1, -1], order='F')
    time_count('Compute xgboost', time() - start_time)
    return xyz_shift_map


param = {}
param['max_depth'] = 3  # depth of tree
param['eta'] = 0.2  # shrinkage parameter
param['silent'] = 1  # not silent
param['objective'] = 'binary:logistic'  # 'multi:softmax'
param['nthread'] = 7  # Number of threads used
param['num_class'] = 1
num_round = 100

# fn = 'CSHL_data_processed/MD589/ThresholdsV2.pkl'
fn = 'CSHL_data_processed/MD589/Thresholds_refined.pkl'
thresholds = pickle.load(open(os.environ['ROOT_DIR'] + fn, 'rb'))

if __name__=='__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--structure", type=str, default='5N_L.pkl',
                        help="atlas information for one structure")
    parser.add_argument("--center", type=str, default='DK52.pkl',
                        help="annotation information for one brain")
    parser.add_argument("mode", type=str,
                        help="Search or refine")
    args = parser.parse_args()
    structure = args.structure
    stack = args.center
    mode = args.mode

    t0 = time()
    savepath = 'CSHL_shift_scores/bitmap_retrained/' + mode+ '/' + stack + '/'
    if not os.path.exists(os.environ['ROOT_DIR'] + savepath):
        os.makedirs(os.environ['ROOT_DIR'] + savepath)

    resol = 0.325
    margin = 200 / resol
    fn = os.environ['ROOT_DIR'] + 'Detection_preparation_v2/' + structure+'.pkl'
    grid3D, total_shape_area, total_sur_area, min_x, min_y, len_max = pickle.load(open(fn,'rb'))
    if mode == 'search':
        step_size = max(round(len_max / 20), round(30 / resol))
        fn = stack + '/' + stack + '_rough_landmarks.pkl'
        model_fn = 'Detection_models/v6_refine/'
    elif mode == 'refine':
        step_size = max(int(len_max / 30), int(20 / resol))
        fn = stack + '/retrained/' + stack + '_search_landmarks.pkl'
        model_fn = 'Detection_models/v6_refine/'
    elif mode == 'retrain':
        step_size = max(int(len_max / 30), int(20 / resol))
        fn = stack + '/' + stack + '_detected_landmarks.pkl'
        model_fn = 'Detection_models/v5_refine/'
    step_z = int(round(step_size * resol / 20))
    half = 15
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
    [left, right, up, down] = [int(max(min(polygon_set[:, 0]) - margin - half * step_size, 0)),
                               int(np.ceil(max(polygon_set[:, 0]) + margin + half * step_size)),
                               int(max(min(polygon_set[:, 1]) - margin - half * step_size, 0)),
                               int(np.ceil(max(polygon_set[:, 1]) + margin + half * step_size))]
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
    # Transfer each feature value into a 99-bit form
    sample_len = len(thresholds[0])
    bit_features = np.zeros([cell_shape_features.shape[0], (cell_shape_features.shape[1] - 3) * sample_len])
    for k in range(len(thresholds)):
        feature_k = cell_shape_features[:, 3 + k].copy()
        for i in range(sample_len):
            if i == 0:
                bit_features[:, k * sample_len + i:(k + 1) * sample_len][thresholds[k][i] >= feature_k] = 1
            indices = (feature_k > thresholds[k][i]) & (thresholds[k][i + 1] >= feature_k) if i < sample_len - 1 \
                else (feature_k > thresholds[k][i])
            bit_features[:, k * sample_len + i + 1:(k + 1) * sample_len][indices] = 1
    cell_shape_features = np.concatenate(
        (cell_shape_features[:, :3], bit_features, cell_shape_features[:, 15].reshape([-1, 1])), axis=1)

    ratio = round(20 / resol)
    rname = structure[:structure.rfind('_')] if structure.rfind('_') != -1 else structure
    bst = pickle.load(open(os.environ['ROOT_DIR'] + model_fn + rname + '.pkl', 'rb'))
    xyz_shift_map = np.zeros([2 * half + 1, 2 * half + 1, 2 * half + 1])
    # for section in section_numbers:
    #     xyz_shift_map = computation(section, xyz_shift_map)
    score_maps = Parallel(n_jobs=10)(delayed(computation)(i, mode, xyz_shift_map) for i in section_numbers)
    xyz_shift_map = np.sum(score_maps, axis=0)


    fn = savepath + structure + '.pkl'
    pickle.dump(xyz_shift_map, open(os.environ['ROOT_DIR'] + fn, 'wb'))
    time_count('Total', time() - t0)
    pickle.dump(time_log, open(os.environ['ROOT_DIR'] + savepath + structure + '_time.pkl', 'wb'))





