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

def features_to_vector(features, thresholds, object_area):
    extracted = []
    n1 = features.shape[0]
    data = np.sort(features,axis=0)
    # for k in list(range(10)) + [14]:
    for k in range(features.shape[1]): #iterate over features, one feature equal one cdf
        data1 = data[:,k]#np.sort(features[:, k]) # sort feature k
        cdf = np.searchsorted(data1, thresholds[k], side='right') / n1
        extracted.extend(cdf)
    extracted.extend([features.shape[0] / object_area * 317 * 317]) # cell number normalized by region area
    extracted.extend([features[:, 12].sum() / object_area]) # cell areas in total normalized by region area
    return extracted

def collect_bitmap_cell_features(loc,all_cells,mask,area):
    coord = all_cells[:, 1:3] - loc
    features_in_box = all_cells[:, 3:]
    coord /= ratio
    coord = np.int16(coord)

    indices = (coord[:, 0] >= 0) & (mask.shape[0] > coord[:, 0])
    coord = coord[indices]
    features_in_box = features_in_box[indices]
    indices = (coord[:, 1] >= 0) & (mask.shape[1] > coord[:, 1])
    coord = coord[indices]
    features_in_box = features_in_box[indices]

    bitvalues = mask[coord[:, 0], coord[:, 1]]
    bitmaps = {}
    for index in np.unique(bitvalues):
        bitmaps[index] = features_to_vector(features_in_box[bitvalues == index], thresholds, area)

    return bitmaps

# def combine_vectors(elements, feature_map):
#     feature_vector = feature_map[elements[0]]
#     if len(elements)>1:
#         start_time = time()
#         vectors = feature_map[elements]
#         time_count('Combine CDFs: transform', time() - start_time)
#         start_time = time()
#         alpha = vectors[:,-2].sum()
#         feature_vector[:-2] = np.dot(vectors[:,-2].reshape([1,-1]), vectors[:,:-2])/alpha
#         feature_vector[-2:] = vectors[:,-2:].sum(axis=0)
#         time_count('Combine CDFs: calculate', time() - start_time)
#     return feature_vector

def combine_vectors(keys, bitmaps, unique_combinations):
    X = unique_combinations[keys].T
    coefficient = X*bitmaps[:,-2]
    alpha = coefficient.sum(axis=1)
    cdfs = np.dot(coefficient,bitmaps[:,:-2])/alpha.reshape([-1,1])
    scalars = np.dot(X,bitmaps[:,-2:])
    feature_vectors = np.concatenate((cdfs,scalars),axis=1)
    return feature_vectors

def computation(section, xyz_shift_map=0):
    print(section - section_numbers[0], (time() - t0) / 60)
    fn = os.environ['ROOT_DIR'] + 'Detection_preparation_mask/' + structure + '/' + \
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
        bitmaps_inner = collect_bitmap_cell_features(loc, features_section, indices_inner, inside_area)
        bitmaps_outer = collect_bitmap_cell_features(loc, features_section, indices_outer, outside_area)
        bitmaps_inner, keys_inner = np.array(list(bitmaps_inner.values())), np.array(list(bitmaps_inner.keys()))
        bitmaps_outer, keys_outer = np.array(list(bitmaps_outer.values())), np.array(list(bitmaps_outer.keys()))
        time_count('Compute CDFs', time() - start_time)

        start_time = time()
        cdf_inner = combine_vectors(keys_inner, bitmaps_inner, unique_combinations_inner)
        cdf_outer = combine_vectors(keys_outer, bitmaps_outer, unique_combinations_outer)
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

    t0 = time()
    savepath = 'CSHL_shift_scores/bitmap/' + stack + '/'
    if not os.path.exists(os.environ['ROOT_DIR'] + savepath):
        os.makedirs(os.environ['ROOT_DIR'] + savepath)

    resol = 0.325
    margin = 200 / resol
    fn = os.environ['ROOT_DIR'] + 'Detection_preparation_v2/' + structure+'.pkl'
    grid3D, total_shape_area, total_sur_area, min_x, min_y, len_max = pickle.load(open(fn,'rb'))
    step_size = max(round(len_max / 20), round(30 / resol))
    # step_size = max(int(len_max / 30), int(20 / resol))
    step_z = int(round(step_size * resol / 20))

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

    ratio = round(20 / resol)
    bst = pickle.load(open(os.environ['ROOT_DIR'] + 'Detection_models/v5/' + structure + '.pkl', 'rb'))
    xyz_shift_map = np.zeros([2 * half + 1, 2 * half + 1, 2 * half + 1])
    score_maps = Parallel(n_jobs=10)(delayed(computation)(i, xyz_shift_map) for i in section_numbers)
    xyz_shift_map = np.sum(score_maps, axis=0)

    # for section in section_numbers:
    #     print(section, (time() - t0)/60)
    #     fn = os.environ['ROOT_DIR'] + 'Detection_preparation_mask/' + structure +'/'+ \
    #          str(section - section_numbers[0]) + '.pkl'
    #     unique_combinations_inner, indices_inner, unique_combinations_outer, indices_outer = \
    #     pickle.load(open(fn,'rb'))
    #     inside_area = total_shape_area[section - section_numbers[0]]
    #     outside_area = total_sur_area[section - section_numbers[0]]
    #     vectors_input = []
    #     # loc_input = []
    #     for k in range(-half, half + 1):
    #         loc = np.array([center[0] + min_x - margin - half * step_size,
    #                         center[1] + min_y - margin - half * step_size])
    #         features_section = cell_shape_features[cell_shape_features[:, 0] == int(section + k * step_z)]
    #         start_time = time()
    #         bitmaps_inner = collect_bitmap_cell_features(loc,features_section,indices_inner,inside_area)
    #         bitmaps_outer = collect_bitmap_cell_features(loc, features_section, indices_outer,outside_area)
    #         bitmaps_inner, keys_inner = np.array(list(bitmaps_inner.values())), np.array(list(bitmaps_inner.keys()))
    #         bitmaps_outer, keys_outer = np.array(list(bitmaps_outer.values())), np.array(list(bitmaps_outer.keys()))
    #         time_count('Compute CDFs', time() - start_time)
    #
    #         start_time = time()
    #         cdf_inner = combine_vectors(keys_inner, bitmaps_inner, unique_combinations_inner)
    #         cdf_outer = combine_vectors(keys_outer, bitmaps_outer, unique_combinations_outer)
    #         time_count('Combine CDFs', time() - start_time)
    #         vectors_input.extend(list(cdf_inner - cdf_outer))
    #
    #         # for i in range(-half, half + 1):
    #         #     for j in range(-half, half + 1):
    #         #         element_inner = np.array([index for index in range(len(keys_inner)) \
    #         #                                   if unique_combinations_inner[keys_inner[index]][
    #         #                                       (i + half) * (2 * half + 1) + j + half]])
    #         #         element_outer = np.array([index for index in range(len(keys_outer)) \
    #         #                                   if unique_combinations_outer[keys_outer[index]][
    #         #                                       (i + half) * (2 * half + 1) + j + half]])
    #         #         # element_inner = [index for index in bitmaps_inner.keys() \
    #         #         #                  if unique_combinations_inner[index][i*(2*half+1)+j]]
    #         #         # element_outer = [index for index in bitmaps_outer.keys() \
    #         #         #                  if unique_combinations_outer[index][i * (2 * half + 1) + j]]
    #         #         if len(element_inner) and len(element_outer):
    #         #             start_time = time()
    #         #             cdf_inner = combine_vectors(element_inner, bitmaps_inner.copy())
    #         #             cdf_outer = combine_vectors(element_outer, bitmaps_outer.copy())
    #         #             # cdf_inner = combine_vectors(element_inner, bitmaps_inner)
    #         #             # cdf_outer = combine_vectors(element_outer, bitmaps_outer)
    #         #             time_count('Combine CDFs: in&out', time() - start_time)
    #         #             feature_vector = cdf_inner - cdf_outer
    #         #             time_count('Combine CDFs', time() - start_time)
    #         #             vectors_input.append(feature_vector)
    #         #             loc_input.append(np.array([j + half, i + half, k + half]))
    #
    #     # start_time = time()
    #     # xtest = xgb.DMatrix(vectors_input)
    #     # score = bst.predict(xtest, output_margin=True, ntree_limit=bst.best_ntree_limit)
    #     # loc_input = np.stack(loc_input)
    #     # xyz_shift_map[loc_input[:,0], loc_input[:,1], loc_input[:,2]] += score
    #     # time_count('Compute xgboost', time() - start_time)
    #     start_time = time()
    #     xtest = xgb.DMatrix(vectors_input)
    #     score = bst.predict(xtest, output_margin=True, ntree_limit=bst.best_ntree_limit)
    #     xyz_shift_map += score.reshape([2 * half + 1, 2 * half + 1, -1], order='F')
    #     time_count('Compute xgboost', time() - start_time)

    fn = savepath + structure + '.pkl'
    pickle.dump(xyz_shift_map, open(os.environ['ROOT_DIR'] + fn, 'wb'))
    time_count('Total', time() - t0)
    pickle.dump(time_log, open(os.environ['ROOT_DIR'] + savepath + structure + '_time.pkl', 'wb'))





