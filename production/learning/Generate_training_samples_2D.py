import numpy as np
import pickle
from time import time
import os
import sqlite3
from skimage import measure
from shapely.geometry import Polygon
from matplotlib.path import Path

def create_annotation(mask):
    contours = measure.find_contours(mask, 0.5, positive_orientation='low')

    size = [len(contour) for contour in contours]
    contour = contours[np.argmax(np.array(size))]
    # Flip from (row, col) representation to (x, y)
    # and subtract the padding pixel
    for i in range(len(contour)):
        row, col = contour[i]
        contour[i] = (col, row)

    # Make a polygon and simplify it
    poly = Polygon(contour)
#     if len(contour)>100:
#         poly = poly.simplify(5.0, preserve_topology=False)
#     else:
    poly = poly.simplify(0.5, preserve_topology=False)
    segmentation = np.array(poly.exterior.coords)#.ravel().tolist()
    return segmentation

def COMs_to_contours(center,structure):
    contours = {}
    contour = np.load('/net/birdstore/Active_Atlas_Data/data_root/atlas_data/atlasV7/structure/'+structure+'.npy')
    idx = np.indices(contour.shape).reshape(contour.ndim, contour.size)
    weights = contour.reshape(1,-1)
    com = np.average(idx, axis = 1, weights=weights[0,:])
    start = int(center[2]-com[2])
    for i in range(contour.shape[2]):
        try:
            polygon = create_annotation(contour[:,:,i])
        except:
            continue
        polygon[:,0] = center[0]+(polygon[:,0]-com[1])*32*1.4154
        polygon[:,1] = center[1]+(polygon[:,1]-com[0])*32*1.4154
        contours[start+i] = polygon
    return contours

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

def features_to_vector(features, thresholds, object_area):
    extracted = []
    n1 = features.shape[0]
#     for k in list(range(10)) + [14]:
    for k in range(features.shape[1]): #iterate over features, one feature equal one cdf
        data1 = np.sort(features[:, k]) # sort feature k
        cdf = np.searchsorted(data1, thresholds[k], side='right') / n1
        extracted.extend(cdf)
    extracted.extend([features.shape[0] / object_area * 317 * 317]) # cell number normalized by region area
    extracted.extend([features[:, 12].sum() / object_area]) # cell areas in total normalized by region area
    return extracted

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

    resol = 0.325
    margin = 200 / resol
    fn = os.environ['ROOT_DIR'] + 'Detection_preparation_v2/' + structure+'.pkl'
    grid3D, total_shape_area, total_sur_area, min_x, min_y, len_max = pickle.load(open(fn, 'rb'))
    # grid3D, total_shape_area, min_x, min_y, len_max = pickle.load(open(fn,'rb'))
    step_size = max(int(len_max / 30), int(20 / resol))
    step_z = int(step_size * resol / 20)


    half = 15
    fn = stack + '/' + stack + '_beth_COMs_new.pkl'
    COMs = pickle.load(open(os.environ['ROOT_DIR'] + fn, 'rb'))
    centroid = COMs[structure]
    contours = COMs_to_contours(centroid,structure)
    seq = sorted(list(contours.keys()))

    Concat = np.concatenate([contours[i] for i in contours])
    center = np.mean(Concat, axis=0)

    polygon_set = []
    for sec in seq:
        polygon = contours[sec].copy()
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
    for section in seq:
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
    positive_sample_features = []
    negative_sample_features = []
    for i in range(-half, half + 1):
        for j in range(-half, half + 1):
            # Collect cells in the shifted 3D shapes
            start_time = time()

            loc = np.array([seq[0], center[0] + min_x-margin + i * step_size, center[1] + min_y-margin + j * step_size])
            inside_shape_features, sur_shape_features, inside_coord, sur_coord = collect_inside_cell_features(loc)
            for sec in seq:
                inside_features = inside_shape_features[inside_coord[:, 0] == sec - seq[0]]
                outside_features = sur_shape_features[sur_coord[:, 0] == sec - seq[0]]
                if inside_features.shape[0] and outside_features.shape[0]:
                    inside_cdf = features_to_vector(inside_features, thresholds, total_shape_area[sec - seq[0]])
                    sur_cdf = features_to_vector(outside_features, thresholds, total_sur_area[sec - seq[0]])
                    feature_vector = np.array(inside_cdf) - np.array(sur_cdf)

                    polygon = contours[sec].copy()
                    outline = Path(polygon)
                    region = polygon.copy()
                    region[:, 0] += i * step_size
                    region[:, 1] += j * step_size
                    if outline.contains_point(list(Polygon(region).centroid.coords)[0]):
                        positive_sample_features.append(feature_vector)
                    else:
                        negative_sample_features.append(feature_vector)
    print(structure, len(positive_sample_features), len(negative_sample_features))

    rname = structure
    if structure.rfind('_') != -1:
        rname = structure[:structure.rfind('_')]
    n_choose = min(int(len(positive_sample_features) * 0.5), 1000)
    indices_choose = np.random.choice(range(len(positive_sample_features)), n_choose, replace=False)
    positive_sample_features = np.array(positive_sample_features)
    positive_sample_features = positive_sample_features[indices_choose]
    save_dir = os.environ['ROOT_DIR'] + 'CSHL_patch_samples_features_v5_extra/' + stack + '/' + rname
    # save_dir = os.environ['ROOT_DIR'] + 'CSHL_patch_samples_features_dm_only/' + stack + '/' + rname
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    pkl_out_file = save_dir + '/' + stack + '_' + structure + '_positive.pkl'
    pickle.dump(positive_sample_features, open(pkl_out_file, 'wb'))

    n_choose = min(int(len(negative_sample_features) * 0.5), 1000)
    indices_choose = np.random.choice(range(len(negative_sample_features)), n_choose, replace=False)
    negative_sample_features = np.array(negative_sample_features)
    negative_sample_features = negative_sample_features[indices_choose]

    pkl_out_file = save_dir + '/' + stack + '_' + structure + '_negative.pkl'
    pickle.dump(negative_sample_features, open(pkl_out_file, 'wb'))
