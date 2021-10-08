import pickle
import os
import numpy as np
from scipy import signal

resol = 0.325


def calcAutoCorrelation(sig, border=12):
    # normalize mean and std
    nsig = sig - np.mean(sig.flatten())
    nsig = nsig / np.std(nsig.flatten())
    small = nsig[border:-border, border:-border, border:-border]
    corr = signal.correlate(nsig, small, mode='same')
    return corr, np.std(corr.flatten()), np.max(corr.flatten())


def save_corrected_coms(contours, fp, fn1, fn2, COMs):
    detection_coms = {}
    detection_maps = {}
    for structure in COMs.keys():

        filename = os.environ['ROOT_DIR'] + 'CSHL_shift_scores/' + fp + structure + '.pkl'
        if not os.path.exists(filename):
            continue
        scores_total = pickle.load(open(filename, 'rb'))
        mean, std = scores_total.mean(), scores_total.std()
        scores_total = (scores_total - mean) / std
        detection_maps[structure] = scores_total

        corr, _std, _max = calcAutoCorrelation(scores_total)

        filename = os.environ['ROOT_DIR'] + 'Detection_preparation_v2/' + structure + '.pkl'
        grid3D, total_shape_area, total_sur_area, min_x, min_y, len_max = pickle.load(open(filename, 'rb'))

        step_size = max(int(len_max / 20), int(30 / resol))
        step_z = int(step_size * resol / 20)
        middle = int(scores_total.shape[0] / 2)

        maxim = []
        for x in range(scores_total.shape[0]):
            for y in range(scores_total.shape[1]):
                for z in range(scores_total.shape[2]):
                    if corr[x, y, z] == corr.max():
                        maxim.append([x, y, z])
        x_max, y_max, z_max = maxim[0]
        print(x_max, y_max, z_max, structure, _max)
        detection_coms[structure] = COMs[structure].copy()
        detection_coms[structure][0] = COMs[structure][0] + (y_max - middle) * step_size
        detection_coms[structure][1] = COMs[structure][1] + (x_max - middle) * step_size
        detection_coms[structure][2] = COMs[structure][2] + (z_max - middle) * step_z

    fn = os.environ['ROOT_DIR'] + fn1
    pickle.dump(detection_coms, open(fn, 'wb'))
    fn = os.environ['ROOT_DIR'] + fn2
    np.savez_compressed(fn, **detection_maps)

if __name__=='__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=str, default='_cdf_3D/',
                        help="directory of saved detection maps")
    parser.add_argument("--com", type=str, default='_correct_coms_v4.pkl',
                        help="directory for saving detected coms")
    parser.add_argument("--maps", type=str, default='_v7.npz/',
                        help="directory for saving normalized score maps")
    args = parser.parse_args()
    src_path = args.src
    com_path = args.com
    map_path = args.maps

    for stack in ['DK43', 'DK41', 'DK39', 'DK55', 'DK52', 'DK63', 'DK46', 'DK54', 'DK61', 'DK62']:
        print(stack)
        fn = stack + '/' + stack + '_affine_COMs.pkl' if stack != 'DK52' else stack + '/' + stack + '_beth_COMs.pkl'
        COMs = pickle.load(open(os.environ['ROOT_DIR'] + fn, 'rb'))
        fn = stack + '/' + stack + '_rough_landmarks.pkl'
        contours = pickle.load(open(os.environ['ROOT_DIR'] + fn, 'rb'))
        save_corrected_coms(contours, stack + src_path, stack + com_path,
                            '/DetectionAnalysis/' + stack + map_path, COMs)
