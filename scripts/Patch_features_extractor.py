import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("stack", type=str, help="The name of the stack")
parser.add_argument("filename", type=str, help="Path to patch file")
parser.add_argument("--yaml", type=str, default=os.path.join(os.environ['REPO_DIR'], 'shape_params.yaml'),
                    help="Path to Yaml file with parameters")
args = parser.parse_args()
stack = args.stack

import cv2
import pickle
import numpy as np
import pandas as pd

import sys
from time import time
from glob import glob
sys.path.append(os.environ['REPO_DIR'])
from extractPatches import patch_extractor
from lib.utils import mark_contours, configuration, run

def CDF(x):
    x = np.sort(x)
    size = x.shape[0]
    y = np.arange(0, size) / size
    return x, y


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


def generator(structure, state, threshold, cell_dir, patch_dir, stack):
    for state in [state]:
        t1 = time()
        savepath = cell_dir + structure + '/'
        pkl_out_file = savepath + stack + '_' + structure + '_' + state + '.pkl'
        cell_out_file = savepath + stack + '_' + structure + '_' + state + '_cells.pkl'
        if os.path.exists(os.environ['ROOT_DIR'] + pkl_out_file):
            print(structure + '_' + state + ' ALREADY EXIST')
            continue
        else:
            if not os.path.exists(os.environ['ROOT_DIR'] + savepath):
                os.makedirs(os.environ['ROOT_DIR'] + savepath)

        if structure == '7nn':
            structure = '7n'

        if state == 'positive':
            setup_download_from_s3(patch_dir + structure)
            patches = [dir for dir in glob(os.environ['ROOT_DIR'] + patch_dir + structure + '/*')]
        else:
            setup_download_from_s3(patch_dir + structure + '_surround_500um_noclass')
            patches = [dir for dir in
                       glob(os.environ['ROOT_DIR'] + patch_dir + structure + '_surround_500um_noclass/*')]

        features = []
        # cell_features = []

        n_choose = min(len(patches), 1000)
        indices_choose = np.random.choice(range(len(patches)), n_choose, replace=False)
        patches = np.array(patches)
        patches = patches[indices_choose]

        for i in range(len(patches)):
            tile = cv2.imread(patches[i], 0)

            if params['preprocessing']['polarity'] == -1:
                tile = 255 - tile
            min_std = params['preprocessing']['min_std']
            _std = np.std(tile.flatten())

            extracted = []
            if _std < min_std:
                print('image', patches[i], 'std=', _std, 'too blank')
                # features.append([0] * 1982)
                features.append([0] * 5942)
            else:
                try:
                    Stats = extractor.segment_cells(tile)
                    cells = extractor.extract_blobs(Stats, tile)
                    cells = pd.DataFrame(cells)
                    cells = cells[cells['padded_patch'].notnull()]
                    cells = cells.drop(['padded_patch', 'left', 'top'], 1)
                    cells = np.asarray(cells)
                    for k in range(len(cells)):
                        cells[k][10] = cells[k][10][:10]
                        if stack != 'DK39':
                            M = transform[cells[k][3]]['M']
                            miu = transform[cells[k][3]]['miu']
                            cells[k][10] = np.dot(cells[k][10], M) + miu
                    origin = np.concatenate((np.array(list(cells[:, 10])).reshape([-1,10]), cells[:, 0:10]), axis=1)
                    # cell_features.append(origin)
                    for k in range(origin.shape[1]):
                        ten = []
                        x, y = CDF(origin[origin[:,13]==15, k])
                        ten.extend([y[np.argmin(np.absolute(x - threshold[k][j]))] for j in range(99)])
                        x, y = CDF(origin[origin[:, 13] == 51, k])
                        ten.extend([y[np.argmin(np.absolute(x - threshold[k][99+j]))] for j in range(99)])
                        x, y = CDF(origin[origin[:, 13] == 201, k])
                        ten.extend([y[np.argmin(np.absolute(x - threshold[k][198+j]))] for j in range(99)])
                        # x, y = CDF(origin[:, k])
                        # ten = [y[np.argmin(np.absolute(x - threshold[k][j]))] for j in range(99)]
                        extracted.extend(ten)
                    extracted.extend([cells.shape[0]])
                    extracted.extend([origin[:, 12].sum() / (317 * 317)])
                    features.append(extracted)
                except:
                    continue
            if i % 10 == 0:
                count = len(features)
                print(structure + '_' + state, count, i, '/', len(patches))

        count = len(features)
        print(structure + '_' + state, count)
        pickle.dump(features, open(os.environ['ROOT_DIR'] + pkl_out_file, 'wb'))
        # pickle.dump(cell_features, open(os.environ['ROOT_DIR'] + cell_out_file, 'wb'))
        # setup_upload_from_s3(pkl_out_file, recursive=False)
        print(structure + '_' + state + ' finished in %5.1f seconds' % (time() - t1))


yamlfile=args.yaml
params=configuration(yamlfile).getParams()
extractor = patch_extractor(params, dm=True, stem=os.path.join(os.environ['ROOT_DIR'], 'diffusionmap', stack, 'diffusionMap'))
if stack!='DK39':
    transform = pickle.load(open(os.path.join(os.environ['ROOT_DIR'], 'diffusionmap', stack, 'transform.pkl'),'rb'))

fn = 'CSHL_data_processed/MD589/ThresholdsV2.pkl'
setup_download_from_s3(fn, recursive=False)
thresholds = pickle.load(open(os.environ['ROOT_DIR'] + fn, 'rb'))

patch_dir = args.filename + '/' + stack + '/'
cell_dir = os.environ['ROOT_DIR'] + args.filename + '_features/'
cell_dir = cell_dir + stack + '/'
if not os.path.exists(cell_dir):
    os.makedirs(cell_dir)

cell_dir = args.filename + '_features/' + stack + '/'

paired_structures = ['5N', '6N', '7N', '7n', 'Amb', 'LC', 'LRt', 'Pn', 'Tz', 'VLL', 'RMC', \
                     'SNC', 'SNR', '3N', '4N', 'Sp5I', 'Sp5O', 'Sp5C', 'PBG', '10N', 'VCA', 'VCP', 'DC']
singular_structures = ['AP', '12N', 'RtTg', 'SC', 'IC']

t0 = time()
all_structures = paired_structures + singular_structures
for struc in all_structures:
    for state in ['positive','negative']:
        generator(struc, state, thresholds, cell_dir, patch_dir, stack)

print('Finished in %5.1f seconds'%(time()-t0))

