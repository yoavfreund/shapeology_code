import argparse

parser = argparse.ArgumentParser()
parser.add_argument("stack", type=str, help="The name of the stack")
parser.add_argument("section", type=int, help="The section number")
parser.add_argument("yaml", type=str, help="Path to Yaml file with parameters")
args = parser.parse_args()
stack = args.stack
section = args.section

import cv2
import numpy as np
import pandas as pd
import os
import sys
import sqlite3
sys.path.append(os.environ['REPO_DIR'])
from extractPatches import patch_extractor
from lib.utils import configuration, run

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

yamlfile=os.environ['REPO_DIR']+args.yaml
params=configuration(yamlfile).getParams()

extractor = patch_extractor(params)
images_all = pd.DataFrame()

fp = os.path.join('CSHL_data_processed', stack, stack + '_sorted_filenames.txt')
setup_download_from_s3(fp, recursive=False)
with open(os.environ['ROOT_DIR']+fp, 'r') as f:
    fn_idx_tuples = [line.strip().split() for line in f.readlines()]
    section_to_filename = {int(idx): fn for fn, idx in fn_idx_tuples}

db_dir = 'CSHL_databases/'
if not os.path.exists(os.environ['ROOT_DIR']+db_dir):
    os.mkdir(os.environ['ROOT_DIR']+db_dir)
db_dir += stack + '/'
if not os.path.exists(os.environ['ROOT_DIR']+db_dir):
    os.mkdir(os.environ['ROOT_DIR']+db_dir)
img_dir = 'CSHL_cells_images/'
if not os.path.exists(os.environ['ROOT_DIR']+img_dir):
    os.mkdir(os.environ['ROOT_DIR']+img_dir)
img_dir += stack + '/'
if not os.path.exists(os.environ['ROOT_DIR']+img_dir):
    os.mkdir(os.environ['ROOT_DIR']+img_dir)

raw_images_root = 'CSHL_data_processed/'+stack+'/'+stack+'_prep2_lossless_gray/'
img_fn = raw_images_root + section_to_filename[section] + '_prep2_lossless_gray.tif'
setup_download_from_s3(img_fn, recursive=False)

db_fp = db_dir+str(section)+'.db'
conn = sqlite3.connect(os.environ['ROOT_DIR']+db_fp)
cur = conn.cursor()

cur.execute('''CREATE TABLE Features 
               (section int, x int, y int,
                DMVec1 real, DMVec2 real, DMVec3 real, DMVec4 real, DMVec5 real,
                DMVec6 real, DMVec7 real, DMVec8 real, DMVec9 real, DMVec10 real,
                area int, height int, horiz_std real, mean real, padded_size int,
                rotation real, rotation_confidence real, std real, vert_std real, width int)''')

img = cv2.imread(os.environ['ROOT_DIR']+img_fn, 2)
m, n = img.shape
xs, ys = np.meshgrid(np.arange(0, n, 1000), np.arange(0, m, 1000), indexing='xy')
locations = np.c_[xs.flat, ys.flat]

for i in range(len(locations)):
    print(i, len(locations))
    left = locations[i][0]
    right = int(min(left + 1000, n))
    up = locations[i][1]
    down = int(min(up + 1000, m))
    tile = img[up:down, left:right]
    if params['preprocessing']['polarity'] == -1:
        tile = 255 - tile
    min_std = params['preprocessing']['min_std']
    _std = np.std(tile.flatten())
    if _std < min_std:
        continue
    else:
        try:
            Stats = extractor.segment_cells(tile)
            cells = extractor.extract_blobs(Stats, tile)
            cells = pd.DataFrame(cells)
            cells = cells[cells['padded_patch'].notnull()]
            cells['section'] = section
            cells['x'] = cells['left'] + left + cells['width'] / 2
            cells['y'] = cells['top'] + up + cells['height'] / 2
            cells = cells.astype({'x': int, 'y': int})
            images = cells[['section', 'x', 'y', 'padded_patch']]
            images_all = pd.concat([images_all, images], ignore_index=True)
            cells = cells.drop(['padded_patch', 'left', 'top'], 1)
            cells = np.asarray(cells)
            for k in range(len(cells)):
                cells[k][0] = cells[k][0][:10]
            features = np.concatenate((cells[:, -3:], np.array(list(cells[:, 0])), cells[:, 1:-3]), axis=1)
            cur.executemany('INSERT INTO Features VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)', features)
            conn.commit()
        except:
            continue
conn.close()

img_out_file = img_dir + str(section) + '.pkl'
images_all.to_pickle(os.environ['ROOT_DIR']+img_out_file)
setup_upload_from_s3(img_out_file, recursive=False)
setup_upload_from_s3(db_fp, recursive=False)

os.remove(os.environ['ROOT_DIR']+img_fn)
os.remove(os.environ['ROOT_DIR']+img_out_file)
