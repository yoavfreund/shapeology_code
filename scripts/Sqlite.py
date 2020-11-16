import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("stack", type=str, help="The name of the brain")
parser.add_argument("file", type=str, help="The path to the image file")
parser.add_argument("--yaml", type=str, default=os.path.join(os.environ['REPO_DIR'], 'shape_params.yaml'),
                    help="Path to Yaml file with parameters")
args = parser.parse_args()
stack = args.stack

from skimage import io
import numpy as np
import pandas as pd
import sys
import sqlite3
import pickle
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

yamlfile=args.yaml
params=configuration(yamlfile).getParams()
# extractor = patch_extractor(params, dm=False)
extractor = patch_extractor(params, dm=True, stem=os.path.join(os.environ['ROOT_DIR'], 'diffusionmap', stack, 'diffusionMap'))
images_all = pd.DataFrame()
if stack!='DK39':
    transform = pickle.load(open(os.path.join(os.environ['ROOT_DIR'], 'diffusionmap', stack, 'transform.pkl'),'rb'))

db_dir = 'CSHL_databases/'
if not os.path.exists(os.environ['ROOT_DIR']+db_dir):
    os.mkdir(os.environ['ROOT_DIR']+db_dir)
db_dir += stack + '/'
if not os.path.exists(os.environ['ROOT_DIR']+db_dir):
    os.mkdir(os.environ['ROOT_DIR']+db_dir)
# img_dir = 'CSHL_cells_images/'
# if not os.path.exists(os.environ['ROOT_DIR']+img_dir):
#     os.mkdir(os.environ['ROOT_DIR']+img_dir)
# img_dir += stack + '/'
# if not os.path.exists(os.environ['ROOT_DIR']+img_dir):
#     os.mkdir(os.environ['ROOT_DIR']+img_dir)

img_fn = args.file
dot = img_fn.rfind('.')
slash = img_fn.rfind('/')
# setup_download_from_s3(img_fn, recursive=False)
section = img_fn[slash+1:dot]
db_fp = os.path.join(db_dir, section + '.db')
conn = sqlite3.connect(os.path.join(os.environ['ROOT_DIR'], db_fp))
cur = conn.cursor()

cur.execute('''CREATE TABLE Features 
               (section int, x int, y int,
                DMVec1 real, DMVec2 real, DMVec3 real, DMVec4 real, DMVec5 real,
                DMVec6 real, DMVec7 real, DMVec8 real, DMVec9 real, DMVec10 real,
                area int, height int, horiz_std real, mean real, padded_size int,
                rotation real, rotation_confidence real, std real, vert_std real, width int)''')

# cur.execute('''CREATE TABLE Features
#                (section int, x int, y int,
#                 area int, height int, horiz_std real, mean real, padded_size int,
#                 rotation real, rotation_confidence real, std real, vert_std real, width int)''')
img = io.imread(img_fn)
m, n = img.shape

if params['preprocessing']['polarity'] == -1:
    tile = 255 - img.copy()
min_std = params['preprocessing']['min_std']
_std = np.std(tile.flatten())
if _std < min_std:
    print('image', img_fn, 'std=', _std, 'too blank, skipping')
else:
    Stats = extractor.segment_cells(tile)
    cells = extractor.extract_blobs(Stats, tile)
    cells = pd.DataFrame(cells)
    cells = cells[cells['padded_patch'].notnull()]
    cells['section'] = int(section)
    cells['x'] = cells['left'] + cells['width'] / 2
    cells['y'] = cells['top'] + cells['height'] / 2
    cells = cells.astype({'x': int, 'y': int})
    images = cells[['section', 'x', 'y', 'padded_patch']]
    images_all = pd.concat([images_all, images], ignore_index=True)
    cells = cells.drop(['padded_patch', 'left', 'top'], 1)
    cells = np.asarray(cells)
    for k in range(len(cells)):
        cells[k][10] = cells[k][10][:10]
        if stack!='DK39':
            M = transform[cells[k][3]]['M']
            miu = transform[cells[k][3]]['miu']
            cells[k][10] = np.dot(cells[k][10],M)+miu
#     features = np.concatenate((cells[:, -3:], cells[:, 0:10]), axis=1)
#     cur.executemany('INSERT INTO Features VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)', features)
    features = np.concatenate((cells[:, -3:], np.array(list(cells[:, 10])).reshape([-1,10]), cells[:, 0:10]), axis=1)
    cur.executemany('INSERT INTO Features VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)', features)
    conn.commit()

conn.close()
# setup_upload_from_s3(db_fp, recursive=False)

# os.remove(os.environ['ROOT_DIR']+img_fn)
