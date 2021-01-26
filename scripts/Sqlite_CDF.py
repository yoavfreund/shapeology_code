import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("stack", type=str, help="The name of the brain")
parser.add_argument("file", type=str, help="The path to the image file")
args = parser.parse_args()
stack = args.stack

from skimage import io as sio
import numpy as np
import pandas as pd
import sys
import sqlite3
import io
import pickle
sys.path.append(os.environ['REPO_DIR'])
from lib.utils import configuration, run

def CDF(x):
    x=np.sort(x)
    size=x.shape[0]
    y=np.arange(0,size)/size
    return x,y

def adapt_array(arr):
    """
    http://stackoverflow.com/a/31312102/190597 (SoulNibbler)
    """
    out = io.BytesIO()
    np.save(out, arr)
    out.seek(0)
    return sqlite3.Binary(out.read())

def convert_array(text):
    out = io.BytesIO(text)
    out.seek(0)
    return np.load(out)

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

db_dir = 'CSHL_databases/'
if not os.path.exists(os.environ['ROOT_DIR']+db_dir):
    os.mkdir(os.environ['ROOT_DIR']+db_dir)
cell_dir = db_dir + stack + '/'
db_dir += stack + '_cdf/'
if not os.path.exists(os.environ['ROOT_DIR']+db_dir):
    os.mkdir(os.environ['ROOT_DIR']+db_dir)

fn = 'CSHL_data_processed/MD589/ThresholdsV2.pkl'
thresholds = pickle.load(open(os.environ['ROOT_DIR']+fn,'rb'))

sqlite3.register_adapter(np.ndarray, adapt_array)
sqlite3.register_converter("array", convert_array)

img_fn = args.file
dot = img_fn.rfind('.')
slash = img_fn.rfind('/')
# setup_download_from_s3(img_fn, recursive=False)

section = img_fn[slash+1:dot]

cell_fp = os.path.join(cell_dir, section + '.db')
conn_cell = sqlite3.connect(os.path.join(os.environ['ROOT_DIR'], cell_fp))
cur_cell = conn_cell.cursor()

raws = cur_cell.execute('SELECT * FROM features')
info = np.array(list(raws))
locations = info[:, 1:3]
features = info[:, 3:]

db_fp = os.path.join(db_dir, section + '.db')
conn = sqlite3.connect(os.path.join(os.environ['ROOT_DIR'], db_fp))
cur = conn.cursor()

cur.execute('''CREATE TABLE Features 
               (section integer, x integer, y integer,
                cdfs array)''')

img = sio.imread(img_fn)
m, n = img.shape
stride = 317
xs, ys = np.meshgrid(np.arange(0, n, stride), np.arange(0, m, stride), indexing='xy')
windows = np.c_[xs.flat, ys.flat]

for i in range(len(windows)):
    if i % 1000 == 0:
        print(i, len(windows))
    left = windows[i][0]
    up = windows[i][1]

    indices_window = np.where((locations[:, 0] > left) & (locations[:, 0] < left + stride) \
                              & (locations[:, 1] > up) & (locations[:, 1] < up + stride))[0]
    if len(indices_window):
        cells = features[indices_window]
        cdf = []
        for k in range(cells.shape[1]):
            x, y = CDF(cells[:, k])
            ten = [y[np.argmin(np.absolute(x - thresholds[k][j]))] for j in range(99)]
            cdf.extend(ten)
        cdf.extend([cells.shape[0]])
        cdf.extend([cells[:, 12].sum() / (317 * 317)])

        cur.execute('INSERT INTO Features VALUES (?,?,?,?)', (section, left+158, up+158, np.array(cdf)))
        conn.commit()

conn.close()
# setup_upload_from_s3(db_fp, recursive=False)

# os.remove(os.environ['ROOT_DIR']+img_fn)
