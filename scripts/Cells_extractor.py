import cv2
from skimage import io
import numpy as np
import pickle
import os
import sys
import shutil
from scipy import stats
from time import time
from glob import glob
from time import sleep
sys.path.append(os.environ['REPO_DIR'])
from lib.utils import configuration, run
from lib.shape_utils import *


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

def setup_upload_from_s3(s3_fp, local_fp, recursive=True):
    s3_fp = 's3://mousebrainatlas-data/' + s3_fp
    local_fp = os.environ['ROOT_DIR'] + local_fp

    if recursive:
        run('aws s3 cp --recursive {0} {1}'.format(local_fp, s3_fp))
    else:
        run('aws s3 cp {0} {1}'.format(local_fp, s3_fp))

def compute(file, yaml_file, stack, section, client, bucket, key):
    t0 = time()
    img_fn = file
    setup_download_from_s3(img_fn, recursive=False)
    run('python {0}/extractPatches.py {1} {2}'.format(os.environ['REPO_DIR'], img_fn, yaml_file))
    params = configuration(yaml_file).getParams()
    size_thresholds = params['normalization']['size_thresholds']
    local_dir = file[:-4] + '_cells/'
    for size in size_thresholds:
        key_item = 'size_of_' + str(size)
        local_fp = local_dir + str(size) + '.bin'
        s3_fp = stack + '/cells/' + str(section) + '_cells/' + str(size) + '.bin'

        def s3_exist(s3_fp):
            try:
                report = client.stat_object(bucket, s3_fp)
                return True
            except:
                return False

        while not s3_exist(s3_fp):
            setup_upload_from_s3(s3_fp, local_fp, recursive=False)
        report = client.stat_object(bucket, s3_fp)
        key[key_item] = int(report.size / 1000)
        os.remove(os.environ['ROOT_DIR'] + local_fp)
    print(file, 'finished in', time()-t0, 'seconds')
    os.remove(os.environ['ROOT_DIR']+img_fn)
    return key

if __name__=="__main__":

    import argparse
    from time import time
    parser = argparse.ArgumentParser()
    parser.add_argument("image_dir", type=str,
                        help="Path to directory saving images")
    parser.add_argument("yaml", type=str,
                    help="Path to Yaml file with parameters")
    args = parser.parse_args()
    yamlfile = args.yaml
    img_dir = args.image_dir
    t0 = time()
    for img in glob(img_dir+'/*'):
        run('python {0}/extractPatches.py {1} {2}'.format(os.environ['REPO_DIR'], img, yamlfile))
    print('Cell extraction finished in', time()-t0, 'seconds')
