import argparse

parser = argparse.ArgumentParser()
parser.add_argument("structure", type=str, help="The name of structure")
args = parser.parse_args()
structure = args.structure

import cv2
import pickle
import numpy as np
import pandas as pd

import os
import sys
from time import time
from glob import glob
import mxnet as mx
from xgboost.sklearn import XGBClassifier
from sklearn import metrics
from lib.utils import run

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

def extract_cnn_features(stack, structure, batch_size, model):
    patch_dir = os.path.join('CSHL_patch_samples', stack)
    setup_download_from_s3(os.path.join(patch_dir, structure))
    setup_download_from_s3(os.path.join(patch_dir, structure + '_surround_500um_noclass'))
    patches = [dir for dir in glob(os.environ['ROOT_DIR'] + patch_dir + '/' + structure + '/*')]
    n_choose = min(len(patches), 1000)
    indices_choose = np.random.choice(range(len(patches)), n_choose, replace=False)
    patches = np.array(patches)
    patches_posi = patches[indices_choose]
    n_pos = len(patches_posi)

    patches = [dir for dir in glob(os.environ['ROOT_DIR'] + patch_dir + '/' + structure + '_surround_500um_noclass/*')]
    n_choose = min(len(patches), 1000)
    indices_choose = np.random.choice(range(len(patches)), n_choose, replace=False)
    patches = np.array(patches)
    patches_nega = patches[indices_choose]
    n_neg = len(patches_nega)

    patches_test = np.concatenate((patches_posi, patches_nega))
    data = (np.array([cv2.imread(dir, 0) for dir in patches_test if cv2.imread(dir, 0).shape == (224, 224)]) - mean_img)[:, None, :, :]
    del_pos = len([cv2.imread(dir, 0) for dir in patches_posi if cv2.imread(dir, 0).shape != (224, 224)])
    del_neg = len([cv2.imread(dir, 0) for dir in patches_nega if cv2.imread(dir, 0).shape != (224, 224)])
    labels = np.r_[np.ones((n_pos - del_pos,)), np.zeros((n_neg - del_neg,))]
    data_iter = mx.io.NDArrayIter(
        data=data,
        batch_size=batch_size,
        shuffle=False)
    outputs = model.predict(data_iter, always_output_list=True)
    features = outputs[0].asnumpy()

    return features, labels

MXNET_ROOTDIR = 'mxnet_models'
model_dir_name = 'inception-bn-blue'
model_name = 'inception-bn-blue'
setup_download_from_s3(os.path.join(MXNET_ROOTDIR, model_dir_name))
mean_img = np.load(os.path.join(os.environ['ROOT_DIR'], MXNET_ROOTDIR, model_dir_name, 'mean_224.npy'))
model_prefix = os.path.join(MXNET_ROOTDIR, model_dir_name, model_name)

batch_size = 32
model0, arg_params, aux_params = mx.model.load_checkpoint(os.path.join(os.environ['ROOT_DIR'],model_prefix), 0)
output_symbol_name='flatten_output'
flatten_output = model0.get_internals()[output_symbol_name]
model = mx.mod.Module(context=mx.cpu(), symbol=flatten_output)
model.bind(data_shapes=[('data', (batch_size,1,224,224))], for_training=False)
model.set_params(arg_params=arg_params, aux_params=aux_params, allow_missing=True)
train_features1, train_labels1 = extract_cnn_features('MD585', structure, batch_size, model)
train_features2, train_labels2 = extract_cnn_features('MD589', structure, batch_size, model)
train_features = np.concatenate((train_features1, train_features2))
train_labels = np.r_[train_labels1, train_labels2]
clf = XGBClassifier(max_depth=5, learning_rate=0.2, n_estimators=100,
                            silent=False, objective='binary:logistic')
clf.fit(train_features, train_labels)

test_features, test_labels = extract_cnn_features('MD594', structure, batch_size, model)
scores = clf.predict(test_features, output_margin=True)
pred = scores>0
acc = sum(pred ==test_labels)/len(test_labels)
fpr, tpr, threshold = metrics.roc_curve(test_labels, scores)
roc_auc = metrics.auc(fpr, tpr)

save_dir = os.path.join(os.environ['ROOT_DIR'], MXNET_ROOTDIR, 'inception-xgb_samples')
if not os.path.exists(save_dir):
    os.mkdir(save_dir)
prefix = os.path.join(MXNET_ROOTDIR, 'inception-xgb_samples', structure)
train_samples = {}
train_samples['features'] = train_features
train_samples['labels'] = train_labels
train_fp = '%s-train.pkl' % prefix
pickle.dump(train_samples, open(os.environ['ROOT_DIR'] + train_fp, 'wb'))
setup_upload_from_s3(train_fp, recursive=False)

test_samples = {}
test_samples['features'] = test_features
test_samples['labels'] = test_labels
test_samples['acc'] = acc
test_samples['auc'] = roc_auc
test_fp = '%s-test.pkl' % prefix
pickle.dump(test_samples, open(os.environ['ROOT_DIR'] + test_fp, 'wb'))
setup_upload_from_s3(test_fp, recursive=False)





