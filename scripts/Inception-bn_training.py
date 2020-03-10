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
import logging
from mxnet.model import save_checkpoint
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

def get_fine_tune_model(symbol, arg_params, num_classes, layer_name='flatten'):
    """
    symbol: the pretrained network symbol
    arg_params: the argument parameters of the pretrained model
    num_classes: the number of classes for the fine-tune datasets
    layer_name: the layer name before the last fully-connected layer
    """
    all_layers = symbol.get_internals()
    net = all_layers[layer_name + '_output']
    net = mx.symbol.FullyConnected(data=net, num_hidden=num_classes, name='fc1')
    net = mx.symbol.SoftmaxOutput(data=net, name='softmax')
    new_args = {k: arg_params[k] for k in arg_params if 'fc1' not in k}
    return (net, new_args)

def fit(symbol, arg_params, aux_params, train, val, batch_size, num_epoch, epoch_end_callback,
       eval_end_callback):
#     devs = [mx.gpu(i) for i in range(num_gpus)]
    mod = mx.mod.Module(symbol=symbol, context=mx.cpu())
    mod.fit(train, val,
        num_epoch=num_epoch,
        arg_params=arg_params,
        aux_params=aux_params,
        allow_missing=True,
        batch_end_callback = mx.callback.Speedometer(batch_size, 10),
        kvstore='local',
        optimizer='sgd',
        optimizer_params={'learning_rate':0.01},
        initializer=mx.init.Xavier(rnd_type='gaussian', factor_type="in", magnitude=2),
        eval_metric='acc',
           epoch_end_callback=epoch_end_callback,
            eval_end_callback=eval_end_callback
           )
    metric = mx.metric.Accuracy()
    return mod.score(val, metric)

def my_epoch_end_callback(prefix, period=1):
    def _callback(epoch, sym, arg, aux):
        if epoch % period == 4:
            save_checkpoint(os.path.join(os.environ['ROOT_DIR'], prefix), epoch, sym, arg, aux)
            symbol_fp = '%s-symbol.json' % prefix
            param_fp = '%s-%04d.params' % (prefix, epoch)
            setup_upload_from_s3(symbol_fp, recursive=False)
            setup_upload_from_s3(param_fp, recursive=False)
    return _callback

def my_eval_end_callback(eval_metric_history):
    """
    Args:
        eval_metric_history (dict): {name: list of values}
    """
    def _callback(param):
        if not param.eval_metric:
            return
        name_value = param.eval_metric.get_name_value()
        for name, value in name_value:
            logging.info('Epoch[%d] Validation-%s=%f', param.epoch, name, value)
            if name not in eval_metric_history:
                eval_metric_history[name] = []
            eval_metric_history[name].append(value)
#             with open(acc_fp, 'a') as f:
#                 f.write('Epoch[%d] Validation-%s=%f\n' % (param.epoch, name, value))
    return _callback

MXNET_ROOTDIR = 'mxnet_models'
model_dir_name = 'inception-bn-blue-softmax'
model_name = 'inception-bn-blue-softmax'
setup_download_from_s3(os.path.join(MXNET_ROOTDIR, model_dir_name, 'mean_224.npy'), recursive=False)
mean_img = np.load(os.path.join(os.environ['ROOT_DIR'], MXNET_ROOTDIR, model_dir_name, 'mean_224.npy'))

model_prefix = os.path.join(MXNET_ROOTDIR, model_dir_name, model_name)
setup_download_from_s3(model_prefix+'-symbol.json', recursive=False)
setup_download_from_s3(model_prefix+'-0000.params', recursive=False)

param_dir = os.path.join(os.environ['ROOT_DIR'], MXNET_ROOTDIR, model_dir_name + '_Kui')
if not os.path.exists(param_dir):
    os.mkdir(param_dir)

patch_dir1 = os.path.join('CSHL_patch_samples','MD585')
patch_dir2 = os.path.join('CSHL_patch_samples','MD589')

setup_download_from_s3(os.path.join(patch_dir1,structure))
setup_download_from_s3(os.path.join(patch_dir1,structure+'_surround_500um_noclass'))
setup_download_from_s3(os.path.join(patch_dir2,structure))
setup_download_from_s3(os.path.join(patch_dir2,structure+'_surround_500um_noclass'))
patches = [dir for dir in glob(os.environ['ROOT_DIR']+patch_dir1+'/'+structure+'/*')]
n_choose = min(len(patches), 1000)
indices_choose = np.random.choice(range(len(patches)), n_choose, replace=False)
patches = np.array(patches)
patches_posi = patches[indices_choose]
patches = [dir for dir in glob(os.environ['ROOT_DIR']+patch_dir2+'/'+structure+'/*')]
n_choose = min(len(patches), 1000)
indices_choose = np.random.choice(range(len(patches)), n_choose, replace=False)
patches = np.array(patches)
patches_posi = np.concatenate((patches_posi, patches[indices_choose]))
n_pos = len(patches_posi)

patches = ([dir for dir in glob(os.environ['ROOT_DIR']+patch_dir1+'/'+structure+'_surround_500um_noclass/*')])
n_choose = min(len(patches), 1000)
indices_choose = np.random.choice(range(len(patches)), n_choose, replace=False)
patches = np.array(patches)
patches_nega = patches[indices_choose]
patches = ([dir for dir in glob(os.environ['ROOT_DIR']+patch_dir2+'/'+structure+'_surround_500um_noclass/*')])
n_choose = min(len(patches), 1000)
indices_choose = np.random.choice(range(len(patches)), n_choose, replace=False)
patches = np.array(patches)
patches_nega = np.concatenate((patches_nega, patches[indices_choose]))
n_neg = len(patches_nega)

patches_train = np.concatenate((patches_posi, patches_nega))
batch_size = 32
train_data = (np.array([cv2.imread(dir, 0) for dir in patches_train]) - mean_img)[:,None,:,:]
train_labels = np.r_[np.ones((n_pos, )), np.zeros((n_neg, ))]
train_data_iter = mx.io.NDArrayIter(
    data=train_data,
    batch_size=batch_size,
    label=train_labels,
    shuffle=True)

patch_dir = os.path.join('CSHL_patch_samples','MD594')
setup_download_from_s3(os.path.join(patch_dir,structure))
setup_download_from_s3(os.path.join(patch_dir,structure+'_surround_500um_noclass'))
patches = [dir for dir in glob(os.environ['ROOT_DIR']+patch_dir+'/'+structure+'/*')]
n_choose = min(len(patches), 1000)
indices_choose = np.random.choice(range(len(patches)), n_choose, replace=False)
patches = np.array(patches)
patches_posi = patches[indices_choose]
n_pos = len(patches_posi)

patches = ([dir for dir in glob(os.environ['ROOT_DIR']+patch_dir+'/'+structure+'_surround_500um_noclass/*')])
n_choose = min(len(patches), 1000)
indices_choose = np.random.choice(range(len(patches)), n_choose, replace=False)
patches = np.array(patches)
patches_nega = patches[indices_choose]
n_neg = len(patches_nega)

patches_test = np.concatenate((patches_posi, patches_nega))
test_data = (np.array([cv2.imread(dir, 0) for dir in patches_test]) - mean_img)[:,None,:,:]
test_labels = np.r_[np.ones((n_pos, )), np.zeros((n_neg, ))]
test_data_iter = mx.io.NDArrayIter(
    data=test_data,
    batch_size=batch_size,
    label=test_labels,
    shuffle=False)

t1 = time()
model0, arg_params, aux_params = mx.model.load_checkpoint(os.path.join(os.environ['ROOT_DIR'],model_prefix), 0)
num_classes = 2
(new_sym, new_args) = get_fine_tune_model(model0, arg_params, num_classes)
eval_metric_history = {}
prefix = os.path.join(MXNET_ROOTDIR, model_dir_name + '_Kui', model_name + '_' + structure)
mod_score = fit(new_sym, new_args, aux_params, train_data_iter, test_data_iter,
                batch_size, num_epoch=50,
                epoch_end_callback=my_epoch_end_callback(prefix, period=5),
               eval_end_callback=my_eval_end_callback(eval_metric_history))
pickle.dump(eval_metric_history, open(os.environ['ROOT_DIR'] + prefix + '_evalMetricHistory.pkl', 'wb'))
setup_upload_from_s3(prefix+'_evalMetricHistory.pkl', recursive=False)

print(structure + ' finished in %5.1f seconds' % (time() - t1))

