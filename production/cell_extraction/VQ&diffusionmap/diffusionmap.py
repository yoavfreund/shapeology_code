import numpy
from matplotlib import pylab, mlab, pyplot
np = numpy
plt = pyplot

from IPython.display import display
from IPython.core.pylabtools import figsize, getfigs

from pylab import *
from numpy import *
import sys
import os
sys.path.append('../scripts/')
from lib.shape_utils import *

import dill
import pickle
from glob import glob
from time import time

from pydiffmap import diffusion_map as dm
# initialize Diffusion map object.
neighbor_params = {'n_jobs': -1, 'algorithm': 'ball_tree'}


def sample_and_mapping(vq_name, dm_name, size, knear=100, eps=2500, train=True):
    vq = pickle.load(open(vq_name, 'rb'))
    Reps, Reps_count = vq[size]
    Reps_mat = pack_pics(Reps)
    data1D=np.concatenate([x.reshape([1,size*size]) for x in Reps_mat])
    if train==True:
        mydmap = dm.DiffusionMap.from_sklearn(n_evecs=100, k=knear, epsilon=eps, alpha=1.0, neighbor_params=neighbor_params)
        dmap = mydmap.fit_transform(data1D)
        dill.dump(mydmap,open(dm_name,'wb'))
    else:
        mydmap = dill.load(open(dm_name, 'rb'))
        dmap = mydmap.transform(data1D)
    return Reps_mat, dmap

def read_files():
    '''
    A generator loads each permuted file.
    :return: a 3D array consisting of patches
    '''
    for filename in glob(patch_dir+'/permuted-*.bin'):
        D=np.fromfile(filename,dtype=np.float16)
        pics=D.reshape([-1,size,size])
        print('in read_files filename=%s, shape='%filename,pics.shape)
        #!rm $data_dir/$filename
        yield pics

def data_stream():
    '''
    A generator takes the 3D array from read-files() into patches.
    :return: one cell patch per time
    '''
    for pics in read_files():
        for i in range(pics.shape[0]):
            yield pics[i,:,:]


def transformation(A,B,parameter=False):
    coef1 = np.dot(A.T, A)/len(A)-np.dot(A.mean(axis=0).reshape(-1,1),A.mean(axis=0).reshape(1,-1))
    coef2 = np.dot(A.T, B)/len(A)-np.dot(A.mean(axis=0).reshape(-1,1),B.mean(axis=0).reshape(1,-1))
    M = np.dot(np.linalg.inv(coef1),coef2).T
    miu = B.mean(axis=0).reshape(1,-1)-np.dot(A.mean(axis=0).reshape(1,-1),M.T)
    A_transform = np.dot(A,M.T)+miu
    if parameter:
        return A_transform,M,miu
    else:
        return A_transform


if __name__=='__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--src_root", type=str, default=os.path.join(os.environ['ROOT_DIR'], 'vq/'),
                        help="Path to directory containing permuted cell files")
    parser.add_argument("--save_dir", type=str, default=os.path.join(os.environ['ROOT_DIR'], 'diffusionmap/'),
                        help="Path to directory saving VQ files")
    parser.add_argument("stack", type=str, help="The name of the brain")

    args = parser.parse_args()
    data_root = args.src_root
    dm_dir = args.save_dir
    stack = args.stack

    if not os.path.exists(dm_dir + stack):
        os.makedirs(dm_dir + stack)

    epsilon = {15:300,51:2500,201:5000}
    transform = {}
    num = 10
    for size in [15,51,201]:
        t0 = time()
        vq_name = data_root + '%s/VQ%d.pkl' % (stack, size)
        dm_name = dm_dir + '%s/diffusionMap-%d.pkl' % (stack, size)
        Reps_mat1, dmap = sample_and_mapping(vq_name, dm_name, size, eps=epsilon[size], train=True)

        dir_name = "permuted-%d" % size
        patch_dir = os.path.join(os.environ['ROOT_DIR'], 'permute/DK39/', dir_name)
        pics_list = []
        i = 0
        for pic in data_stream():
            pics_list.append(np.array(pic, dtype=np.float32))
            if size==201:
                i += 1
                if i >= 50000:
                    break
        pics = pack_pics(pics_list)
        data1D = np.concatenate([x.reshape([1, size * size]) for x in pics])
        mydmap = dill.load(open(dm_name, 'rb'))
        dmap2_full = mydmap.transform(data1D)

        dm_name = dm_dir+'%s/diffusionMap-%d.pkl'%('DK39', size)
        mydmap = dill.load(open(dm_name, 'rb'))
        dmap1_full = mydmap.transform(data1D)

        dmap2_T, M, miu = transformation(dmap2_full[:, :num], dmap1_full[:, :num], parameter=True)
        transform[size] = {}
        transform[size]['M'] = M.T
        transform[size]['miu'] = miu

        print(size, 'Finished in', time() - t0, 'seconds')

    fn = os.path.join(dm_dir + stack + '/transform.pkl')
    pickle.dump(transform, open(fn, 'wb'))