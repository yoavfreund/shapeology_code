import numpy
import matplotlib
from matplotlib import pylab, mlab, pyplot
np = numpy
plt = pyplot

from IPython.display import display
from IPython.core.pylabtools import figsize, getfigs

from pylab import *
from numpy import *

import pickle as pk
from glob import glob
import os
import sys
from time import time,asctime
sys.path.append('../scripts/')
from lib.shape_utils import *

time_log=[]
def clock(message):
    print('%s \t%8.1f'%(message, time()))
    time_log.append((message, time()))

def read_files():
    for filename in glob(patch_dir+'/permuted-*.bin'):
        D=np.fromfile(filename,dtype=np.float16)
        pics=D.reshape([-1,size,size])
        print('in read_files filename=%s, shape='%filename,pics.shape)
        #!rm $data_dir/$filename
        yield pics

def data_stream():
    for pics in read_files():
        for i in range(pics.shape[0]):
            yield pics[i,:,:]

def dist2(a,b):
    diff=(a-b)**2
    return sum(diff.flatten())

def refineKmeans(data,Reps):
    new_Reps=[np.zeros(Reps[0].shape) for r in Reps]
    Reps_count=[0.0 for r in Reps]
    error=0
    for i in range(data.shape[0]):
        patch=data[i,:,:]
        dists=[dist2(patch,r) for r in Reps]
        _argmin=np.argmin(dists)
        _min=min(dists)
        new_Reps[_argmin]+=patch
        Reps_count[_argmin]+=1
        error+=_min
    error /= data.shape[0]
    final_Reps=[]
    final_counts=[]
    for i in range(len(new_Reps)):
        if Reps_count[i]>5:
            final_Reps.append(new_Reps[i]/Reps_count[i])
            final_counts.append(Reps_count[i])
    return final_Reps,final_counts,error

def Kmeanspp(data,n=100,scale=550):
    Reps=[data[0,:,:]]

    Statistics=[]
    j=1
    mean_prob=1
    alpha=0.01
    for i in range(1,data.shape[0]):
        _min=100000
        patch=data[i,:,:]
        for r in Reps:
            _min=min(_min,dist2(patch,r))
        Prob=_min/scale
        print('\r','i=%10d,  #reps=%10d  Prob=%8.6f,scale=%8.4f'%(i,len(Reps),Prob,scale),end='')
        Statistics.append((i,len(Reps),_min))
        if np.random.rand()<Prob:
            Reps.append(patch)
            j+=1
        if j>=n:
            break
        if Prob>0.5:
            scale=scale*2
        else:
            scale=scale-1
    return Reps,Statistics

def Kmeans(data,n=100,scale=550):
    t0 = time()
    Reps, Statistics = Kmeanspp(data, n, scale)
    number = int(n*25)
    clock('Kmeans++ finished')
    print('Kmeans++ finished in', time() - t0, 'seconds')
    for i in range(5):
        Reps, final_counts, error = refineKmeans(data[(i + 1) * number:(i + 2) * number, :, :], Reps)
        clock('Kmeans refine iteration '+ str(i))
        print('refine iteration %2d, error=%7.3f, n_Reps=%5d' % (i, error, len(Reps)))
    return Reps, final_counts


if __name__=='__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--samples_size", type=int, default=500000,
                        help="Number of samples")
    parser.add_argument("--src_root", type=str, default=os.path.join(os.environ['ROOT_DIR'], 'permute/'),
                        help="Path to directory containing permuted cell files")
    parser.add_argument("--save_dir", type=str, default='vq/',
                        help="Path to directory containing cell files")
    parser.add_argument("padded_size", type=int,
                        help="One of the three padded size")


    # Add parameters for the size of patches and the path to the credential yaml file
    # Define file name based on size.

    args = parser.parse_args()
    size = args.padded_size
    N = args.samples_size
    # config = configuration(args.yaml)
    # params = config.getParams()
    root_dir = os.path.join(os.environ['ROOT_DIR'], args.save_dir)
    if not os.path.exists((root_dir)):
        os.makedirs(root_dir)
    data_dir = args.src_root
    dir_name = "permuted-%d" % size
    patch_dir = os.path.join(data_dir, dir_name)

    clock('Process Begin')
    pics_list = []
    i = 0
    for pic in data_stream():
        pics_list.append(np.array(pic, dtype=np.float32))
        i += 1
        if i >= N:
            break
    pics = pack_pics(pics_list)
    clock('Load samples')

    t0 = time()
    VQ = {}
    Reps, final_count = Kmeans(pics, n=int(N/250), scale=3000)
    VQ[size] = (Reps, final_count)
    pk.dump(VQ, open(root_dir + 'VQ' + str(size) + '.pkl', 'wb'))
    print('Finished in', time() - t0, 'seconds')
    clock('Process finished')

    log_fp = os.path.join(os.environ['ROOT_DIR'], 'TimeLog/')
    if not os.path.exists(log_fp):
        os.mkdir(log_fp)
    pk.dump(time_log, open(log_fp + 'Time_log_kmeans_'+ str(size)+'_'+asctime()+'.pkl', 'wb'))




