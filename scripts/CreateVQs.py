import pickle as pk
import numpy as np
#from numpy import *
from glob import glob
import sys
from lib.shape_utils import *
from lib.utils import configuration

from os import system
from os.path import isfile
import matplotlib.pyplot as plt

def dist2(a,b):
    diff=(a-b)**2
    return sum(diff.flatten())

class VQ_creator:
    """ a class for creating a VQ sequence """
    def __init__(self, params, _size):
        """
        Initialize a VQ sequence creator.
        :param params: parameters loaded from a yaml file
        :param _size: one of the three padded sizes
        """
        self.params = params
        self.size_thresholds = params['normalization']['size_thresholds']
        self.patch_dir = params['paths']['patches'] + '/' + str(_size)
        self.size = _size
        self.scale = 100

    def read_files(self):
        for filename in glob(self.patch_dir+'/*.bin'):
            D=np.fromfile(filename,dtype=np.float16)
            print('in read_files filename=%s, shape='%filename,D.shape)
            pics=D.reshape([-1,self.size,self.size])
            yield pics

    def data_stream(self):
        for pics in self.read_files():
            for i in range(pics.shape[0]):
                yield pics[i, :, :]

    def Kmeanspp(self, Reps=[], n=100):
        if len(Reps) == 0:
            Reps = [next(self.data_stream())]

        Statistics = []
        j = len(Reps)
        i = 0
        for patch in self.data_stream():
            _min = 100000
            for r in Reps:
                _min = min(_min, dist2(patch, r))

            if _min > self.scale:
                self.scale *= 1.5
                print('scale=', self.scale)

            Prob = _min / self.scale
            print('\r', 'i=%10d,  #reps=%10d  Prob=%8.6f' % (i, len(Reps), Prob), end='')
            i += 1
            Statistics.append((i, len(Reps), _min))
            if np.random.rand() < Prob:
                Reps.append(patch)
                j += 1
            if j >= n:
                break
        return Reps, Statistics

    def refineKmeans(self, Reps, per_rep_sample=100, refinement_iter=3):
        _shape = Reps[0].shape
        new_Reps = [np.zeros(_shape) for r in Reps]
        _area = _shape[0] * _shape[1]
        Reps_count = [0.0 for r in Reps]
        error = 0
        count = per_rep_sample * len(Reps)
        i = 0
        for patch in self.data_stream():
            dists = [dist2(patch, r) for r in Reps]
            _argmin = np.argmin(dists)
            _min = min(dists)
            new_Reps[_argmin] += patch
            Reps_count[_argmin] += 1
            error += _min
            i += 1
            if i >= count:
                break
        error /= (count * _area)
        final_Reps = []
        final_counts = []
        for i in range(len(new_Reps)):
            if Reps_count[i] > refinement_iter:
                final_Reps.append(new_Reps[i] / Reps_count[i])
                final_counts.append(Reps_count[i])
        return final_Reps, final_counts, error


    def Kmeans(self,Reps=[],n=100):
        Reps,Statistics = self.Kmeanspp(Reps,n)
        for i in range(5):
            Reps,final_counts,error = self.refineKmeans(Reps)
            print('refine iteration %2d, error=%7.3f, n_Reps=%5d'%(i,error,len(Reps)))
        return Reps,final_counts



if __name__=='__main__':

    import argparse
    from time import time
    parser = argparse.ArgumentParser()
    parser.add_argument("filestem", type=str,
                    help="Process <filestem>.tif into <filestem>_extracted.pkl")
    parser.add_argument("yaml", type=str,
                    help="Path to Yaml file with parameters")
    
    # Add parameters for size of mexican hat and size of cell, threshold, percentile
    # Define file name based on size. Use this name for log file and for countours image.
    # save parameters in a log file ,
    
    args = parser.parse_args()
    config = configuration(args.yaml)
    params=config.getParams()

    self.size_thresholds = params['normalization']['size_thresholds']
    s3dir=params['paths']['patches']
    gen=filtered_images(s3dir,smooth_threshold=0.35,reduce_res=True)
    VQ={}
    for n in self.size_thresholds:
        for c in range(10):
            print('========   n=%5d, c=%1d ==========='%(n,c))
            Reps,final_count=Kmeans(gen,n=n)
            VQ[(n,c)]=(Reps,final_count)

        #plot_patches(pack_pics(Reps),_titles=['n=%4d:#=%4d'%(n,x) for x in final_count])

    pk.dump(VQ,open('VQ5.pkl','wb'))
