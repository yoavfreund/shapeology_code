import pickle as pk
import numpy as np
#from numpy import *
from glob import glob
from subprocess import Popen,PIPE
import sys
import traceback
from os import system
from os.path import isfile
import matplotlib.pyplot as plt
from astropy.convolution import Gaussian2DKernel,convolve

def refineKmeans(data_stream,Reps,per_rep_sample=100,refinement_iter=3):
    _shape=Reps[0].shape
    new_Reps=[np.zeros(_shape) for r in Reps]
    _area=_shape[0]*_shape[1]
    Reps_count=[0.0 for r in Reps]
    error=0
    count=per_rep_sample*len(Reps)
    i=0
    for patch in data_stream: 
        dists=[dist2(patch,r) for r in Reps]
        _argmin=argmin(dists)
        _min=min(dists)
        new_Reps[_argmin]+=patch
        Reps_count[_argmin]+=1
        error+=_min
        i+=1
        if i >= count:
            break
    error /= (count*_area)
    final_Reps=[]
    final_counts=[]
    for i in range(len(new_Reps)):
        if Reps_count[i]>refinement_iter:
            final_Reps.append(new_Reps[i]/Reps_count[i])
            final_counts.append(Reps_count[i])
    return final_Reps,final_counts,error

def Kmeans(data_stream,Reps=[],n=100):
    Reps,Statistics = Kmeanspp(data_stream,Reps,n)
    for i in range(5):
        Reps,final_counts,error = refineKmeans(data_stream,Reps)
        print('refine iteration %2d, error=%7.3f, n_Reps=%5d'%(i,error,len(Reps)))
    return Reps,final_counts

global scale
scale=550
def Kmeanspp(data_stream,Reps=[],n=100):
    global scale
    if len(Reps)==0:
        Reps=[next(data_stream)]

    Statistics=[]
    j=len(Reps)
    i=0
    for patch in data_stream: 
        _min=100000
        for r in Reps:
            _min=min(_min,dist2(patch,r))

        if _min>scale:
            scale*=1.5
            print('scale=',scale)

        Prob=_min/scale
        print('\r','i=%10d,  #reps=%10d  Prob=%8.6f'%(i,len(Reps),Prob),end='')
        i+=1
        Statistics.append((i,len(Reps),_min))
        if np.random.rand()<Prob:
            Reps.append(patch)
            j+=1
        if j>=n:
            break
    return Reps,Statistics

def plot_statistics(Statistics,alpha=0.05,_start=10): 
    N=[x[1] for x in Statistics]
    d=[x[2] for x in Statistics]

    s=mean(d[:_start])
    smoothed=[s]*_start
    for x in d[_start:]:
        s=(1-alpha)*s + alpha*x
        smoothed.append(s)
    loglog(N[_start:],smoothed[_start:])
    xlabel('N')
    ylabel('smoothed distance')
    grid(which='both')

def filtered_images(s3_dir='s3://mousebraindata-open/MD657/permuted',reduce_res=True,smooth_threshold=0.4):
    for pic in data_stream(s3_dir):
        err,sub=calc_err(pic)
        if err>smooth_threshold:
            continue
        if reduce_res:
            yield sub
        else:
            yield pic

if __name__=='__main__':

    _size=41
    gen=filtered_images(smooth_threshold=0.35,reduce_res=True)
    VQ={}
    for n in [10,100,500]:
        for c in range(10):
            print('========   n=%5d, c=%1d ==========='%(n,c))
            Reps,final_count=Kmeans(gen,n=n)
            VQ[(n,c)]=(Reps,final_count)

        #plot_patches(pack_pics(Reps),_titles=['n=%4d:#=%4d'%(n,x) for x in final_count])

    pk.dump(VQ,open('VQ5.pkl','wb'))
