"""
Utilities for plotting patches
Intended for use in notebooks
"""

import psutil
from os import system
from subprocess import Popen,PIPE
from time import time

import numpy as np
from numpy import sqrt, mean
#utils that do not require plotting
from subprocess import Popen,PIPE
from os import system
from os.path import isfile,getmtime

import cv2

#import matplotlib.pyplot as plt
#from astropy.convolution import Gaussian2DKernel,convolve


def plot_patches(fig,data,h=15,w=15,_titles=[]):
    for i in range(h*w):
        if i>=data.shape[0]:
            break
        ax=fig.add_subplot(h,w,i+1);
        pic=np.array(data[i,:,:],dtype=np.float32)

        subfig=ax.imshow(pic,cmap='gray')
        if(len(_titles)>i):
            plt.title(_titles[i])
        subfig.axes.get_xaxis().set_visible(False)
        subfig.axes.get_yaxis().set_visible(False)

def pack_pics(Reps):
    size=Reps[0].shape[0]
    _len=len(Reps)
    Reps_mat=np.zeros([_len,size,size], np.float16)
    for i in range(_len):
        Reps_mat[i,:,:]=Reps[i]
    return Reps_mat

