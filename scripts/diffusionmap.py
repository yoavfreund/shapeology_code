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
sys.path.append('../scripts/lib/')
from shape_utils import *

import dill
from PIL import Image

from pydiffmap import diffusion_map as dm
# initialize Diffusion map object.
neighbor_params = {'n_jobs': -1, 'algorithm': 'ball_tree'}


def scatter_images(pics,dmap,d1=0,d2=1,canvas_sz=1000):
    canvas_size=np.array([canvas_sz,canvas_sz])
    _minx=min(dmap[:,d1])
    _maxx=max(dmap[:,d1])
    _miny=min(dmap[:,d2])
    _maxy=max(dmap[:,d2])
    shift_x = -_minx
    scale_x = canvas_size[0]/(_maxx - _minx)
    shift_y = -_miny
    scale_y = canvas_size[1]/(_maxy - _miny)

    x=[int((_x+shift_x)*scale_x) for _x in dmap[:,d1]]
    y=[int((_y+shift_y)*scale_y) for _y in dmap[:,d2]]

    image_size=np.array(pics.shape[1:])
    canvas=Image.new('LA', tuple(canvas_size+image_size))
    for i in range(pics.shape[0]):
        gray = pics[i,:,:]
        gray = np.uint8(gray/gray.max()*255)
        img = Image.fromarray(gray)
        img.putalpha(255)
        canvas.paste(img, (x[i],y[i]))
    return canvas

def plot_dm(vq_name, dm_name, stack, size, knear=100, eps=3000, canvas_sz=1000, train=True):
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

    save_dir = os.path.join(os.environ['ROOT_DIR'], 'scatterplots/%s/scatterplots-%d/'%(stack, size))
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    for d1 in range(5):
        for d2 in range(d1+1,8):
            fn = save_dir + 'scatter%d%d.png'%(d1,d2)
            canvas=scatter_images(Reps_mat,dmap,d1=d1,d2=d2,canvas_sz=canvas_sz)
            canvas.save(fn)

if __name__=='__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--samples_size", type=int, default=500000,
                        help="Number of samples")
    parser.add_argument("--src_root", type=str, default=os.path.join(os.environ['ROOT_DIR'], 'permute/'),
                        help="Path to directory containing permuted cell files")
    parser.add_argument("--save_dir", type=str, default='vq/',
                        help="Path to directory saving VQ files")
    parser.add_argument("padded_size", type=int,
                        help="One of the three padded size")

