import cv2
from cv2 import moments,HuMoments
import pickle
import numpy as np

from astropy.convolution import MexicanHat2DKernel,convolve
from photutils.detection import find_peaks

from label_patch import diffusionMap
from patch_normalizer import normalizer
from lib.shape_utils import find_threshold

mexicanhat_2D_kernel = 10000*MexicanHat2DKernel(10)


class patch_extractor:
    def __init__(self,min_std=10,kernel=mexicanhat_2D_kernel,percentile=0.9):
        """Initialize a patch extractor. The extractor works by first checking if the grey value std is too small, in which case it aborts.
        otherwise it uses a mexican-hat kernel followed by photutils.detection.find_peaks to define the center of cells.

        :param min_std: defines what is considered an image that is "too flat" or "too empty"
        :param kernel: the kernel used for preprocessing
        :param percentile: the percentile of the values after convolution that are to be considered for find_peaks.
        :returns: 
        :rtype: 

        """
        self.min_std=min_std
        self.kernel=kernel
        self.percentile=percentile
        self.threshold=0.6
        self.Norm=normalizer(cell_size=41)
        #self.DM = diffusionMap('../notebooks/diffusionMap.pkl')
        self.cell_size=self.Norm.cell_size
        self.center=self.Norm.center


    def check_and_normalize(self,window,min_std=10):
        """ Compute the mean and td of the window, if std<min_std, return None
        otherwise, return a normalized window with mean 0 and std 1

        :param window: 2d np.array with the grey-val image of a tile
        :param min_std: thereshold for rejecting tile
        :returns: None or normalized 2D image
        :rtype: 2d np.array

        """
        # find whether window mosly blank and should be ignored.
        F=window.flatten()
        _mean=np.mean(F)
        _std=np.std(F)
        if _std < self.min_std:
            return None # window too uniform - background
        n_window=(window-_mean)/_std
        return n_window

    def preprocess(self,window):
        P=convolve(window,mexicanhat_2D_kernel)
        thr=find_threshold(P,self.percentile)
        Peaks=find_peaks(P,thr,footprint=self.Norm.footprint)
        return window,P,Peaks

    def extract_patches(self,_mean,Peaks,P):
        markers=_mean*np.float32(0)

        X=list(Peaks["x_peak"])
        Y=list(Peaks["y_peak"])

        extracted=[]
        for i in range(len(X)):
            corner_x=np.uint16(X[i]-self.center)
            corner_y=np.uint16(Y[i]-self.center)

            # ignore patches that extend outside of window
            if(corner_x<0 or corner_y<0 or \
               corner_x+self.cell_size>markers.shape[1] or corner_y+self.cell_size>markers.shape[0]):
                continue

            # extract patch
            ex=np.array(_mean[corner_y:corner_y+self.cell_size,corner_x:corner_x+self.cell_size])
            normalized_patch,rotation,confidence=self.Norm.normalize_patch(ex)
            #description=self.DM.label_patch([normalized_patch])
            extracted.append({'i':i,
                              'X':X[i],
                              'Y':Y[i],
                              'original_patch':ex,
                              'normalized_patch':normalized_patch,
                              'rotation':rotation,
                              'confidence':confidence,
                              #'description': description
            })

            # compute 
            # mark location of extracted patches
            #markers[corner_y:corner_y+cell_size,corner_x:corner_x+cell_size]=stamp

        return extracted
    

if __name__=="__main__":

    import argparse
    from time import time
    parser = argparse.ArgumentParser()
    parser.add_argument("filestem", type=str,
                    help="Process <filestem>.tif into <filestem>_extracted.pkl")

    parser.add_argument("-n", "--noinvert", action="store_true",
                        help="choose not to invert polarity of image, default is to invert (cells are dark)")
    args = parser.parse_args()
    infile = args.filestem+'.tif'
    outfile= args.filestem+'_extracted.pkl'
    polarity=-1
    if args.noinvert:
        polarity=1

    extractor=patch_extractor()
    
    window=cv2.imread(infile,cv2.IMREAD_GRAYSCALE | cv2.IMREAD_ANYDEPTH)

    window=polarity*window  # cells should be bright, background dark

    n_window=extractor.check_and_normalize(window)

    if n_window is None:
        print('image',infile,'too blank, skipping')
    else:
        t0=time()
        print('processing',infile,'into',outfile)
        _mean,P,Peaks=extractor.preprocess(n_window)    
        print('found',len(Peaks),'patches')
        extracted=extractor.extract_patches(_mean,Peaks,P)

        pickle.dump(extracted,open(outfile,'wb'))
        print('finished in %5.1f seconds'%(time()-t0))
