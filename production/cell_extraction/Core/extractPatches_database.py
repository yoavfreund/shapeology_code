import cv2
import pickle
import numpy as np
import random
import os
import sys
from skimage import io
import pandas as pd
import zlib
from time import time

from label_patch import diffusionMap
from patch_normalizer_v2 import normalizer
from lib.utils import mark_contours, configuration
from lib.shape_utils import *
sys.path.append('/home/k1qian/data/Github/pipeline/in_development/Will/showcase/cell_extraction/add_cell_data/')
from controller import FeaturesController
from cell_model import Cell

class patch_extractor:
    def __init__(self,params,dm=True,stem='diffusionMap'):
        """Initialize a patch extractor. 
        The extractor works by first checking if the gray 
        value std is too small, in which case it aborts.

        :param infile: filename of tile 
        :param params: parameters
        :returns: 
        :rtype: 

        """
        self.params=params
        self.min_area=params['preprocessing']['min_area']
        self.Norm=normalizer(params)
        self.preprocess_kernel=self.Norm.circle_patch(radius=1)
        self.dm_dir=stem
        self.train=dm
        #self.tile_stats={'tile name':infile}

        self.size_thresholds = params['normalization']['size_thresholds']
        if dm:
            self.DM = {size: diffusionMap(self.dm_dir + '-%d.pkl'%size) for size in self.size_thresholds}


        self.V={size:[] for size in self.size_thresholds} # storage for normalized patches


    def segment_cells(self,gray):
        """
        Segment cells from a given gray image
        :param gray: a gray-scale image
        :return:
        """
        offset = self.params['preprocessing']['offset']

        thresh = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
                                       cv2.THRESH_BINARY,101,offset)

        # erosion to seperate weakly linked blobs
        clean=cv2.erode(thresh,self.preprocess_kernel,iterations = 8)
        Stats=cv2.connectedComponentsWithStats(thresh)
        return Stats

    def extract_blobs(self,Stats,gray):
        """given a set of connected components extract convexified components from gray image and annotate on color image(tile)

        :param Stats: Output from cv2.connectedComponentsWithStats
        :param tile: original image
        :param gray: tile transfrmed to gray-scale
        :returns: 
        :rtype: 

        """
         # parse Stats
        no_blobs,seg,props,location = Stats

        left= props[:,0]
        top = props[:,1]
        width = props[:,2]
        height = props[:,3]
        area = props[:,4]

        # marked_tile=np.copy(tile)
        size_step=20
        extracted=[]
        H,W=seg.shape
        for i in range(1,no_blobs):
            if area[i]<self.min_area:
                continue
            #extract patch
            t,b,l,r = top[i],top[i]+height[i],left[i],left[i]+width[i]
            if t==0 or b==H or l==0 or r==W: #ignore patches that touch the boundary (likely to be partial)
                continue

            # Extract connected component
            sub_mask = np.array((seg[t:b,l:r]==i)*1,dtype=np.uint8)
            # recover threshold that was used
            masked_image=np.copy(gray[t:b,l:r])
            masked_image[sub_mask==0]=255
            _thr=np.min(masked_image.flatten())

            # compute convex hull of sub_mask
            contours, hierarchy = cv2.findContours(sub_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            convex_contour=cv2.convexHull(contours[0][:,0,:],returnPoints=True)
            slate=np.zeros([b-t,r-l,3],dtype=np.uint8)
            convex_mask = cv2.drawContours(slate, [convex_contour],0,(0,255,0),-1)[:,:,1]
            #compute Threshold used 
            masked_image=np.array(gray[t:b,l:r],dtype=np.int16)-_thr
            masked_image[convex_mask==0]=0
            masked_image[masked_image<0]=0

            properties={'left':left[i],
                        'top':top[i],
                        'width':width[i],
                        'height':height[i],
                        'area':area[i]}
            try:
                more_properties = self.Norm.normalize_patch(masked_image, properties)
            # except Exception as e:
            #     print(f'Error {e}')
            except:
                continue
            
            properties.update(more_properties)
            extracted.append(properties)

            # padded_patch=properties['padded_patch']
            # padded_size=properties['padded_size']
            #
            # if not padded_patch is None:
            #     self.V[padded_size].append(padded_patch)

            #print(properties.keys())
            #break
            # cv2.drawContours(marked_tile[t:b,l:r], [convex_contour],0,(0,255,0),1)

        ## compute diffusion vectors
        # self.timestamps = []
        # self.timestamps.append(('before DM',time()))
        if self.train:
            self.computeDMs(extracted)
        # self.timestamps.append(('after DM',time()))
            
        return extracted #,marked_tile

    def computeDMs(self,extracted):
        #self.timestamps.append(('start compute DM', time()))
        patchesBySize={size:[] for size in self.size_thresholds} # storage for normalized patches
        patchIndex={size:[] for size in self.size_thresholds}
      
        #collect patches by size
        for i in range(len(extracted)):
            properties=extracted[i]
            padded_size=properties['padded_size']
            patch = properties['padded_patch']
            if patch is None:
                continue
            patchesBySize[padded_size].append(patch.flatten())
            patchIndex[padded_size].append(i)

        # compute DM for each size
        for size in self.size_thresholds:
            _size=size    #temporary: until we have maps for all sizes

            asList=patchesBySize[_size]
            indexList=patchIndex[_size]
            _len=len(asList)
            if _len:
                asMat=np.zeros([_len,_size*_size])
                for i in range(_len):
                    asMat[i,:]=asList[i]
                #print('size os asMat:',asMat.shape)
                # self.timestamps.append(('befor transform DM', time()))

                DMMat=self.DM[_size].transform(asMat)
                # self.timestamps.append(('after transform DM', time()))

                #print(asMat.shape,DMMat.shape)

                # insert DM vectors back into properties
                for i in range(len(asList)):
                    index=indexList[i]
                    extracted[index]['DMVec']=DMMat[i,:]

if __name__=="__main__":

    import argparse
    from time import time
    parser = argparse.ArgumentParser()
    parser.add_argument("stack", type=str, help="The name of the brain")
    parser.add_argument("file", type=str,
                        help="Process <filename>.tif into <filename>_cells")
    parser.add_argument("--yaml", type=str, default=os.environ['REPO_DIR'] + 'shape_params.yaml',
                        help="Path to Yaml file with parameters")
    parser.add_argument("--save_dir", type=str, default=os.path.join(os.environ['ROOT_DIR'], 'cells/'),
                        help="Path to directory saving images")
    # Add parameters for size of mexican hat and size of cell, threshold, percentile
    # Define file name based on size. Use this name for log file and for countours image.
    # save parameters in a log file ,
    
    args = parser.parse_args()
    config = configuration(args.yaml)
    params = config.getParams()
    stack = args.stack

    _dir=args.save_dir
    filename=args.file
    dot = filename.rfind('.')
    slash = filename.rfind('/')
    section = int(filename[slash+1:dot])
    out_dir = os.path.join(_dir, filename[slash+1:dot]+'_cells')
    if not os.path.exists(_dir):
        os.makedirs(_dir)

    extractor = patch_extractor(params, dm=False)
    min_std = params['preprocessing']['min_std']

    t0 = time()
    img = io.imread(filename)
    tile = 255 - img.copy()
    _std = np.std(tile.flatten())
    if _std < min_std:
        print('image', filename, 'std=', _std, 'too blank, skipping')
    else:
        Stats = extractor.segment_cells(tile)
        extracted = extractor.extract_blobs(Stats, tile)

        cells = pd.DataFrame(extracted)
        cells = cells[cells['Normalized_patch'].notnull()]
        cells['section'] = int(section)
        cells['x'] = cells['left'] + cells['width'] / 2
        cells['y'] = cells['top'] + cells['height'] / 2
        cells = cells.astype({'x': int, 'y': int})
        cells = cells.drop(['left', 'top'], 1)
        cells = cells.to_dict('records')

        controller = FeaturesController()
        patches = []

        for i in range(len(cells)):
            properties = cells[i]
            cell = Cell(id=None, prep_id=stack, section=properties['section'], x=properties['x'], y=properties['y'], \
                        cell_width=properties['width'], cell_height=properties['height'], cell_area=properties['area'], \
                        rotation=properties['rotation'], rotation_confidence=properties['rotation_confidence'], \
                        img_width=properties['Normalized_patch'].shape[0], \
                        cell_images=zlib.compress(properties['Normalized_patch'].copy(order='C')))
            controller.add_row(cell)
            patches.append(properties['Normalized_patch'])


        fn = out_dir + '.pkl'
        pickle.dump(patches,open(fn,'wb'))
        print(os.path.getsize(fn))

        del patches
        print(filename, 'finished in', time() - t0, 'seconds')