import cv2
from cv2 import moments,HuMoments
import pickle
import numpy as np

#from label_patch import diffusionMap
from patch_normalizer import normalizer
from lib.utils import mark_contours, configuration

class patch_extractor:
    def __init__(self,infile,params):
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
        
        self.tile_stats={'tile name':infile}
        #self.DM = diffusionMap('../notebooks/diffusionMap.pkl')

    def segment_cells(self,gray):
        offset = self.params['preprocessing']['offset']

        thresh = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
                                       cv2.THRESH_BINARY,101,offset)

        # erosion to seperate weakly linked blobs
        clean=cv2.erode(thresh,self.preprocess_kernel,iterations = 8)
        Stats=cv2.connectedComponentsWithStats(thresh)
        return Stats

    def extract_blobs(self,Stats,tile,gray):
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

        marked_tile=np.copy(tile)
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
            im2, contours, hierarchy = cv2.findContours(sub_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
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
            more_properties = self.Norm.normalize_patch(masked_image, properties)
            properties.update(more_properties)
            extracted.append(properties)
            #print(properties.keys())
            #break
            cv2.drawContours(marked_tile[t:b,l:r], [convex_contour],0,(0,255,0),1)
        return extracted,marked_tile
    
if __name__=="__main__":

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

    stem=args.filestem
    infile = stem+'.tif'
    out_stem= stem+'.'+params['name']
    outfile= out_stem+'_extracted.pkl'
    annotated_infile=out_stem+'_contours.jpg'

    extractor=patch_extractor(infile,params)

    tile=cv2.imread(infile)
    #print('tile is of type',type(tile[0,0,0]))
    gray = cv2.cvtColor(tile,cv2.COLOR_BGR2GRAY)
    print('gray is of type',type(gray[0,0]))

    if params['preprocessing']['polarity']==-1:
        gray = 255-gray

    #n_window=extractor.check_and_normalize(gray)

    min_std=params['preprocessing']['min_std']
    _std = np.std(gray.flatten())
    
    if _std < min_std:
        print('image',infile,'std=',_std, 'too blank, skipping')
    else:
        t0=time()
        print('processing',infile,'into',outfile)
        Stats=extractor.segment_cells(gray)
        extracted,marked_tile = extractor.extract_blobs(Stats,tile,gray)

        print('extracted',len(extracted),'patches')
        
        pickle.dump(extracted,open(outfile,'wb'))
        print('patches written to',outfile)

        cv2.imwrite(annotated_infile,marked_tile)
        print('annotated image written to',annotated_infile)
        print('finished in %5.1f seconds'%(time()-t0))
