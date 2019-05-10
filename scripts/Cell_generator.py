import argparse

parser = argparse.ArgumentParser()
parser.add_argument("stack", type=str, help="The name of the stack")
parser.add_argument("yaml", type=str, help="Path to Yaml file with parameters")
args = parser.parse_args()
stack = args.stack

import cv2
#from cv2 import moments,HuMoments
import pickle
import numpy as np


import os
import sys
from time import time
from glob import glob
from extractPatches import patch_extractor
#from label_patch import diffusionMap
#from patch_normalizer import normalizer
from lib.utils import mark_contours, configuration

yamlfile=os.environ['REPO_DIR']+args.yaml
params=configuration(yamlfile).getParams()

paired_structures = ['5N', '6N', '7N', '7n', 'Amb', 'LC', 'LRt', 'Pn', 'Tz', 'VLL', 'RMC', \
                     'SNC', 'SNR', '3N', '4N', 'Sp5I', 'Sp5O', 'Sp5C', 'PBG', '10N', 'VCA', 'VCP', 'DC']
singular_structures = ['AP', '12N', 'RtTg', 'SC', 'IC']

all_structures = paired_structures + singular_structures
patch_dir = os.environ['ROOT_DIR']+'CSHL_patches/'+stack+'/'
if not os.path.exists(os.environ['ROOT_DIR']+'CSHL_cells_dm/'):
    os.mkdir(os.environ['ROOT_DIR']+'CSHL_cells_dm/')
cell_dir = os.environ['ROOT_DIR']+'CSHL_cells_dm/'+stack+'/'
print(cell_dir)
if not os.path.exists(cell_dir):
    os.mkdir(cell_dir)

t0=time()

for structure in all_structures:
    t1=time()
    for state in ['positive','negative']:

        savepath = cell_dir + structure + '/'
        if os.path.exists(savepath):
            print(structure + ' ALREADY EXIST')
            continue
        else:
            os.mkdir(savepath)

        if state=='positive':
            patches = [dir for dir in glob(patch_dir+structure+'/*')]
        else:
            patches = [dir for dir in glob(patch_dir+structure+'_surround_200um_noclass/*')]

        #count=0
        cells=[]
        pkl_out_file = savepath+stack+'_'+state+'.pkl'
        for i in range(len(patches)):
            extractor=patch_extractor(patches[i],params)
            tile=cv2.imread(patches[i],0)
            if params['preprocessing']['polarity']==-1:
                tile = 255-tile
            min_std=params['preprocessing']['min_std']
            _std = np.std(tile.flatten())

            if _std < min_std:
                print('image',patches[i],'std=',_std, 'too blank, skipping')
            else:
                Stats=extractor.segment_cells(tile)
                extracted= extractor.extract_blobs(Stats,tile)
            cells.extend(extracted)
                # for j in range(len(extracted)):
                #     try:
                #         filename=savepath+str(extracted[j]['padded_size'])+'/'+str(count)+'.tif'
                #         count+=1
                #         img=extracted[j]['padded_patch']
                #         img=img/img.max()*255
                #         img=img.astype(np.uint8)
                #         cv2.imwrite(filename, img)
                #     except:
                #         continue
            #if count>100000:
                #break
        pickle.dump(cells, open(pkl_out_file, 'wb'))
    print(structure+'finished in %5.1f seconds'%(time()-t1))
print('Finished in %5.1f seconds'%(time()-t0))

