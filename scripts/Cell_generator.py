import argparse

parser = argparse.ArgumentParser()
parser.add_argument("stack", type=str, help="The name of the stack")
parser.add_argument("structure", type=str, help="The nth group of structures")
parser.add_argument("state", type=str, help="Positive or negative samples")
parser.add_argument("yaml", type=str, help="Path to Yaml file with parameters")
args = parser.parse_args()
stack = args.stack
struc = args.structure
state = args.state
import cv2
#from cv2 import moments,HuMoments
import pickle
import numpy as np
import pandas as pd

import os
import sys
from time import time
from glob import glob
from extractPatches import patch_extractor
#from label_patch import diffusionMap
#from patch_normalizer import normalizer
from lib.utils import mark_contours, configuration, run

import ray
ray.init()
#ray.init(object_store_memory=70000000000,redis_max_memory=30000000000)

@ray.remote
def generator(structure, state, cell_dir, patch_dir, stack, params):
    for state in [state]:
        t1 = time()
        savepath = cell_dir + 'Properties/' + structure + '/'
        img_path = cell_dir + 'Images/' + structure + '/'
        pkl_out_file = savepath+stack+'_'+structure+'_'+state+'.pkl'
        img_out_file = img_path+stack+'_'+structure+'_'+state+'_images.pkl'
        if os.path.exists(pkl_out_file):
            print(structure +'_'+state+ ' ALREADY EXIST')
            continue
        else:
            if not os.path.exists(savepath):
                os.mkdir(savepath)
                os.mkdir(img_path)

        if state=='positive':
            patches = [dir for dir in glob(patch_dir+structure+'/*')]
        else:
            patches = [dir for dir in glob(patch_dir+structure+'_surround_200um_noclass/*')]

        cells=[]

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
                if i%10==0:
                    count = len(cells)
                    print(structure + '_' + state, count, i, '/', len(patches))

        cells = pd.DataFrame(cells)
        cells = cells[cells['padded_patch'].notnull()]
        images = cells[['padded_size','padded_patch']]
        cells = cells.drop('padded_patch',1)
        count = len(cells)
        print(structure + '_' + state, count)
        cells.to_pickle(pkl_out_file)
        images.to_pickle(img_out_file)
        #pickle.dump(cells, open(pkl_out_file, 'wb'))
        #s3_directory = 's3://mousebrainatlas-data/CSHL_cells_dm/'+stack+'/'+structure+'/'
        #run('aws s3 cp "{0}" {1}/'.format(pkl_out_file,s3_directory))
        print(structure + '_'+state+ ' finished in %5.1f seconds' % (time() - t1))

yamlfile=os.environ['REPO_DIR']+args.yaml
params=configuration(yamlfile).getParams()

paired_structures = ['5N', '6N', '7N', '7n', 'Amb', 'LC', 'LRt', 'Pn', 'Tz', 'VLL', 'RMC', \
                     'SNC', 'SNR', '3N', '4N', 'Sp5I', 'Sp5O', 'Sp5C', 'PBG', '10N', 'VCA', 'VCP', 'DC']
singular_structures = ['AP', '12N', 'RtTg', 'SC', 'IC']

all_structures = paired_structures + singular_structures
patch_dir = os.environ['ROOT_DIR']+'CSHL_regions/'+stack+'/'
cell_dir = os.environ['ROOT_DIR']+'CSHL_cells_features/'
if not os.path.exists(cell_dir):
    os.mkdir(cell_dir)
cell_dir = cell_dir+stack+'/'
print(cell_dir)
if not os.path.exists(cell_dir):
    os.mkdir(cell_dir)
    os.mkdir(cell_dir+'Images/')
    os.mkdir(cell_dir+'Properties/')

#t0=time()

#assert structure

ray.get(generator.remote(struc, state, cell_dir, patch_dir, stack, params))


#print('Finished in %5.1f seconds'%(time()-t0))
