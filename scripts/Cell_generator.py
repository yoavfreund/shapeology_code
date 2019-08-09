import argparse

parser = argparse.ArgumentParser()
parser.add_argument("stack", type=str, help="The name of the stack")
parser.add_argument("structure", type=str, help="The nth group of structures")
parser.add_argument("state", type=str, help="Positive or negative samples")
parser.add_argument("yaml", type=str, help="Path to Yaml file with parameters")
parser.add_argument("filename", type=str, help="Path to patch file")
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

def CDF(x):
    x=np.sort(x)
    size=x.shape[0]
    y=np.arange(0,size)/size
    return x,y

def setup_download_from_s3(rel_fp, recursive=True):
    s3_fp = 's3://mousebrainatlas-data/' + rel_fp
    local_fp = os.environ['ROOT_DIR'] + rel_fp

    if os.path.exists(local_fp):
        print('ALREADY DOWNLOADED FILE')
        return

    if recursive:
        run('aws s3 cp --recursive {0} {1}'.format(s3_fp, local_fp))
    else:
        run('aws s3 cp {0} {1}'.format(s3_fp, local_fp))

def setup_upload_from_s3(rel_fp, recursive=True):
    s3_fp = 's3://mousebrainatlas-data/' + rel_fp
    local_fp = os.environ['ROOT_DIR'] + rel_fp

    if recursive:
        run('aws s3 cp --recursive {0} {1}'.format(local_fp, s3_fp))
    else:
        run('aws s3 cp {0} {1}'.format(local_fp, s3_fp))

@ray.remote
def generator(structure, state, threshold, cell_dir, patch_dir, stack, params):
    for state in [state]:
        t1 = time()
        extractor = patch_extractor(params)
        #savepath = cell_dir + 'Properties/' + structure + '/'
        #img_path = cell_dir + 'Images/' + structure + '/'
        savepath = cell_dir + structure + '/'
        pkl_out_file = savepath+stack+'_'+structure+'_'+state+'.pkl'
        #img_out_file = img_path+stack+'_'+structure+'_'+state+'_images.pkl'
        if os.path.exists(os.environ['ROOT_DIR']+pkl_out_file):
            print(structure +'_'+state+ ' ALREADY EXIST')
            continue
        else:
            if not os.path.exists(os.environ['ROOT_DIR']+savepath):
                os.mkdir(os.environ['ROOT_DIR']+savepath)
                #os.mkdir(img_path)

        if structure == '7nn':
            structure = '7n'

        if state=='positive':
            setup_download_from_s3(patch_dir+structure)
            patches = [dir for dir in glob(os.environ['ROOT_DIR']+patch_dir+structure+'/*')]
        else:
            setup_download_from_s3(patch_dir+structure+'_surround_500um_noclass')
            patches = [dir for dir in glob(os.environ['ROOT_DIR']+patch_dir+structure+'_surround_500um_noclass/*')]

        features=[]

        n_choose = min(len(patches), 1000)
        indices_choose = np.random.choice(range(len(patches)), n_choose, replace=False)
        patches = np.array(patches)
        patches = patches[indices_choose]

        for i in range(len(patches)):
            tile=cv2.imread(patches[i],0)
            # contours, _ = cv2.findContours(tile.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            # if state=='positive':
            #     if len(contours)==1:
            #         object_area = cv2.contourArea(contours[0])
            #     else:
            #         areas=[]
            #         for j in range(len(contours)):
            #             areas.extend([cv2.contourArea(contours[j])])
            #         object_area = max(areas)
            # else:
            #     if len(contours)==2:
            #         object_area = cv2.contourArea(contours[0])-cv2.contourArea(contours[1])
            #     else:
            #         areas=[]
            #         for j in range(len(contours)):
            #             areas.extend([cv2.contourArea(contours[j])])
            #         areas=np.sort(areas)
            #         object_area = areas[-1]-areas[-2]
            
            if params['preprocessing']['polarity']==-1:
                tile = 255-tile
            min_std=params['preprocessing']['min_std']
            _std = np.std(tile.flatten())
            
            extracted = []
            if _std < min_std:
                print('image',patches[i],'std=',_std, 'too blank')
                # features.append([0] * 201)
                features.append([0] * 1981)
                # features.append([0] * 1581)
            else:
                try:
                    Stats = extractor.segment_cells(tile)
                    cells = extractor.extract_blobs(Stats,tile)
                    cells = pd.DataFrame(cells)
                    cells = cells[cells['padded_patch'].notnull()]
                    cells = cells.drop(['padded_patch','left','top'],1)
                    cells = np.asarray(cells)
                    for k in range(len(cells)):
                        cells[k][0] = cells[k][0][:10]
                    origin = np.concatenate((np.array(list(cells[:,0])),cells[:,1:]),axis=1)
                    for k in range(origin.shape[1]):
                        x, y = CDF(origin[:,k])
                        # ten = [x[np.argmin(np.absolute(y - 0.1*(1+j)))] for j in range(10)]
                        ten = [y[np.argmin(np.absolute(x-threshold[k][j]))] for j in range(99)]
                        extracted.extend(ten)
                    extracted.extend([cells.shape[0]])
                    features.append(extracted)
                except:
                    continue
            if i%10==0:
                count = len(features)
                print(structure + '_' + state, count, i, '/', len(patches))
                    
#                 Stats=extractor.segment_cells(tile)
#                 extracted= extractor.extract_blobs(Stats,tile)
#                 cells.extend(extracted)
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
                

#         cells = pd.DataFrame(cells)
#         cells = cells[cells['padded_patch'].notnull()]
#         images = cells[['padded_size','padded_patch']]
#         cells = cells.drop('padded_patch',1)
        count = len(features)
        print(structure + '_' + state, count)
#         cells.to_pickle(pkl_out_file)
#         images.to_pickle(img_out_file)
        pickle.dump(features, open(os.environ['ROOT_DIR']+pkl_out_file, 'wb'))
        setup_upload_from_s3(pkl_out_file, recursive=False)
        #s3_directory = 's3://mousebrainatlas-data/CSHL_cells_dm/'+stack+'/'+structure+'/'
        #run('aws s3 cp {0} {1}'.format(pkl_out_file,s3_directory))
        print(structure + '_'+state+ ' finished in %5.1f seconds' % (time() - t1))

yamlfile=os.environ['REPO_DIR']+args.yaml
params=configuration(yamlfile).getParams()

fn = 'CSHL_data_processed/MD589/ThresholdsV2.pkl'
setup_download_from_s3(fn, recursive=False)
thresholds = pickle.load(open(os.environ['ROOT_DIR']+fn,'rb'))
# threshold = thresholds[struc]

patch_dir = args.filename+'/'+stack+'/'
cell_dir = os.environ['ROOT_DIR']+args.filename+'_features_V2/'
if not os.path.exists(cell_dir):
    os.mkdir(cell_dir)
cell_dir = cell_dir+stack+'/'
if not os.path.exists(cell_dir):
    os.mkdir(cell_dir)
    #os.mkdir(cell_dir+'Images/')
    #os.mkdir(cell_dir+'Properties/')

cell_dir = args.filename+'_features_V2/'+stack+'/'
#t0=time()

#assert structure

ray.get(generator.remote(struc, state, thresholds, cell_dir, patch_dir, stack, params))


#print('Finished in %5.1f seconds'%(time()-t0))

