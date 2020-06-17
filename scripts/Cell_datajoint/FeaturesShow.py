import argparse

parser = argparse.ArgumentParser()
parser.add_argument("stack", type=str, help="The name of the stack")
parser.add_argument("structure", type=str, help="The name of the structure")
args = parser.parse_args()
stack = args.stack
struc = args.structure

import pickle as pk
import numpy as np
import pandas as pd
from glob import glob
from matplotlib import pyplot, pylab
import datajoint as dj
from minio import Minio
import json
import yaml
import sys, os

sys.path.append('./lib')
from utilities import *
sys.path.append('../lib')
from utils import run

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

def CDF(x):
    x=np.sort(x)
    size=x.shape[0]
    y=np.arange(0,size)/size
    pyplot.plot(x,y);



setup_download_from_s3('CSHL_cells_features/'+stack+'/Properties')
img_fp = 'Pictures/'+struc+'/'
if not os.path.exists('Pictures'):
    os.mkdir('Pictures')
if not os.path.exists(img_fp):
    os.mkdir(img_fp)

#credFiles= '/data/Github/VaultBrain/credFiles_aws.yaml'
credFiles= '/Users/kuiqian/Github/VaultBrain/credFiles.yaml'
dj.config['database.host'] = get_dj_creds(credFiles)['database.host']
dj.config['database.user'] = get_dj_creds(credFiles)['database.user']
dj.config['database.port'] = get_dj_creds(credFiles)['database.port']
dj.config['database.password'] = get_dj_creds(credFiles)['database.password']
dj.conn()
schema = dj.schema('kui_diffusionmap')
schema.spawn_missing_classes()

structureTable = Structure.fetch(as_dict=True)
strucDF = pd.DataFrame(structureTable)

posi_fp = os.environ['ROOT_DIR'] + strucDF[strucDF['structure']==struc]['positive_s3_fp'].to_list()[0]
nega_fp = os.environ['ROOT_DIR'] + strucDF[strucDF['structure']==struc]['negative_s3_fp'].to_list()[0]
posi = pd.read_pickle(posi_fp)
posi = posi.drop(['left','top'],1)
nega = pd.read_pickle(nega_fp)
nega = nega.drop(['left','top'],1)

pyplot.figure(figsize=[12,16])
for i in range(1,posi.shape[1]):
    pyplot.subplot(4,3,i)
    item = posi.columns[i]
    CDF(posi[item])
    CDF(nega[item])
    pyplot.title(struc+'_'+item)
pyplot.legend(['Positive','Negative'])
fn = img_fp+struc+'_features.jpg'
pyplot.savefig(fn)

pyplot.figure(figsize=[16,8])
size = 201
items = ['area','rotation','height','width','std','mean','horiz_std','vert_std']
for i in range(8):
    pyplot.subplot(2,4,i+1)
    item = items[i]
    CDF(posi.groupby('padded_size').get_group(size)[item])
    CDF(nega.groupby('padded_size').get_group(size)[item])
    pyplot.title(struc+'_'+item+'_'+str(size))
pyplot.legend(['Positive','Negative'])
fn = img_fp+struc+'_'+str(size)+'_features.jpg'
pyplot.savefig(fn)

pyplot.figure(figsize=[16,8])
size = 51
items = ['area','rotation','height','width','std','mean','horiz_std','vert_std']
for i in range(8):
    pyplot.subplot(2,4,i+1)
    item = items[i]
    CDF(posi.groupby('padded_size').get_group(size)[item])
    CDF(nega.groupby('padded_size').get_group(size)[item])
    pyplot.title(struc+'_'+item+'_'+str(size))
pyplot.legend(['Positive','Negative'])
fn = img_fp+struc+'_'+str(size)+'_features.jpg'
pyplot.savefig(fn)

pyplot.figure(figsize=[12,8])
items = ['area', 'rotation']
size = [15, 51, 201]
for i in range(2):
    item = items[i]
    for j in range(3):
        pyplot.subplot(2,3,i*3+j+1)
        CDF(posi.groupby('padded_size').get_group(size[j])[item])
        CDF(nega.groupby('padded_size').get_group(size[j])[item])
        pyplot.title(struc+'_'+item+'_'+str(size[j]))
pyplot.legend(['Positive','Negative'])
fn = img_fp+struc+'_features_by_size.jpg'
pyplot.savefig(fn)

size = [15, 51, 201]
posi_dm = [[],[],[]]
nega_dm = [[],[],[]]
for j in range(3):
    dm = [features['DMVec'] for id, features in posi.groupby('padded_size').get_group(size[j]).iterrows()]
    posi_dm[j] = np.array(dm)
    dm = [features['DMVec'] for id, features in nega.groupby('padded_size').get_group(size[j]).iterrows()]
    nega_dm[j] = np.array(dm)

pyplot.figure(figsize=[12,16])
dx = 0
for i in range(4):
    for j in range(3):
        pyplot.subplot(4,3,i*3+j+1)
        CDF(posi_dm[j][:,i+dx])
        CDF(nega_dm[j][:,i+dx])
        pyplot.title(struc+'_'+str(i+dx)+'_'+str(size[j]))
pyplot.legend(['Positive','Negative'])
fn = img_fp+struc+'_dm.jpg'
pyplot.savefig(fn)

pyplot.figure(figsize=[12,4])
for i in range(1,4):
    pyplot.subplot(1,3,i)
    pylab.scatter(posi_dm[2][:,dx],posi_dm[2][:,i+dx],s=1)
    pylab.scatter(nega_dm[2][:,dx],nega_dm[2][:,i+dx],s=1,alpha=0.3)
    pyplot.title(struc+'_'+str(i+dx)+'_'+str(size[j]))
pyplot.legend(['Positive','Negative'])
fn = img_fp+struc+'_dm2d.jpg'
pyplot.savefig(fn)

dm = np.asarray(posi)
for i in range(len(dm)):
        dm[i][0] = dm[i][0][:10]
posi_dm = np.array(list(dm[:,0]))
dm = np.asarray(nega)
for i in range(len(dm)):
        dm[i][0] = dm[i][0][:10]
nega_dm = np.array(list(dm[:,0]))
pyplot.figure(figsize=[12,16])
for i in range(10):
    pyplot.subplot(4,3,i+1)
    CDF(posi_dm[:,i])
    CDF(nega_dm[:,i])
    pyplot.title(struc+'_'+str(i))
pyplot.legend(['Positive','Negative'])
fn = img_fp+struc+'_10dm.jpg'
pyplot.savefig(fn)
