import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("stack", type=str, help="The name of the brain")
parser.add_argument("img_file", type=str, help="The path to the image file")
parser.add_argument("db_file", type=str, help="The path to the image file")
parser.add_argument("--yaml", type=str, default=os.path.join(os.environ['REPO_DIR'], 'shape_params.yaml'),
                    help="Path to Yaml file with parameters")
args = parser.parse_args()
stack = args.stack

from skimage import io, color
from skimage.transform import rescale
import numpy as np
import pandas as pd
import sys
import sqlite3
import cv2
import colorsys

sys.path.append(os.environ['REPO_DIR'])
from lib.utils import configuration, run

img_fn = args.img_file
dot = img_fn.rfind('.')
slash = img_fn.rfind('/')
section = img_fn[slash + 1:dot]

# conn = sqlite3.connect('/data/Shapeology_Files/BstemAtlasDataBackup/ucsd_brain/CSHL_databases/DK39/186.db')
# cur = conn.cursor()
# raws = cur.execute('SELECT * FROM features')
# info = np.array(list(raws))
# features = info[:, 3:]
# _min = features[:,:3].min(axis=0)
# _max = features[:,:3].max(axis=0)
# setup_download_from_s3(img_fn, recursive=False)

db_fp = args.db_file
conn = sqlite3.connect(db_fp)
cur = conn.cursor()
raws = cur.execute('SELECT * FROM features')
info = np.array(list(raws))
locations = info[:, 1:3]
features = info[:, 3:]

# locations = locations[features[:,13]==201]
# features = features[features[:,13]==201]
# select = features[features[:,13]==15][:,:3]
# features[features[:,13]==15][:,:3]=(select-select.min(axis=0))/(select.max(axis=0)-select.min(axis=0))
# select = features[features[:,13]==51][:,:3]
# features[features[:,13]==51][:,:3]=(select-select.min(axis=0))/(select.max(axis=0)-select.min(axis=0))
# select = features[features[:,13]==201][:,:3]
# features[features[:,13]==201][:,:3]=(select-select.min(axis=0))/(select.max(axis=0)-select.min(axis=0))
# top3 = (features[:,:3]-_min)/(_max-_min)
# top3[top3<0] = 0
# top3[top3>1] = 1
# top3 = (features[:,:3]-features[:,:3].min(axis=0))/(features[:,:3].max(axis=0)-features[:,:3].min(axis=0))
top3 = features[:, :5]
img = io.imread(img_fn)
# img = img[:20000,:20000]
m, n = img.shape

thresh = cv2.adaptiveThreshold(255 - img.copy(), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 101, -20)
Stats = cv2.connectedComponentsWithStats(thresh)
mask = Stats[1]

# hsv_result = np.zeros([m, n, 3],dtype='float32')
# hsv_result[:,:,2] = img.copy()/255
rgb = color.gray2rgb(img.copy())

for idx in range(len(top3)):
    if idx % 1000 == 0:
        print(idx, len(top3))
    height = int(features[idx, 11])
    width = int(features[idx, 10])
    #     height = int(features[idx, 1])
    #     width = int(features[idx, 0])
    size = int(features[idx, 13])
    if size == 15:
        continue
    #     if size==201:
    #         h = 0
    #     elif size==51:
    #         h = 0.6
    #     else:
    #         continue
    cx = int(locations[idx, 0] - width / 2.0)
    cy = int(locations[idx, 1] - height / 2.0)
    try:
        objects = np.unique(mask[cy:cy + height, cx:cx + width])  # [1:]
        counts = [(mask[cy:cy + height, cx:cx + width] == object_id).sum() for object_id in objects if object_id]
        object_id = objects[np.argmax(np.array(counts))] if objects[0] else objects[np.argmax(np.array(counts)) + 1]
    except:
        continue

    #     h, s, v = np.array(colorsys.rgb_to_hsv(top3[idx, 0], top3[idx, 1], top3[idx, 2]))
    #     hsv_result[cy:cy + height, cx:cx + width, 0] = h * (mask[cy:cy + height, cx:cx + width] == object_id) + \
    #                                 hsv_result[cy:cy + height, cx:cx + width, 0] * (mask[cy:cy + height, cx:cx + width] != object_id)
    #     hsv_result[cy:cy + height, cx:cx + width, 1] = s * (mask[cy:cy + height, cx:cx + width] == object_id) + \
    #                                                    hsv_result[cy:cy + height, cx:cx + width, 1] * (
    #                                                                mask[cy:cy + height, cx:cx + width] != object_id)
    #     hsv_result[cy:cy + height, cx:cx + width, 2] = v * (mask[cy:cy + height, cx:cx + width] == object_id) + \
    #                                                    hsv_result[cy:cy + height, cx:cx + width, 2] * (
    #                                                                mask[cy:cy + height, cx:cx + width] != object_id)
    #     if top3[idx, 0]>=8:
    #         h = 0
    #     elif top3[idx, 0]>=1:
    #         if top3[idx, 2]>=2:
    #             h=0.1
    #         else:
    #             h=0.2
    #     elif top3[idx, 0]>=-2:
    #         h = 0.3
    #     elif top3[idx, 0]>=-6:
    #         h = 0.4
    #     elif top3[idx, 0]>=-8:
    #         h = 0.5
    #     elif top3[idx, 0]>=-12:
    #         h = 0.6
    #     else:
    #         h = 0.7

    #     if top3[idx, 0]>=8:
    #         h = 0
    #     elif top3[idx, 0]>=2:
    #         if top3[idx, 4]>=2:
    #             h=0.1
    #         else:
    #             h=0.2
    #     elif top3[idx, 0]>=-2:
    #         h = 0.3
    #     elif top3[idx, 0]>=-8:
    #         if top3[idx, 4]>=-2:
    #             h=0.4
    #         else:
    #             h=0.5
    #     elif top3[idx, 0]>=-13:
    #         h = 0.6
    #     else:
    #         h = 0.7
    #     hsv_result[cy:cy + height, cx:cx + width, 0] = h * (mask[cy:cy + height, cx:cx + width] == object_id) + \
    #                                 hsv_result[cy:cy + height, cx:cx + width, 0] * (mask[cy:cy + height, cx:cx + width] != object_id)
    #     hsv_result[cy:cy + height, cx:cx + width, 1] = 0.8 * (mask[cy:cy + height, cx:cx + width] == object_id) + \
    #                                                    hsv_result[cy:cy + height, cx:cx + width, 1] * (
    #                                                                mask[cy:cy + height, cx:cx + width] != object_id)
    #     cell = color.hsv2rgb(hsv_result[cy:cy + height, cx:cx + width, :])
    #     cell = cell * 255
    #     cell = cell.astype(np.uint8)
    #     rgb[cy:cy + height, cx:cx + width, :] = cell.copy()
    #     if top3[idx, 0]>=8:
    #         if top3[idx, 2]<=-1.8:
    #             h=0
    #         else:
    #             h=0.6
    #     else:
    #         if top3[idx, 2]<=-1.8:
    #             h=0.2
    #         else:
    #             h=0.4
    if size == 51:
        if -90 <= features[idx, 14] < 0 or features[idx, 14] > 90:
            h = 0
        else:
            h = 0.6
    else:
        if top3[idx, 2] > -1.8:
            if -90 <= features[idx, 14] < 0 or features[idx, 14] > 90:
                h = 0.2
            else:
                h = 0.4
        else:
            h = 0.8
    color_contour = np.array(colorsys.hsv_to_rgb(h, 0.9, 1)) * 255
    color_contour = [int(channel) for channel in color_contour]

    sub_mask = np.array((mask[cy:cy + height, cx:cx + width] == object_id) * 1, dtype=np.uint8)
    contours, hierarchy = cv2.findContours(sub_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    convex_contour = cv2.convexHull(contours[0][:, 0, :], returnPoints=True)
    cv2.drawContours(rgb[cy:cy + height, cx:cx + width, :], [convex_contour], 0, color_contour, 3)

rgb8 = rescale(rgb, 1.0 / 4, multichannel=True, anti_aliasing=True)
rgb8 = rgb8 * 255
rgb8 = rgb8.astype(np.uint8)
# io.imsave(os.environ['ROOT_DIR']+stack+'_'+section+'.jpg',rgb)
io.imsave(os.environ['ROOT_DIR'] + stack + '_' + section + '_down4.jpg', rgb8)

