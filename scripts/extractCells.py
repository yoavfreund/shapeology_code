import cv2
import pickle
import numpy as np

from label_patch import diffusionMap
from patch_normalizer import normalizer
from lib.utils import mark_contours, configuration
from extractPatches import patch_extractor
