# 特征图可视化Copyright (c) Facebook, Inc. and its affiliates.
import argparse
import glob
import multiprocessing as mp
import numpy as np
import os
import tempfile
import time
import warnings
import cv2
import tqdm

from Visualizer.visualizer import get_local
get_local.activate()
#from detectron2.config import get_cfg
from fewx.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.utils.logger import setup_logger

from predictor import VisualizationDemo

import numpy as np
import cv2
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import os

import torch
import json
from PIL import Image, ImageDraw
import numpy as np
import matplotlib.pyplot as plt

    
    
def Have_a_Look(image,x):
    x += 1
    print(x)
    #[N，C，H，w] ->[C，H，w]
    im = np.squeeze(image.detach().cpu().numpy())
    #[C，H，W]->[H，W，C]
    im = np.transpose(im,[1,2,0])
    feature_map_combination=[]
    # channle = np.zeros((30,30))
    # channle = np.zeros((15,15))
    for i in range(1024):
        print(i)
        channle_i = im[: ,:,i]
        
        feature_map_combination.append(channle_i)
        # if i == 95:
        #     channle = channle/96 
    # channle = im[: ,:,0]
    feature_map_sum = sum(one for one in feature_map_combination)/1024
    print(feature_map_sum.shape)
    plt.figure()
    plt.xticks([]),plt.yticks([])#
    plt.axis('off')
    plt.margins(0,0)
    plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)
    #plt. subplots_adjust(left=None， bottom=None，right=None，top=None， wspace=None，hspace=None)
    #plt.imshow(feature_map_sum,cmap='gray')
    ratio=1
    img_path = "-----------------"
    print("load image from: ", img_path)
    img = Image.open(img_path, mode='r')
    img_h, img_w = img.size[0], img.size[1]
    plt.subplots(nrows=1, ncols=1, figsize=(0.02 * img_h, 0.02 * img_w))
    # scale the image
    img_h, img_w = int(img.size[0] * ratio), int(img.size[1] * ratio)
    img = img.resize((img_h, img_w))
    plt.imshow(img, alpha=1)
    plt.axis('off')
    # normalize the attention map
    mask = cv2.resize(feature_map_sum, (img_h, img_w))
    normed_mask = mask / mask.max()
    normed_mask = (normed_mask * 255).astype('uint8')
    plt.imshow(normed_mask, alpha=0.6, interpolation='nearest', cmap='Greys') # OrRd YlGnBu ,hot_r,Greys
    plt.savefig( '-------------------- '+str(x),dpi=100,bbox_inches='tight', pad_inches = -0.1)
    
    

    

def visualize_grid_attention_v2(img_path, save_path, attention_mask, ratio=1, cmap="jet", save_image=False,
                             save_original_image=False, quality=200):
    """
    img_path:   image file path to load
    save_path:  image file path to save
    attention_mask:  2-D attention map with np.array type, e.g, (h, w) or (w, h)
    ratio:  scaling factor to scale the output h and w
    cmap:  attention style, default: "jet"
    quality:  saved image quality
    """
    img_path = "---------------"
    print("load image from: ", img_path)
    img = Image.open(img_path, mode='r')
    img_h, img_w = img.size[0], img.size[1]
    plt.subplots(nrows=1, ncols=1, figsize=(0.02 * img_h, 0.02 * img_w))

    # scale the image
    img_h, img_w = int(img.size[0] * ratio), int(img.size[1] * ratio)
    img = img.resize((img_h, img_w))
    plt.imshow(img, alpha=1)
    plt.axis('off')

    # normalize the attention map
    mask = cv2.resize(attention_mask, (img_h, img_w))
    normed_mask = mask / mask.max()
    normed_mask = (normed_mask * 255).astype('uint8')
    plt.imshow(normed_mask, alpha=0.5, interpolation='nearest', cmap=cmap)

    if save_image:
        # build save path
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        img_name = img_path.split('/')[-1].split('.')[0] + "_with_attention.jpg"
        img_with_attention_save_path = os.path.join(save_path, img_name)
        
        # pre-process and save image
        print("save image to: " + save_path + " as " + img_name)
        plt.axis('off')
        plt.subplots_adjust(top=1, bottom=0, right=1,  left=0, hspace=0, wspace=0)
        plt.margins(0, 0)
        plt.savefig(img_with_attention_save_path, dpi=quality)

    if save_original_image:
        # build save path
        if not os.path.exists(save_path):
            os.mkdir(save_path)

        # save original image file
        print("save original image at the same time")
        img_name = img_path.split('/')[-1].split('.')[0] + "_original.jpg"
        original_image_save_path = os.path.join(save_path, img_name)
        img.save(original_image_save_path, quality=quality)