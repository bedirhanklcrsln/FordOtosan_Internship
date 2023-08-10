import json
import os
import cv2
import tqdm
import glob
import numpy as np
import matplotlib.pyplot as plt
from preprocess import image_mask_check
from constant import *


# Create a list which contains every file name in masks folder
mask_list = os.listdir(MASK_DIR)
# Remove hidden files if any
for f in mask_list:
    if f.startswith('.'):
        mask_list.remove(f)


image_path_list = glob.glob(os.path.join(IMAGE_DIR, '*'))
image_path_list.sort()

mask_path_list = glob.glob(os.path.join(MASK_DIR, '*'))
mask_path_list.sort()
# For every mask image
if image_mask_check(image_path_list, mask_path_list):
    for mask_name in tqdm.tqdm(mask_list):
    # Name without extension
        mask_name_without_ex = mask_name.split('.')[0]

    
    # Access required folders
        mask_path      = os.path.join(MASK_DIR, mask_name)
        image_path     = os.path.join(IMAGE_DIR, mask_name)
        image_out_path = os.path.join(IMAGE_OUT_DIR, mask_name)
    

    # Read mask and corresponding original image    
        mask  = cv2.imread(mask_path, 0).astype(np.uint8)
        image = cv2.imread(image_path).astype(np.uint8)
 
        # Change the color of the pixels on the original image that corresponds
        # to the mask part and create new image
        same_img  = image.copy()
        image[mask==1, :] = (255,200,200)
        image[mask==2, :] = (0,255,0)
        image[mask==3, :] = (255,0,0)
        image[mask==4, :] = (0,0,255)
        new_img = (image/2 + same_img/2).astype(np.uint8)

    # Write output image into IMAGE_OUT_DIR folder
        cv2.imwrite(image_out_path, new_img)
    # Visualize created image if VISUALIZE option is chosen
        if VISUALIZE:
            cv2.waitKey(1)
