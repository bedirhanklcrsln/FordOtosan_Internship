import json
import os
import cv2
import tqdm
import numpy as np
import matplotlib.pyplot as plt
from constant import *


# Create a list which contains every file name in masks folder
mask_list = os.listdir(MASK_DIR)
# Remove hidden files if any
for f in mask_list:
    if f.startswith('.'):
        mask_list.remove(f)

#json okutup image'a maske uygulama
json_list = os.listdir(JSON_DIR)
for json_name in tqdm.tqdm(json_list):

    # Access and open json file as dictionary
    json_path = os.path.join(JSON_DIR, json_name)
    json_file = open(json_path, 'r')

    # Load json data
    json_dict = json.load(json_file)
    #ekleme

# For every mask image
for mask_name in tqdm.tqdm(mask_list):
    # Name without extension
    mask_name_without_ex = mask_name.split('.')[0]

    # Access required folders
    mask_path      = os.path.join(MASK_DIR, mask_name)
    image_path     = os.path.join(IMAGE_DIR, mask_name)
    image_out_path = os.path.join(IMAGE_OUT_DIR, mask_name)

    # Read mask and corresponding original image

    mask = cv2.imread(image_path).astype(np.uint8)
    image = cv2.imread(image_path).astype(np.uint8)
   
    # Change the color of the pixels on the original image that corresponds
    # to the mask part and create new image
  #  image[mask!=0] = (255,125,125)


   # mask = np.zeros((json_dict["size"]["height"], json_dict["size"]["width"]), dtype=np.uint8)
    for obj in json_dict["objects"]:
        # Check the objects ‘classTitle’ is ‘Freespace’ or not.
        if obj['classTitle']=='Freespace':
            mask = cv2.fillPoly(mask, np.array([obj['points']['exterior']]), color=(255,125,125))
            #image[mask!=0] = (0,0,255)

    for obj in json_dict["objects"]:
        if obj['classTitle']=='Solid Line':
                mask = cv2.polylines(mask,np.array([obj['points']['exterior']]),False,color=(150,255,150),thickness=5)
                #image[mask!=0] = (0,255,0)
                #ekleme


    cv2.imwrite(image_out_path,mask)

    # Visualize created image if VISUALIZE option is chosen
    if VISUALIZE:
        cv2.waitKey(1)
