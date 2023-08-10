import json
import numpy as geek
import os
import numpy as np
import cv2
import tqdm
from PIL import Image
from constant import JSON_DIR, MASK_DIR
import matplotlib.pyplot as plt


# Create a list which contains every file name in "jsons" folder
json_list = os.listdir(JSON_DIR)

""" tqdm Example Start"""

iterator_example = range(1000000)

for i in tqdm.tqdm(iterator_example):
    pass

""" rqdm Example End"""


# For every json file
for json_name in tqdm.tqdm(json_list):

    # Access and open json file as dictionary
    json_path = os.path.join(JSON_DIR, json_name)
    json_file = open(json_path, 'r')

    # Load json data
    json_dict = json.load(json_file)

    # Create an empty mask whose size is the same as the original image's size   

    mask = geek.empty([json_dict["size"]["height"],json_dict["size"]["width"]],dtype=np.uint8)
    

    mask_path = os.path.join(MASK_DIR, json_name[:-5])

    # For every objects
    for obj in json_dict["objects"]:
        # Check the objects ‘classTitle’ is ‘Freespace’ or not.
        if obj['classTitle']=='Freespace':
            mask = cv2.fillPoly(mask, np.array([obj['points']['exterior']]), color=1)

    for obj in json_dict["objects"]:
        if obj['classTitle']=='Dashed Line':
                mask = cv2.polylines(mask,np.array([obj['points']['exterior']]),False,color=2)

    for obj in json_dict["objects"]:
        if obj['classTitle']=='Exit/Merge Line':
                mask = cv2.polylines(mask,np.array([obj['points']['exterior']]),False,color=3)

    for obj in json_dict["objects"]:
        if obj['classTitle']=='Solid Line':
                mask = cv2.polylines(mask,np.array([obj['points']['exterior']]),False,color=4)
    
    
            
    
    
   

    # cv.line() ile dashed line yap


    # Write mask image into MASK_DIR folder
    cv2.imwrite(mask_path, mask.astype(np.uint8))