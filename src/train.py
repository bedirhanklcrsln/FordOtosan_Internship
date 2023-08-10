from model import FoInternNet
from preprocess import tensorize_image, tensorize_mask, image_mask_check
import os
import glob
import numpy as np
import torch.nn as nn
import torch.optim as optim

######### PARAMETERS ##########
valid_size = 0.3
test_size  = 0.1
batch_size = 4
epochs = 20
cuda = False
input_shape = (224, 224)
n_classes = 5
###############################

######### DIRECTORIES #########
SRC_DIR = os.getcwd()
ROOT_DIR = os.path.join(SRC_DIR, '..')
DATA_DIR = os.path.join(ROOT_DIR, 'data')
IMAGE_DIR = '../../data/images'
MASK_DIR = '../../data/masks'

###############################


# PREPARE IMAGE AND MASK LISTS
image_path_list = glob.glob(os.path.join(IMAGE_DIR, '*'))
image_path_list.sort()

mask_path_list = glob.glob(os.path.join(MASK_DIR, '*'))
mask_path_list.sort()

# DATA CHECK
image_mask_check(image_path_list, mask_path_list)

# SHUFFLE INDICES
indices = np.random.permutation(len(image_path_list))

# DEFINE TEST AND VALID INDICES
test_ind  = int(len(indices) * test_size)
valid_ind = int(test_ind + len(indices) * valid_size)

#print("test : "+ str(indices))
#print("valid : "+ str(valid_ind))

# SLICE TEST DATASET FROM THE WHOLE DATASET
test_input_path_list = image_path_list[:test_ind]
test_label_path_list = mask_path_list[:test_ind]

# SLICE VALID DATASET FROM THE WHOLE DATASET
valid_input_path_list = image_path_list[test_ind:valid_ind]
valid_label_path_list = mask_path_list[test_ind:valid_ind]



# SLICE TRAIN DATASET FROM THE WHOLE DATASET
train_input_path_list = image_path_list[valid_ind:]
train_label_path_list = mask_path_list[valid_ind:]

##print(len(train_input_path_list))
#print(train_input_path_list)
##print("\n")
#print(train_label_path_list)

# DEFINE STEPS PER EPOCH
steps_per_epoch = len(train_input_path_list)//batch_size

# CALL MODEL
model = FoInternNet(input_size=input_shape, n_classes=5)

# DEFINE LOSS FUNCTION AND OPTIMIZER
criterion = nn.BCELoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# IF CUDA IS USED, IMPORT THE MODEL INTO CUDA
if cuda:
    model = model.cuda()


for ind in range(steps_per_epoch):
        optimizer.zero_grad()
        batch_input_path_list = test_input_path_list[batch_size*ind:batch_size*(ind+1)]
        batch_label_path_list = test_label_path_list[batch_size*ind:batch_size*(ind+1)]
        Xbatch = tensorize_image(batch_input_path_list, input_shape, cuda)
        ybatch = tensorize_mask(batch_label_path_list, input_shape, n_classes, cuda)
        output = model(Xbatch)
        loss = criterion(output, ybatch)
        loss.backward()
        optimizer.step()
        accuracy = (output.round() == ybatch).float().mean()
        print(f"Accuracy {accuracy}")
        print(f'Finished ind : {ind}, latest loss : {loss}')
        if(len(batch_input_path_list)==0):
            break

# TRAINING THE NEURAL NETWORK

 
   
# compute accuracy (no_grad is optional)
