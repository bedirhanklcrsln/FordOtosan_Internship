from model import FoInternNet
from preprocess import tensorize_image, tensorize_mask, image_mask_check
import os
import glob
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import joblib
from matplotlib.pylab import plt
from pickle import dump
from pickle import load
from numpy import arange


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

trlist = []
tracc = []
#epochs = range(1, 2)
for epoch in range(epochs):
    for (train_input, train_label) in zip(train_input_path_list, train_label_path_list):
        Xbatch = tensorize_image([train_input], input_shape, cuda)
        ybatch = tensorize_mask([train_label], input_shape, n_classes, cuda)

        model.zero_grad()
        output = model(Xbatch)
        loss = criterion(output, ybatch)
        loss.backward()
        optimizer.step()
        taccuracy = (output.round() == ybatch).float().mean()
    print(f"loss : {loss} in epoch : {epoch}")
    print(f"Train Accuracy {taccuracy}")
    tracc.append(taccuracy)  
    trlist.append(loss.item())

   # print(trlist)

 
# Generate a sequence of integers to represent the epoch numbers





# save the model to disk
filename = 'finalized_model.sav'
joblib.dump(model, filename)


loaded_model = joblib.load(filename)


#taccuracy = (output.round() == ybatch).float().mean()
#print(f"Train Accuracy {taccuracy}")
valacc= []
valloss = []
#with torch.no_grad():
for epoch in range(epochs):
    for (valid_input_path, valid_label_path) in zip(valid_input_path_list, valid_label_path_list):
            Xbatch = tensorize_image([valid_input_path], input_shape, cuda)
            ybatch = tensorize_mask([valid_label_path], input_shape, n_classes, cuda)
            model.zero_grad()
            output = model(Xbatch)
            loss = criterion(output, ybatch)
            loss.backward()
            optimizer.step()
            vaccuracy = (output.round() == ybatch).float().mean()
    print(f"Validation Accuracy {vaccuracy}")
    print(f"Validation loss : {loss} in epoch : {epoch}")
    valacc.append(vaccuracy)
    valloss.append(loss.item())



epochss = range(1, 21)
plt.plot(epochss, trlist, label = 'Training Loss')
plt.plot(epochss, valloss, label = 'Validation Loss')
plt.title('Training - Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.xticks(arange(0, 21, 1))
plt.yticks(arange(0, 2, 1))
plt.legend()
plt.show()

plt.plot(epochss,tracc,label = 'Training Accuracy')
plt.plot(epochss,valacc, label = 'Validation Accuracy')
plt.title('Training - Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.xticks(arange(0, 21, 1))
plt.yticks(arange(0, 2, 1))
plt.legend()
plt.show()
#print(f"Validation Accuracy {vaccuracy}")


 
# some time later...
 

 
   
# compute accuracy (no_grad is optional)
