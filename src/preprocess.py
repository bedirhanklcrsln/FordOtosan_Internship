"""
We need to prepare the data to feed the network: we have - data/masks, data/images - directories where we prepared masks and input images. Then, convert each file/image into a tensor for our purpose.

We need to write two functions in src/preprocess.py:
    - one for feature/input          -> tensorize_image()
    - the other for mask/label    -> tensorize_mask()


Our model will accepts the input/feature tensor whose dimension is
[batch_size, output_shape[0], output_shape[1], 3]
&
the label tensor whose dimension is
[batch_size, output_shape[0], output_shape[1], 2].

At the end of the task, our data will be ready to train the model designed.

"""


import glob
import cv2
import torch
import numpy as np
from constant import *
import numpy as geek


#tamam
def tensorize_image(image_path_list, output_shape, cuda=False):
   
    """

    
    Parameters
    ----------
    image_path_list : list of strings
        [“data/images/img1.png”, .., “data/images/imgn.png”] corresponds to
        n images to be trained each step.
    output_shape : tuple of integers
        (n1, n2): n1, n2 is width and height of the DNN model’s input.
    cuda : boolean, optional
        For multiprocessing,switch to True. The default is False.

    Returns
    -------
    torch_image : Torch tensor
        Batch tensor whose size is [batch_size, output_shape[0], output_shape[1], C].       For this case C = 3.

    """

    # Create empty list
    local_image_list = []
    global torch_image
    # For each image
    for image_path in image_path_list:

       
        # Access and read image
        image = cv2.imread(image_path)
        
        # Resize the image according to defined shape
        image = cv2.resize(image, output_shape)

        # Change input structure according to pytorch input structure
        torchlike_image = torchlike_data(image)

        # Add into the list
        local_image_list.append(torchlike_image)
        # Convert from list structure to torch tensor
        torch_image = torch.tensor(local_image_list,dtype=torch.float32)
        

    # If multiprocessing is chosen
    if cuda:
        torch_image = torch_image.cuda()

    return torch_image

# one hot encoder çalışınca tamam
def tensorize_mask(mask_path_list, output_shape, n_class, cuda=False):
    """

    
    Parameters
    ----------
    mask_path_list : list of strings
        [“data/masks/mask1.png”, .., “data/masks/maskn.png”] corresponds
        to n masks to be used as labels for each step.
    output_shape : tuple of integers
        (n1, n2): n1, n2 is width and height of the DNN model’s input.
    n_class : integer
        Number of classes.
    cuda : boolean, optional
        For multiprocessing, switch to True. The default is False.

    Returns
    -------
    torch_mask : TYPE
        DESCRIPTION.

    """
    global torch_mask
    # Create empty list
    local_mask_list = []

    # For each masks
    for mask_path in mask_path_list:


       # try:
            #mask = cv2.imread(mask_path,0)
            #mask=cv2.resize(mask,output_shape)
            #print("selss")
        #except Exception as e:
           # print(str(e))
        # Access and read mask
        mask = cv2.imread(mask_path,0)

        # Resize the image according to defined shape
        mask = cv2.resize(mask, output_shape)

        # Apply One-Hot Encoding to image
        mask = one_hot_encoder(mask, n_class)
        
        # Change input structure according to pytorch input structure
        torchlike_mask = torchlike_data(mask)

        local_mask_list.append(torchlike_mask)

        torch_mask = torch.tensor(local_mask_list,dtype=torch.float32)

    if cuda:
        torch_mask = torch_mask.cuda()

    return torch_mask

#tamam
def image_mask_check(image_path_list, mask_path_list):
    """
    Since it is supervised learning, there must be an expected output for each
    input. This function assumes input and expected output images with the
    same name.

    Parameters
    ----------
    image_path_list : list of strings
        [“data/images/images1.png”, .., “data/images/imagesn.png”] corresponds
        to n original image to be used as features.
    mask_path_list : list of strings
        [“data/masks/mask1.png”, .., “data/masks/maskn.png”] corresponds
        to n masks to be used as labels.

    Returns
    -------
    bool
        Returns true if there is expected output/label for each input.

    """

    # Check list lengths
    if(len(image_path_list) != len(mask_path_list)):
        print("Image and masks folders do not match")
        return False
    for image_path, mask_path in zip(image_path_list,mask_path_list):
        image_name = image_path.split("/")[-1].split("\\")[-1]
        mask_name = mask_path.split("/")[-1].split("\\")[-1]
        if(image_name!= mask_name):
            print("sa")
            print("Image and mask folders do not match")
            return False
        else:
            print("Image and mask folders match")
            return True
    else:
        True
   # return True



############################ TODO ################################
def torchlike_data(data):
    """
    Change data structure according to Torch Tensor structure where the first
    dimension corresponds to the data depth.


    Parameters
    ----------
    data : Array of uint8
        Shape : HxWxC.

    Returns
    -------
    torchlike_data_output : Array of float64
        Shape : CxHxW.

    """
    # Obtain channel value of the input
    
    n_channels = data.shape[2]
    # Create and empty image whose dimension is similar to input
    torchlike_data_output = geek.empty((n_channels, data.shape[0], data.shape[1]))
    # For each channel
    for channel in range(n_channels):
     torchlike_data_output[channel] = data[:,:,channel]

    return torchlike_data_output

def one_hot_encoder(data, n_class):
    """
    Returns a matrix containing as many channels as the number of unique
    values in the input Matrix, where each channel represents a unique class.


    Parameters
    ----------
    data : Array of uint8
        2D matrix.
    n_class : integer
        Number of class.

    Returns
    -------
    encoded_data : Array of int64
        Each channel labels for a class.

    """

    if len(data.shape) != 2:
        print("It should be same with the layer dimension, in this case it is 2")
    
    
    #if len(np.unique(data)) != n_class:
        #print("The number of unique values in 'data' must be equal to the n_class")
        

    # Define array whose dimensison is (width, height, number_of_class)
    encoded_data = np.zeros((*data.shape, n_class), dtype=np.uint)
    # Define labels
    labels = [[0,0,0,0,0],[0,0,0,0,1],[0,0,0,1,0],[0,0,0,1,1],[0,0,1,0,0]]

    for i in range(n_class):
        encoded_data[data == i] = labels[i]    

    return encoded_data
############################ TODO END ################################





if __name__ == '__main__':

    # Access images
    image_list = glob.glob(os.path.join(IMAGE_DIR, '*'))
    image_list.sort()

    batch_image_list = image_list[:BACTH_SIZE]
    batch_image_tensor = tensorize_image(batch_image_list, (224, 224))

    # Access masks
    mask_list = glob.glob(os.path.join(MASK_DIR, '*'))
    mask_list.sort()

    batch_mask_list = mask_list[:BACTH_SIZE]
    batch_mask_tensor = tensorize_mask(batch_mask_list,(224,224),5)

    # Check image-mask match
    if image_mask_check(image_list, mask_list):

        print("for features:\ndtype is " + str(batch_image_tensor.dtype))
        print("type is "+str(type(batch_image_tensor)))
        print("the size should be [" + str(BACTH_SIZE)+",3 "+str(HEIGHT)+", "+str(WIDTH)+"]")
        print("size is "+str(batch_image_tensor.shape)+"\n")

        print("for features:\ndtype is " + str(batch_mask_tensor.dtype))
        print("type is "+str(type(batch_mask_tensor)))
        print("the size should be [" + str(BACTH_SIZE)+",5 "+str(HEIGHT)+", "+str(WIDTH)+"]")
        print("size is "+str(batch_mask_tensor.shape)+"\n")
        
