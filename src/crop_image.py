
import torch
import torch.autograd as autograd
import math
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Transformer
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

def load_data(range_min, range_max, chunk_x, chunk_y):
    chunked_images_arr = []
    chunked_segmented_images_arr = []

    for i in range (range_min, range_max):
        index = str(i)
        if len(index) == 1:
            index = '00' + index
        elif len(index) == 2:
            index = '0' + index
        
        data_filename = './dataset/brats2020-training/crop/BraTS20_Training_' + index + '_imgs.npy'
        segmented_filename = './dataset/brats2020-training/crop/BraTS20_Training_' + index + '_seg.npy'
        
        data = np.load(data_filename)
        segmented = np.load(segmented_filename)

        data = data.transpose((2, 1, 3, 0))
        segmented = segmented.transpose((1, 0, 2))
        #segmented = np.clip(segmented, 0.0, 1.0)


        for j in range(0, data.shape[2]):
            
            #Offset to make x by y image centered
            x_offset = int((data.shape[0] - chunk_x) / 2)
            y_offset = int((data.shape[1] - chunk_y) / 2)
 
            chunked_image           =      data[(x_offset):(x_offset+chunk_x), (y_offset):(y_offset+chunk_y), j]
            chunked_segmented_image = segmented[(x_offset):(x_offset+chunk_x), (y_offset):(y_offset+chunk_y), j]

            
            #Uncomment to look at tumors in the center 
            '''
            if(j==100):
                plt.imshow(chunked_segmented_image[:,:])
                plt.show()
                plt.imshow(chunked_image[:,:,1:4])
                plt.show()
            '''
            if(chunked_image.shape[0] != chunk_x or chunked_image.shape[1] != chunk_y):
                print('size mismatch, skipping')
            else:
                chunked_images_arr = chunked_images_arr + [chunked_image]
                chunked_segmented_images_arr = chunked_segmented_images_arr + [chunked_segmented_image]

        print('loaded file' + data_filename)
    
 
    print("copying from list to array")
    chunked_images = np.zeros((len(chunked_images_arr), chunk_x, chunk_y, 4), dtype=np.float32)
    chunked_segmented_images = np.zeros((len(chunked_images_arr), chunk_x, chunk_y), dtype=np.float32)

    for i in range(0, len(chunked_images_arr)):
        chunked_images[i] = chunked_images_arr[i]
        chunked_segmented_images[i] = chunked_segmented_images_arr[i]

    print('finished copying')
    
    return (chunked_images, chunked_segmented_images)