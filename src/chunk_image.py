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

def chunk_slice(chunk_x, chunk_y, image, segmented_image):
    assert(image.shape[0] == segmented_image.shape[0])
    assert(image.shape[1] == segmented_image.shape[1])

    columns = int(image.shape[0] / chunk_x)
    rows = int(image.shape[1] / chunk_y)
 
    chunked_image = np.zeros((columns * rows, chunk_x, chunk_y, image.shape[2]), dtype=np.float32)
    chunked_segmented_image = np.zeros((columns * rows, chunk_x, chunk_y), dtype=np.float32)

    chunked_image_arr = []
    chunked_segmented_image_arr = []

    temp_vector = np.zeros((chunk_x, chunk_y, image.shape[2]))

    for i in range(0, (columns-1)):
        for j in range(0, (rows-1)):

            temp_vector = np.copy(image[i * chunk_x:(chunk_x + i*chunk_x), j*chunk_y:(chunk_y + j*chunk_y), :]) 
            
            chunked_image_arr.append(temp_vector)
            temp_vector = np.copy(segmented_image[i * chunk_x:(chunk_x + i*chunk_x), j*chunk_y:(chunk_y + j*chunk_y)])
            chunked_segmented_image_arr.append(temp_vector)

            

    return (chunked_image_arr, chunked_segmented_image_arr)


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
        segmented = np.clip(segmented, 0.0, 1.0)


        for j in range(0, data.shape[2]):
            chunked_image, chunked_segmented_image = chunk_slice(chunk_x, chunk_y, data[:, :, j], segmented[:, :, j])

            chunked_images_arr = chunked_images_arr + chunked_image
            chunked_segmented_images_arr = chunked_segmented_images_arr + chunked_segmented_image

        #print('loaded file' + data_filename)
    

    #print("copying from list to array")
    chunked_images = np.zeros((len(chunked_images_arr), chunk_x, chunk_y, 4), dtype=np.float32)
    chunked_segmented_images = np.zeros((len(chunked_images_arr), chunk_x, chunk_y), dtype=np.float32)
    images_have_tumor = np.zeros((len(chunked_images_arr),), dtype=np.float32)

    for i in range(0, len(chunked_images_arr)):
        chunked_images[i] = chunked_images_arr[i]
        chunked_segmented_images[i] = chunked_segmented_images_arr[i]
        #images_have_tumor[i] = images_have_tumor_arr[i]
    #print('finished copying')
    
    return (chunked_images, chunked_segmented_images)