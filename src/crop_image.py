
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
import random

def load_data(range_min, range_max):


    chunked_images_arr = []
    chunked_segmented_images_arr = []

    for i in range (range_min, range_max):
        index = str(i)
        if len(index) == 1:
            index = '00' + index
        elif len(index) == 2:
            index = '0' + index
        
        data_filename = './dataset/brats2020-training/full/BraTS20_Training_' + index + '_imgs.npy'
        segmented_filename = './dataset/brats2020-training/full/BraTS20_Training_' + index + '_seg.npy'
        
        data = np.load(data_filename)
        segmented = np.load(segmented_filename)

        data = data.transpose((3, 1, 2, 0))
        segmented = segmented.transpose((0, 1, 2))
        chunked_images_arr = chunked_images_arr + [data]
        chunked_segmented_images_arr = chunked_segmented_images_arr + [segmented]
    
 
    #print("copying from list to array")
    chunked_images = np.zeros((len(chunked_images_arr), 155, 240, 240, 4), dtype=np.float32)
    chunked_segmented_images = np.zeros((len(chunked_images_arr), 155, 240, 240), dtype=np.float32)

    for i in range(0, len(chunked_images_arr)):
        chunked_images[i] = chunked_images_arr[i]
        chunked_segmented_images[i] = chunked_segmented_images_arr[i]

    #print('finished copying')
    
    return (chunked_images, chunked_segmented_images)