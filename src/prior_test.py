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
#from VIT import ViT
from UNIT import SeTr
from dice_loss import DiceLoss
#from chunk_image import load_data
from crop_image import load_data
from UNET import UNet
import cv2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

chunk_y = 240
chunk_z = 240

#This function acquired from stack overflow
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def masked_mse(output, target, mask):
    diff = output - target
    diff = diff * mask
    return torch.sum(diff*diff)

def train_setr():
    full_image = torch.from_numpy(cv2.cvtColor(cv2.imread("dataset/prior/library.png")[:400,:400], cv2.COLOR_BGR2RGB)).unsqueeze(0).permute(0, 3, 2, 1).to(device) / 255.0
    mask = torch.from_numpy(cv2.imread("dataset/prior/library_mask.png")[:400,:400]).unsqueeze(0).to(device).permute(0, 3, 2, 1) / 255.0

    noise = torch.normal(mean=torch.zeros((1,400,400,3))).permute(0, 3, 2, 1).to(device)

    plt.imshow(full_image.cpu().permute(0, 3, 2, 1)[0])
    plt.show()

    plt.imshow(mask.cpu().permute(0, 3, 2, 1)[0])
    plt.show()

    model = SeTr(image_size=400, patch_size=10, dim=128, depth=1, heads=10, mlp_dim=128, pool='mean').to(device)
    parameter_count = count_parameters(model)
    print('model parameter count: ' + str(parameter_count))

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0003)

    epoch = 0
    while epoch != 10000:


        true_output = model(noise)
        
        optimizer.zero_grad()
        loss = masked_mse(true_output, full_image, mask)
        loss.backward()
        optimizer.step()

        running_loss = loss.item()

        if(epoch%10==9):
            plt.imshow(true_output.permute(0, 3, 2, 1).cpu().detach().numpy()[0])
            plt.show()

        print('[%d] loss: %.3f' % (epoch + 1, running_loss))

        epoch = epoch + 1

    return model

def main():
    model = train_setr()

main()
