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
from UNIT import UNIT
from dice_loss import DiceLoss
#from chunk_image import load_data
from crop_image import load_data

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

chunk_x = 50
chunk_y = 50




def dice_loss(pred, target, smooth = 1.):
    pred = pred.contiguous()
    target = target.contiguous()    

    intersection = (pred * target).sum(dim=2).sum(dim=2)
    
    loss = (1 - ((2. * intersection + smooth) / (pred.sum(dim=2).sum(dim=2) + target.sum(dim=2).sum(dim=2) + smooth)))
    
    return loss.mean()

# 0 goes to [1, 0], 1 goes to [0, 1]
def to_one_hot_2d(n):
    result = np.array((2,))
    if n == 0.0:
        result = np.array([1, 0])
    else:
        result = np.array([0, 1])

    result = torch.from_numpy(result)

    return result

def train_unit():
    model = UNIT(image_size=50, patch_size=5, dim=30, depth=4, heads=10, mlp_dim=200, channels=4).to(device)
    #, dropout=0.3, emb_dropout=0.3
    #criterion = nn.BCELoss()
    #criterion = DiceLoss()
    #criterion = nn.CrossEntropyLoss()
    criterion = DiceLoss()

    optimizer = optim.Adam(model.parameters())
    #scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5)
    epoch = 0
    while epoch != 10000:

        batch_size = 5
        current_batch_index = (int(epoch/1) % 29)
        lower_index = (current_batch_index * batch_size) + 11
        upper_index = ((1+current_batch_index) * batch_size) + 11
        #lower_index = 1
        #upper_index = 3
        print('lower_index:' + str(lower_index))
        print('upper_index:' + str(upper_index))
        chunked_images, chunked_segmented_images = load_data(lower_index, upper_index, chunk_x, chunk_y)

        for batch_epoch in range(0, 1):
            optimizer.zero_grad()

            running_loss = 0.0
        

            total_size = chunked_images.shape[0]
            subbatch_size = int(total_size / 1)
            current_subbatch_index = 0#epoch % 1
        

            model_input = np.copy(chunked_images[current_subbatch_index*subbatch_size:((1 + current_subbatch_index)*subbatch_size)])
            model_output = chunked_segmented_images[current_subbatch_index*subbatch_size:((1 + current_subbatch_index)*subbatch_size)]
        
            model_input= np.copy(chunked_images)
            model_output = np.copy(chunked_segmented_images)

            model_input = model_input.transpose((0, 3, 1, 2))
            model_input = torch.from_numpy(model_input).to(device)

            model_output = model_output.transpose((0, 1, 2))
            model_output = torch.LongTensor(model_output).to(device)

            outputs = model(model_input)
            
            loss = criterion(outputs, model_output)
            loss.backward()
            optimizer.step()
            #scheduler.step(loss)

            running_loss += loss.item()

            print('[%d] loss: %.3f' % (epoch + 1, running_loss))

            if(epoch%20 == 2):
                evaluate(model, True)
                model.train()
            elif(epoch % 10 == 2):
                evaluate(model, False)
                model.train()
            
            epoch = epoch + 1 

            del model_input
            del model_output
            del outputs
            del loss
            torch.cuda.empty_cache()
        
        del chunked_images
        del chunked_segmented_images
        torch.cuda.empty_cache()

    return model

test_losses = []
def evaluate(model, show_it):
    model.eval()
    chunked_images, chunked_segmented_images = load_data(1, 3, chunk_x, chunk_y) 

    model_input = np.copy(chunked_images)
    model_output = chunked_segmented_images
        
    model_input = model_input.transpose((0, 3, 1, 2))
    model_input = torch.from_numpy(model_input).to(device)

    print(model_input.shape)

    model_output = model_output.transpose((0, 1, 2))
    model_output = torch.LongTensor(model_output).to(device)



    outputs = model(model_input)

    #criterion = nn.MSELoss()
    criterion = nn.CrossEntropyLoss()
    loss = criterion(outputs, model_output)
    loss_amount = loss.item()
    test_losses.append(loss_amount)
    print('----------------------Test Loss: ' + str(loss_amount))

    #show_it = True
    if show_it:
        model_input_cpu = model_input.cpu().detach().numpy()
        outputs_cpu = np.argmax(outputs.cpu().detach().numpy(), axis=1)
        model_output_cpu = model_output.cpu().detach().numpy()
        f, ax = plt.subplots(6, 2)
        ax[0, 0].imshow(model_input_cpu[11, 1:4, :, :].transpose((1, 2, 0)))
        ax[0, 1].imshow(chunked_images[12, :, :, 1:4])
        ax[1, 0].imshow(outputs_cpu[22, :, :])
        ax[1, 1].imshow(model_output_cpu[22, :, :])
        ax[2, 0].imshow(outputs_cpu[10, :, :])
        ax[2, 1].imshow(model_output_cpu[10, :, :])
        ax[3, 0].imshow(outputs_cpu[23, :, :])
        ax[3, 1].imshow(model_output_cpu[23, :, :])
        ax[4, 0].imshow(outputs_cpu[14, :, :])
        ax[4, 1].imshow(model_output_cpu[14, :, :])
        ax[4, 0].imshow(outputs_cpu[5, :, :])
        ax[4, 1].imshow(model_output_cpu[5, :, :])
        ax[5, 0].imshow(outputs_cpu[2, :, :])
        ax[5, 1].imshow(model_output_cpu[2, :, :])
        plt.show()

        plt.plot(test_losses)
        plt.show()
    
    del model_input
    del model_output
    del outputs
    del chunked_images
    del loss

def main():
    model = train_unit()
    evaluate(model, True)

main()
