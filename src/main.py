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
from UNIT import SeTr, UNIT
from dice_loss import DiceLoss
#from chunk_image import load_data
from crop_image import load_data
from UNET import UNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

chunk_x = 155
chunk_y = 240
chunk_z = 240

def dice_coef(y_true, y_pred, smooth=1.0):
    y_true_f = torch.flatten(y_true)
    y_pred_f = torch.flatten(y_pred)
    intersection = torch.sum(y_true_f * y_pred_f)
    result = (2. * intersection + smooth) / (torch.sum(y_true_f) + torch.sum(y_pred_f) + smooth)
    return result

#This function acquired from stack overflow
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def train_unit():
    model = UNIT().to(device)
    parameter_count = count_parameters(model)
    print('model parameter count: ' + str(parameter_count))

    #criterion = nn.CrossEntropyLoss()
    criterion = DiceLoss()

    optimizer = optim.Adam(model.parameters(), lr=0.003)
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer)

    epoch = 0
    while epoch != 10000:
        batch_size = 2
        current_batch_index = (int(epoch/1) % (int(340/batch_size)))
        lower_index = (current_batch_index * batch_size) + 11
        upper_index = ((1+current_batch_index) * batch_size) + 11
        
        if epoch < 100:
            lower_index = 12
            upper_index = 14

        if epoch == 80:
            optimizer = optim.Adam(model.parameters(), lr=0.00003)

        if epoch == 400:
            optimizer = optim.Adam(model.parameters(), lr=0.000003)

        #print('lower_index:' + str(lower_index))
        #print('upper_index:' + str(upper_index))
        chunked_images, chunked_segmented_images = load_data(lower_index, upper_index, chunk_x, chunk_y)

        running_loss = 0.0
        
 
        model_input = np.copy(chunked_images[current_subbatch_index*subbatch_size:((1 + current_subbatch_index)*subbatch_size)])
        model_output = chunked_segmented_images[current_subbatch_index*subbatch_size:((1 + current_subbatch_index)*subbatch_size)]
      
        model_input= np.copy(chunked_images)
        model_output = np.copy(chunked_segmented_images)

        model_input = model_input.transpose((0, 3, 1, 2))
        model_input = torch.from_numpy(model_input).to(device)

        model_output = model_output.transpose((0, 1, 2))
        model_output = torch.LongTensor(model_output).to(device)

        outputs = model(model_input)
            
        optimizer.zero_grad()
        loss = criterion(outputs, model_output)
        loss.backward()
        optimizer.step()
        scheduler.step(loss)

        running_loss += loss.item()

        print('[%d] loss: %.3f' % (epoch + 1, running_loss))

        if (epoch%500 == 3):
            evaluate(model, True)
            model.train()
        elif(epoch % 30 == 2):
            evaluate(model, False)
            model.train()
        elif (epoch%500 == 4):
            evaluate_train(model, True)
            model.train()
        elif (epoch%30 == 5):
            evaluate_train(model, False)
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

train_losses = []

train_dice_losses_0 = []
train_dice_losses_1 = []
train_dice_losses_2 = []
train_dice_losses_3 = []

def evaluate_train(model, show_it):
    model.eval()
    chunked_images, chunked_segmented_images = load_data(12, 14, chunk_x, chunk_y) 

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
    #criterion = DiceLoss()
    loss = criterion(outputs, model_output)
    loss_amount = loss.item()
    train_losses.append(loss_amount)
    print('----------------------Train Loss: ' + str(loss_amount))

    dice_var1 = torch.argmax(outputs, axis=1)
    dice_var1 = F.one_hot(dice_var1, num_classes=4).float()
    dice_var1 = torch.transpose(dice_var1, 1, 3)

    dice_var2 = F.one_hot(model_output, num_classes=4).float()
    dice_var2 = torch.transpose(dice_var2, 1, 3)
    
    dice_loss_0 = dice_coef(dice_var1[:,0,:,:], dice_var2[:,0,:,:])
    dice_loss_1 = dice_coef(dice_var1[:,1,:,:], dice_var2[:,1,:,:])
    dice_loss_2 = dice_coef(dice_var1[:,2,:,:], dice_var2[:,2,:,:])
    dice_loss_3 = dice_coef(dice_var1[:,3,:,:], dice_var2[:,3,:,:])

    train_dice_losses_0.append(dice_loss_0)
    train_dice_losses_1.append(dice_loss_1)
    train_dice_losses_2.append(dice_loss_2)
    train_dice_losses_3.append(dice_loss_3)

    print('----------------------Dice Loss 0: ' + str(dice_loss_0))
    print('----------------------Dice Loss 1: ' + str(dice_loss_1))
    print('----------------------Dice Loss 2: ' + str(dice_loss_2))
    print('----------------------Dice Loss 3: ' + str(dice_loss_3))

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

        plt.plot(train_losses)
        plt.show()

        f2, ax2 = plt.subplots(4)
        ax2[0].plot(train_dice_losses_0)
        ax2[1].plot(train_dice_losses_1)
        ax2[2].plot(train_dice_losses_2)
        ax2[3].plot(train_dice_losses_3)
        plt.show()
    
    del model_input
    del model_output
    del outputs
    del chunked_images
    del loss




test_losses = []

test_dice_losses_0 = []
test_dice_losses_1 = []
test_dice_losses_2 = []
test_dice_losses_3 = []

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
    #criterion = DiceLoss()
    loss = criterion(outputs, model_output)
    loss_amount = loss.item()
    test_losses.append(loss_amount)
    print('----------------------Test Loss: ' + str(loss_amount))

    dice_var1 = torch.argmax(outputs, axis=1)
    dice_var1 = F.one_hot(dice_var1, num_classes=4).float()
    dice_var1 = torch.transpose(dice_var1, 1, 3)

    dice_var2 = F.one_hot(model_output, num_classes=4).float()
    dice_var2 = torch.transpose(dice_var2, 1, 3)
    
    dice_loss_0 = dice_coef(dice_var1[:,0,:,:], dice_var2[:,0,:,:])
    dice_loss_1 = dice_coef(dice_var1[:,1,:,:], dice_var2[:,1,:,:])
    dice_loss_2 = dice_coef(dice_var1[:,2,:,:], dice_var2[:,2,:,:])
    dice_loss_3 = dice_coef(dice_var1[:,3,:,:], dice_var2[:,3,:,:])

    test_dice_losses_0.append(dice_loss_0)
    test_dice_losses_1.append(dice_loss_1)
    test_dice_losses_2.append(dice_loss_2)
    test_dice_losses_3.append(dice_loss_3)

    print('----------------------Dice Loss 0: ' + str(dice_loss_0))
    print('----------------------Dice Loss 1: ' + str(dice_loss_1))
    print('----------------------Dice Loss 2: ' + str(dice_loss_2))
    print('----------------------Dice Loss 3: ' + str(dice_loss_3))

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

        f2, ax2 = plt.subplots(4)
        ax2[0].plot(test_dice_losses_0)
        ax2[1].plot(test_dice_losses_1)
        ax2[2].plot(test_dice_losses_2)
        ax2[3].plot(test_dice_losses_3)
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
