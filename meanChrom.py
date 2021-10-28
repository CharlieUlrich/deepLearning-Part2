# LOAD THE DATASETS
import cv2 
import os
import glob
import torch
import random
import numpy as np
torch.set_default_tensor_type(torch.FloatTensor)

img_dir = "./face_images/*.jpg"
files = glob.glob(img_dir)
data = []

for fl in files:
    img = cv2.imread(fl)
    data.append(img) #img.T to get in project spec format

# Load data into tensor of size nImages x Channels x Height x Width
    # nImages = number of images in the folder
    # Channels = 3 (RBG colors)
    # Height, Width = 128

# Here's an issue. OpenCV expects images to be in format nImages x Height x Width x Channels  augimg[0][x][y][0]
    # Does it affect anything having the dimensions being out of order? 
    # We at least need to use the OpenCV ordering for preprocessing.
imgTens= torch.tensor(data)

# Randomly shuffle the data using torch.randperm
index = torch.randperm(imgTens.shape[0])
imgTens = imgTens[index].view(imgTens.size())

# AUGMENT YOUR DATA
# Augment by a small factor such as 10 to reduce overfitting by using OpenCV to transform your original images
# there must be a better way of doing this than what I have going on. This is just ugly.
augImg = torch.cat((imgTens, imgTens, imgTens, imgTens, imgTens, imgTens, imgTens, imgTens, imgTens, imgTens), 0)

for i in range(imgTens.shape[0], augImg.shape[0]):
    currImg = augImg[i].numpy()

    # horizontal flips - 50% chance
    if random.random() < 0.5:
        currImg = cv2.flip(currImg, 0)

    # random crops - crop size ranges from 64 to 128
    cropSize = np.random.randint(64, 128)

    newX = np.random.randint(0, cropSize)
    newY = np.random.randint(0, cropSize)

    cropped = currImg[newY: newY + cropSize, newX: newX + cropSize]
    currImg = cv2.resize(cropped, (128, 128))

    # scalings of input RBG values by single scaler randomly chosen between [0.6, 1.0]
    randScalar = random.uniform(0.6, 1)
    for i in range(3):
        currImg[:, :, i] = currImg[:, :, i] * randScalar

    augImg[i] = torch.from_numpy(currImg)

# CONVERT YOUR IMAGES TO L * a * b * COLOR SPACE
for i in range(augImg.shape[0]):
    augImg[i] = torch.from_numpy(cv2.cvtColor(augImg[i].numpy(), cv2.COLOR_BGR2LAB))
# BUILD A SIMPLE REGRESSOR
    # Using convolutional layers, that predict the mean chrominance values for the entire input image
    # Input: grayscale image (only the L* channel)
    # Output: predicts mean chrominance (take the mean across all pixels to obtain mean a* and mean b*) values across all pixels of the image, ignoring pixel location

# ONCE YOU HAVE THIS WORKING, MAKE A COPY OF THIS CODE SO THAT YOU CAN SUBMIT IT LATER.

# import np.linalg from numpy
#  for i in augImg:
#      for x in i:
#          for y in x:
#              y[0] = float(y[0]/100);

normalGreyImg = torch.zeros(7500, 128, 128)
meanChromTest = torch.zeros(7500, 2)
for i in range(7500):
    LChan, AChan, BChan = cv2.split(augImg[i].numpy())

    meanChromTest[i, 0] = np.mean(AChan)
    meanChromTest[i, 1] = np.mean(BChan)

    LChan = (LChan - np.min(LChan)) / (np.max(LChan) - np.min(LChan))
    normalGreyImg[i, :, :] = torch.from_numpy(LChan)


import torch.nn as nn 
import torch.nn.functional as F
import torchvision
import torchvision.transforms as tf

class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=3, kernel_size=65)
        self.conv2 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=33)
        self.conv3 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=17)
        self.conv4 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=9)
        self.conv5 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=5)
        self.conv6 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3)
        self.conv7 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=1)


        self.out = nn.Linear(in_features=2, out_features=2)

    
    def forward(self, t):

        #hidden conv layer
        t = self.conv1(t)
        t = F.relu(t)
        #t = F.max_pool2d(t, kernel_size=3, stride=1)

        #hidden conv layer
        t = self.conv2(t)
        t = F.relu(t)
        #t = F.max_pool2d(t, kernel_size=3, stride=1)

        #hidden conv layer
        t = self.conv3(t)
        t = F.relu(t)
        #t = F.max_pool2d(t, kernel_size=3, stride=1)

        #hidden conv layer
        t = self.conv4(t)
        t = F.relu(t)
        #t = F.max_pool2d(t, kernel_size=3, stride=1)

        #hidden conv layer
        t = self.conv5(t)
        t = F.relu(t)
        #t = F.max_pool2d(t, kernel_size=3, stride=1)

        #hidden conv layer
        t = self.conv6(t)
        t = F.relu(t)
        #t = F.max_pool2d(t, kernel_size=3, stride=1)

        #hidden conv layer
        t = self.conv7(t)
        t = F.relu(t)
        #t = F.max_pool2d(t, kernel_size=3, stride=1)

        #output 
        t.reshape(-1,2)
        print(t.shape)
        t = self.out(t)

        return t

a = Network()
for name, param in a.named_parameters():
        print(name, '\t\t', param.shape)
b = normalGreyImg.unsqueeze(1)
print(b.shape)
print(b.type())
pred = a(b)
print(pred)
print(pred.shape)
# for z in augImg:0
#     for i in range(1,7):
#         loopedOut = module1(z[:,:,0])
#         loopedOut = module2(loopedOut)
        
#         kernel_size[0] = (kernel_size[0]+1)/2
#         kernel_size[1] = (kernel_size[1]+1)/2
#         padding = (kernel_size-1)/2

#         module1 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=kernel_size, padding=padding)
#         module2 = nn.ReLU()
