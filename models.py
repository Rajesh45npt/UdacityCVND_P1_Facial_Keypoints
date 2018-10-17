## TODO: define the convolutional neural network architecture

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
        # self.conv1 = nn.Conv2d(1, 32, 5)
    
        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting
        
        # --------------------- Implementation Results -------------------------- #
        # I defined the input shape as (1, 224, 224)
        
        # [CONV]-[BN]-[MAXPOOL]
        # 1x224x224 to 32x112x112
        self.conv1 = nn.Conv2d(1, 32, 3, stride=1, padding=1)
        self.batch_norm1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)

        # [CONV]-[BN]-[MAXPOOL]
        # 32x112x112 to 64x56x56
        self.conv2 = nn.Conv2d(32, 64, 3, stride=1, padding=1)
        self.batch_norm2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)

        # [CONV]-[BN]-[MAXPOOL]
        # 64x56x56 to 128x28x28
        self.conv3 = nn.Conv2d(64, 128, 3, stride=1, padding=1)
        self.batch_norm3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2, 2)

        # [CONV]-[BN]-[MAXPOOL]
        # 128x28x28 to 256x14x14
        self.conv4 = nn.Conv2d(128, 256, 3, stride=1, padding=1)
        self.batch_norm4 = nn.BatchNorm2d(256)
        self.pool4 = nn.MaxPool2d(2, 2)
        
        # [CONV]-[BN]-[MAXPOOL]
        # 256x14x14 to 256x7x7
        self.conv5 = nn.Conv2d(256, 256, 3, stride=1, padding=1)
        self.batch_norm5 = nn.BatchNorm2d(256)
        self.pool5 = nn.MaxPool2d(2, 2)

        # FC layers
        self.fc1 = nn.Linear(7*7*256, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 136) #output

        # initialize weights
        torch.nn.init.kaiming_normal_(self.conv1.weight)
        torch.nn.init.kaiming_normal_(self.conv2.weight)
        torch.nn.init.kaiming_normal_(self.conv3.weight)
        torch.nn.init.kaiming_normal_(self.conv4.weight)
        torch.nn.init.kaiming_normal_(self.conv5.weight)
        torch.nn.init.kaiming_normal_(self.fc1.weight)
        torch.nn.init.kaiming_normal_(self.fc2.weight)
        torch.nn.init.kaiming_normal_(self.fc3.weight)

    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        ## x = self.pool(F.relu(self.conv1(x)))
        
        # Conv. layers
        x = self.pool1(F.relu(self.batch_norm1(self.conv1(x))))
        x = self.pool2(F.relu(self.batch_norm2(self.conv2(x))))
        x = self.pool3(F.relu(self.batch_norm3(self.conv3(x))))
        x = self.pool4(F.relu(self.batch_norm4(self.conv4(x))))
        x = self.pool5(F.relu(self.batch_norm5(self.conv5(x))))

        # Flatten
        x = x.view(x.size(0), -1)

        # FC layers
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x) # Activation function is identity function (regression problem)

        # a modified x, having gone through all the layers of your model, should be returned
        return x
