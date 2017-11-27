# import packages here
import cv2
import numpy as np
import matplotlib.pyplot as plt
import glob
import random

import torch
import torchvision
import torchvision.transforms as transforms

from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

# ==========================================
#    Load Training Data and Testing Data
# ==========================================

class_names = [name[18:] for name in glob.glob('../HW3/data/train/*')]
class_names = dict(zip(xrange(len(class_names)), class_names))
print class_names

def img_norm(img):
    return np.float32(img) / 255 # normalize img pixels to [0, 1]

def load_dataset(path, img_size, num_per_class=-1, batch_num=1, shuffle=False, augment=False, is_color=False):

    data = []
    labels = []

    if is_color:
        channel_num = 3
    else:
        channel_num = 1

    # read images and resizing
    for id, class_name in class_names.iteritems():
        img_path_class = glob.glob(path + class_name + '/*.jpg')
        if num_per_class > 0:
            img_path_class = img_path_class[:num_per_class]
        labels.extend([id]*len(img_path_class))
        for filename in img_path_class:
            if is_color:
                img = cv2.imread(filename)
            else:
                img = cv2.imread(filename, 0)

            # resize the image
            img = cv2.resize(img, img_size, cv2.INTER_LINEAR)

            if is_color:
                img = np.transpose(img, [2, 0, 1])

            # norm pixel values to [0, 1]
            data.append(img_norm(img))

    # TODO: data augmentation
    # write your code below

    # randomly permute (this step is important for training)
    if shuffle:
        bundle = zip(data, labels)
        random.shuffle(bundle)
        data, labels = zip(*bundle)

    # divide data into minibatches of TorchTensors
    if batch_num > 1:
        batch_data = []
        batch_labels = []
        for i in xrange(len(data) / batch_num):
            minibatch_d = data[i*batch_num: (i+1)*batch_num]
            minibatch_d = np.reshape(minibatch_d, (batch_num, channel_num, img_size[0], img_size[1]))
            batch_data.append(torch.from_numpy(minibatch_d))

            minibatch_l = labels[i*batch_num: (i+1)*batch_num]
            batch_labels.append(torch.LongTensor(minibatch_l))
        data, labels = batch_data, batch_labels

    return zip(batch_data, batch_labels)

# load data into size (64, 64)

img_size = (64, 64)
batch_num = 50 # training sample number per batch

# load training dataset
trainloader_small = load_dataset('../HW3/data/train/', img_size, batch_num=batch_num, shuffle=True, augment=True)
train_num = len(trainloader_small)
print "Finish loading %d minibatches(=%d) of training samples." % (train_num, batch_num)

# load testing dataset
testloader_small = load_dataset('../HW3/data/test/', img_size, num_per_class=100, batch_num=batch_num)
test_num = len(testloader_small)
print "Finish loading %d minibatches(=%d) of testing samples." % (test_num, batch_num)

# show some images
def imshow(img):
    npimg = img.numpy()
    if len(npimg.shape) > 2:
        npimg = np.transpose(img, [1, 2, 0])
    plt.figure
    plt.imshow(npimg, 'gray')
    plt.show()
img, label = trainloader_small[0][0][11][0], trainloader_small[0][1][11]
print class_names[label]
imshow(img)






class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        self.conv1 = nn.Conv2d(1, 9, 9)
        self.conv2 = nn.Conv2d(6, 16, 5)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # If the size is a square you can only specify a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


net = Net()
print(net)
