# %%----------------------------------------IMPORT PACKAGES --------------------------------------------------
from __future__ import print_function, division

import os
from _csv import reader

from sklearn import preprocessing
import os
import torch
import pandas as pd

from PIL import Image
import numpy as np
from numpy import array
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import cohen_kappa_score, f1_score, accuracy_score

# Ignore warnings
import warnings

warnings.filterwarnings("ignore")

plt.ion()  # interactive mode

# %%---------------------------------------INTRODUCTION----------------------------------------------------
# 1. GOAL - Identify Interesting or Not Interesting images.  This is the first step in training the AI to identify and
#    classify marine life seen in the video.  The goal is 75 % accuracy with a xxx confidence level.
#
# 1. DATA SOURCE - This data is downloaded from NOAA Federal from OKEANOS Expedition on 9/26/2017, Expedition 708,
#    Dive 20, from tatorapp.com.
#    The images obtained were in 1 second increments (29,913 images), taken from  video obtained from the Dive
#    (101 mp4 files, each around 5 minutes in duration ~= 505 minutes ~= 8.42 hours (if each mp4 file was exactly
#    5 minutes, that would equate to 30,300 images).
#    Image details are:
#         Dimensions (W x H): 1920 x 1080
#         Bit depth: 24
#         Channels: 3
#
#    We randomly selected 1500 images to be in our dataset.

#    We then manually classified each image as INTERESTING or NOT INTERESTING.  The more narrow we are at this stage,
#    the better the results, so something was interesting if the image contained a fish, crustacean, starfish, or
#    sea urchin.  Due to time constraints, only 400 images were classified, reducing the dataset down to 400 from 1500.
#    Eventually all 29,913 images need to be classified to better train the model for the next step.
# %%----------------------------------------DATA IMPORT --------------------------------------------------
path_wd = os.getcwd()
data_dir = '/home/ubuntu/Deep-Learning/FinalProject/data_subset'  # os.listdir('../FinalProject/data_subset')

# read csv file as a list of lists
with open('labels.csv', 'r') as read_obj:
    # pass the file object to reader() to get the reader object
    csv_reader = reader(read_obj)  # was csv.reader
    # Pass reader object to list() to get a list of lists
    list_of_rows = list(csv_reader)

# %% --------------------------------------- Set-Up --------------------------------------------------------------------
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# use_gpu = torch.cuda.is_available()
SEED_T = torch.manual_seed(42)
SEED = np.random.seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# %% ----------------------------------- Hyper Parameters --------------------------------------------------------------
LR = 1e-1
N_EPOCHS = 30
BATCH_SIZE = 40
DROPOUT = 0.25

# %%----------------------------------------DATA PREPARATION----------------------------------------------

dtype = torch.float
# device = torch.device("cpu") # Uncomment this to run on CPU
device = torch.device("cuda:0")  # Uncomment this to run on GPU

# convert tuples to numpy arrays
arr_of_rows = np.array(list_of_rows)
# arr_of_rows = arr_of_rows.convert_objects(convert_numeric_=True)
print('Array Type: {}'.format(type(arr_of_rows)))  # type  numpy.ndarray
print('Array Shape: {}'.format(np.shape(arr_of_rows)))  # shape  (400, 2)
print(list_of_rows)  # [['56321613_EX1708_VID_20170927T010000Z_ROVHD.mp4_1678.png', 'interesting'],

image_names = arr_of_rows[:, 0]  # x
image_names_list = image_names.tolist()
interest_stat = arr_of_rows[:, 1]  # y
interest_stat = interest_stat.astype('str').reshape(-1)

le = preprocessing.LabelEncoder()  # label encoder  encodes target labels with values between 0 and n_classes-1, done to y target
le.fit(["interesting", "not_interesting"])  # should have 2 values; fit label encoder
interest_stat = le.transform(interest_stat)  # transforms labels to normalized encoding
print('image names shape , interest_stat.shape')
print(image_names.shape, interest_stat.shape)
#print(image_names)
#print(interest_stat)
# this code produces the matrix of the image file
# array(Image.open("/home/ubuntu/Deep-Learning/FinalProject/data_subset/56321613_EX1708_VID_20170927T010000Z_ROVHD.mp4_3326.png",mode='r'))
#
# img_PIL=Image.open(os.path.join(data_dir,image_names))
image_PIL = image_names_list

for img in image_PIL:
    Image.open(os.path.join(data_dir, img), mode='r')
    # print(img, image_PIL[img].shape, sample['interest_stat'].shape)


# %%-----------------------------helper function--------------------------------------
def show_interest_stat(interest_stat):
    """Show image with interest_stat"""
    #    plt.imshow(interest_stat)
    plt.hist(interest_stat, bins=6)


# plt.figure()
plt.show()

show_interest_stat(interest_stat)  # abt 325 interesting, 75 not interesting

# %%---------------------------------------------------------------------------------------------

Allimages = []
Counter = {}
for i in interest_stat:
    if str(i) in Counter:
        Counter[str(i)] = Counter[str(i)] + 1
    else:
        Counter[str(i)] = 1
Counts = sorted(Counter.items(), key=lambda kv: kv[1])
for i in Counts:
    print(i)


# ('0', 72)
# ('1', 328)
# print(Counts)
# %%-------------------------DEFINING THE ITERATOR---------------------------------------------------
class ImageDataset(Dataset):
    # Image dataset

    def __init__(self, dataset):
        # data
        self.transforms = transformations
        self.data_info = dataset
        self.image_arr = self.data_info[:, 0]
        self.label_arr = self.data_info[:, 1]
        # Calculate len
        self.data_len = len(self.label_arr)

    def __getitem__(self, index):
        # Get image name from the pandas df
        single_image_name = self.image_arr[index]
        # Open image  #TypeError: join() argument must be str or bytes, not 'ndarray'
        img_as_img = Image.open(os.path.join('/home/ubuntu/Deep-Learning/FinalProject/data_subset', np.array2string(single_image_name).strip("'")))
        # Transform image to tensor
        img_as_tensor = self.transforms(img_as_img)
        # Get label(class) of the image based on the cropped pandas column
        single_image_label = self.label_arr[index]
        #target = np.fromstring(single_image_label[1:-1], sep=' ')
        target = np.array(int(single_image_label))
        target = torch.from_numpy(target)
        return img_as_tensor, target

    def __len__(self):
        return self.data_len


# --------------------------------image augmentation---------------------

transformations = transforms.Compose([
    transforms.Resize((800, 600), interpolation=Image.BICUBIC),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ColorJitter(brightness=10, saturation=10, contrast=30, hue=0.1),
    # transforms.CenterCrop(224),
    transforms.ToTensor()])

# -------------------------SPLIT DATA-----------------------------
# print('Split Data')
# This script generates the training set
# and the held out set
x_train, x_test, y_train, y_test = train_test_split(image_PIL, interest_stat, random_state=SEED, test_size=0.2)  # took out stratify

# summarize dataset shape
# print('Train shape', x_train.shape, y_train.shape) can't use shape on list, would need to use np.array([])
# print('Test shape', x_test.shape, y_test.shape)

num_samples_train = len(x_train)
num_samples_test = len(x_test)
print('num_train= ', num_samples_train)
print('num_test= ', num_samples_test)

'''
target_total = np.load("target_total_120_160.npy")
target_category_coding = np.load("target_category_coding_120_160.npy")
print(x_total.shape, target_total.shape)
x_train, x_test, y_train, y_test = train_test_split(x_total, target_total, random_state=SEED, test_size=0.2,
                                                    stratify=target_category_coding)
print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
# Reshapes to (n_examples, n_channels, height_pixels, width_pixels)
x_train, x_test = x_train / 255, x_test / 255
All_frequency = y_train.sum(axis=0)
print(All_frequency)
#----------------------------------------------------------------
'''
# train dataset
x_train = np.array(x_train)
y_train = np.array(y_train)
x_test = np.array(x_test)
y_test = np.array(y_test)

train_dataset = np.vstack((x_train, y_train)).T
test_dataset = np.vstack((x_test, y_test)).T

# trainset = ImageDataset(x_train,y_train,# csv_file='/home/ubuntu/Deep-Learning/FinalProject/labels.csv',   data_dir='/home/ubuntu/Deep-Learning/FinalProject/data_subset')
trainset = ImageDataset(train_dataset)

train_loader = DataLoader(trainset,
                          batch_size=BATCH_SIZE,
                          shuffle=True,
                          num_workers=4)

# valset= ImageDataset(x_test,y_test, #csv_file='/home/ubuntu/Deep-Learning/FinalProject/labels.csv', data_dir='/home/ubuntu/Deep-Learning/FinalProject/data_subset')
valset = ImageDataset(test_dataset)

val_loader = DataLoader(valset,
                        batch_size=BATCH_SIZE,  # len of validation set  WAS 100
                        shuffle=True,
                        num_workers=4)

# --------------------------------------to help determine layer dimensions-------------------------
# >>> input, target = batch # input is a ?d tensor, target is ?d
# >>> bs=62, c=3, h=800, w=600 = input.size()
# %% -------------------------------------- CNN Class ------------------------------------------------------------------
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.conv1 = nn.Conv2d(40, 3, (800, 600))
        #self.pool1 = nn.MaxPool2d((2, 2))

#       self.conv2 = nn.Conv2d(7, 14, (3, 3))
#         self.convnorm2 = nn.BatchNorm2d(14)
#         self.pool2 = nn.MaxPool2d((2, 2))
#
#         self.conv3 = nn.Conv2d(14, 28, (3, 3))
#
#         self.conv4= nn.Conv2d(28, 56, (3, 3))
#         self.convnorm4 = nn.BatchNorm2d(56)
#         self.pool4 = nn.MaxPool2d((2, 2))
#
#         self.conv5= nn.Conv2d(56, 112, (3, 3))
#         self.convnorm5 = nn.BatchNorm2d(112)
#         self.pool5 = nn.MaxPool2d((2, 2))
#
#         self.conv6 = nn.Conv2d(112, 224, (2, 2))
#         self.convnorm6 = nn.BatchNorm2d(224)
#         self.pool6 = nn.MaxPool2d((2, 2))
#

        self.linear1 = nn.Linear((3*800*600), BATCH_SIZE)
        self.linear1_bn = nn.BatchNorm1d(BATCH_SIZE)
        self.drop = nn.Dropout(DROPOUT)
        self.linear2 = nn.Linear(BATCH_SIZE, 1)
        self.act = torch.relu

    def forward(self, x):
        #x = self.pool1(self.act(self.conv1(x)))
        x = self.act(self.conv1(x))
        # x = self.pool2(self.convnorm2(self.act(self.conv2(x))))
        # x = self.act(self.conv3(x))
        # x = self.pool4(self.convnorm4(self.act(self.conv4(x))))
        # x = self.pool5(self.convnorm5(self.act(self.conv5(x))))
        # x = self.pool6(self.convnorm6(self.act(self.conv6(x))))

        x = self.drop((self.linear1_bn(self.act(self.linear1(x.view(len(x), -1))))))
        return self.linear2(x)


# %% -------------------------------------- Training Prep ----------------------------------------------------------
model = CNN().to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=LR)
criterion = nn.BCEWithLogitsLoss()

# %% -------------------------------------- Training Loop ----------------------------------------------------------
print("Starting training loop...")
for epoch in range(N_EPOCHS):
    model.train()
    loss_train = 0
    for iter, traindata in enumerate(train_loader):
        train_inputs, train_labels = traindata
        train_inputs=train_inputs.permute(1, 0, 2, 3)
        train_inputs, train_labels = train_inputs.to(device), train_labels.to(device)
        optimizer.zero_grad()
        logits = model(train_inputs)
        loss = criterion(logits, train_labels)
        loss.backward()
        optimizer.step()
        loss_train += loss.item()
        torch.save(model.state_dict(), "cnnmodel.pt")
        print('Batch {} and Loss {:.5f}'.format(iter,loss_train/BATCH_SIZE))
    model.load_state_dict(torch.load("cnnmodel.pt"))
    model.eval()

    with torch.no_grad():
        for iter, valdata in enumerate(val_loader, 0):
            val_inputs, val_labels = valdata
            val_inputs1, val_labels1 = val_inputs.to(device), val_labels.to(device)
            y_test_pred = model(val_inputs1)
            tar_=val_labels.cpu().numpy()
            loss = criterion(y_test_pred, val_labels1)
            loss_test = loss.item()
            print('Validation Loss {:.5f}'.format(loss_test))
            validation_output=np.where(y_test_pred.cpu().numpy()>0.5,1,0)


    print("Epoch {} | Train Loss {:.5f}".format( epoch, loss_train/BATCH_SIZE))




