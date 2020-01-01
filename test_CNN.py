import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import SubsetRandomSampler

import torchvision.datasets as datasets
import torchvision.transforms as transforms

from torchviz import make_dot, make_dot_from_trace

from torchsummary import summary

from PIL import Image

import numpy as np
import time

from vgg import *


# Training class
class Trainer:
    def __init__(self, criterion, optimizer, device):
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        
    def train_loop(self, model, train_loader, val_loader):
        for epoch in range(EPOCHS):
            print("---------------- Epoch {} ----------------".format(epoch+1))
            self._training_step(model, train_loader, epoch)
            
            self._validate(model, val_loader, epoch)
    
    def test(self, model, test_loader):
            print("---------------- Testing ----------------")
            acc = self._validate(model, test_loader, 0, state="Testing")
            return acc
            
    def _training_step(self, model, loader, epoch):
        model.train()
        
        interface.state = 0
        interface.epoch_cur = epoch
        for step, (X, y) in enumerate(loader):
            X, y = X.to(self.device), y.to(self.device)
            N = X.shape[0]
            
            self.optimizer.zero_grad()
            outs = model(X)
            loss = self.criterion(outs, y)
            
            if step >= 0 and (step % PRINT_FREQ == 0):
                self._state_logging(outs, y, loss, step, epoch, "Training")
            
            loss.backward()
            self.optimizer.step()
        
        scheduler.step()
            
    def _validate(self, model, loader, epoch, state="Validate"):
        model.eval()
        outs_list = []
        loss_list = []
        y_list = []
        
        with torch.no_grad():
            for step, (X, y) in enumerate(loader):
                X, y = X.to(self.device), y.to(self.device)
                N = X.shape[0]
                
                outs = model(X)
                loss = self.criterion(outs, y)
                
                y_list.append(y)
                outs_list.append(outs)
                loss_list.append(loss)
            
            y = torch.cat(y_list)
            outs = torch.cat(outs_list)
            loss = torch.mean(torch.stack(loss_list), dim=0)
            acc = self._state_logging(outs, y, loss, step, epoch, state)
            return acc
                
                
    def _state_logging(self, outs, y, loss, step, epoch, state):
        acc = self._accuracy(outs, y)
        print("[{:3d}/{}] {} Step {:03d} Loss {:.3f} Acc {:.3f}".format(epoch+1, EPOCHS, state, step, loss, acc))
        return acc
            
    def _accuracy(self, output, target):
        batch_size = target.size(0)

        pred = output.argmax(1)
        correct = pred.eq(target)
        acc = correct.float().sum(0) / batch_size

        return acc


class Dataset(Dataset):
    def __init__(self, image_list, label_list, transform=None, train=True):
        
        # image_path
        self.image_path = ''
        if train == True:
            self.image_path = TRAIN_PATH
        else:
            self.image_path = TEST_PATH
        
        # Transforms
        # self.to_tensor = transforms.ToTensor()
        
        # image name        
        self.image_list = image_list
        
        # image labels
        self.label_list = label_list
        
        # Calculate len
        self.data_len = len(self.label_list)
        
        # transform
        self.transform = transform
                
    def __len__(self):
        return self.data_len
    
    def __getitem__(self, index):
        # Get image name from the pandas df
        single_image_name = self.image_list[index]
        
        # Open image and convert to RGB
        img_as_img = Image.open(self.image_path + single_image_name)
        img_as_img = img_as_img.convert('RGB')
        
        # Transform image to tensor
        img_as_tensor = self.transform(img_as_img)
        # img_as_tensor = self.to_tensor(img_as_tensor)
        
        # Get label(class) of the image based on the cropped pandas column
        single_image_label = self.label_list[index]

        return (img_as_tensor, single_image_label)


if __name__ == '__main__':

    labels = {
        'Abyssinian': 0,
        'Bengal': 1,
        'Birman': 2, 
        'Bombay': 3, 
        'British_Shorthair': 4,
        'Egyptian_Mau': 5,
        'Maine_Coon': 6, 
        'Persian': 7,
        'Ragdoll': 8,
        'Russian_Blue': 9,
        'Siamese': 10,
        'Sphynx': 11
    }

    MEAN = [0.49139968, 0.48215827, 0.44653124]
    STD = [0.2023, 0.1994, 0.2010]

    EPOCHS = 20
    BATCH_SIZE = 16
    PRINT_FREQ = 16
    #TRAIN_NUMS = 900   # for training and validation set
    TRAIN_NUMS = 1720   # for training and validation set
    TESTING_NUMS = 1

    CUDA = True

    PATH_TO_SAVE_DATA = "./"

    TRAIN_PATH = "./dataset/crop_image/train/"
    TEST_PATH = "./dataset/crop_image/test/"
    MODEL_PATH = "./model/model.pth"

    name_file = open(TEST_PATH + 'test.txt', 'r')
    test_image_list = name_file.readlines()
    name_file.close()

    label_file = open(TEST_PATH + 'label.txt', 'r')
    test_label_list = label_file.readlines()
    label_file.close()

    for i in range(len(test_image_list)):
        test_image_list[i] = test_image_list[i].strip()
        test_label_list[i] = int(test_label_list[i].strip())

    # define datatransform
    data_transform = transforms.Compose([
                      transforms.Resize(size=(224, 224)),
                      transforms.ToTensor(),
                      transforms.Normalize(MEAN, STD)
                      # data augmentaion
                    ])

    test_dataset = Dataset(test_image_list, test_label_list, data_transform, train=False)
    test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # CUDA
    # testing CUDA is available or not
    if CUDA:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")
    print(device)

    model = vgg('VGG16', pretrained=True, path = MODEL_PATH)
    model.cuda()
    #summary(model, (3, 224, 224))

    # define loss, optimizer and scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(params=model.parameters(), lr=1e-3, momentum=0.9, weight_decay=5e-4) # weight_decay can be smaller
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    # start testing
    trainer = Trainer(criterion, optimizer, device)
    vgg_accuracy = 0
    for i in range(TESTING_NUMS):
        acc = trainer.test(model, test_loader)
        vgg_accuracy += acc
    print('VGG16 Accuracy: {:.3f}'.format(vgg_accuracy/TESTING_NUMS))
