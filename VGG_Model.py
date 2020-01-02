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

from IRE import *

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
            self._validate(model, test_loader, 0, state="Testing")
            interface.Save_Info()
            
    def _training_step(self, model, loader, epoch):
        model.train()
        
        interface.state = 0
        interface.epoch_cur = epoch
        for step, (X, y) in enumerate(loader):
            X, y = X.to(self.device), y.to(self.device)
            N = X.shape[0]
            if epoch == EPOCHS - 1:
                interface.RecordLabel(y)
            
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

        if state == 'Validate':
            interface.state = 1
        elif state == 'Testing':
            interface.state = 2
        
        with torch.no_grad():
            for step, (X, y) in enumerate(loader):
                X, y = X.to(self.device), y.to(self.device)
                N = X.shape[0]
                if (epoch == EPOCHS - 1) or (interface.state == 2):
                    interface.RecordLabel(y)
                
                outs = model(X)
                loss = self.criterion(outs, y)
                
                y_list.append(y)
                outs_list.append(outs)
                loss_list.append(loss)
            
            y = torch.cat(y_list)
            outs = torch.cat(outs_list)
            loss = torch.mean(torch.stack(loss_list), dim=0)
            self._state_logging(outs, y, loss, step, epoch, state)
                
                
    def _state_logging(self, outs, y, loss, step, epoch, state):
        acc = self._accuracy(outs, y)
        print("[{:3d}/{}] {} Step {:03d} Loss {:.3f} Acc {:.3f}".format(epoch+1, EPOCHS, state, step, loss, acc))
            
    def _accuracy(self, output, target):
        batch_size = target.size(0)

        pred = output.argmax(1)
        correct = pred.eq(target)
        acc = correct.float().sum(0) / batch_size

        return acc

# Datasets

# create dataset class
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


# VGG16 Model
class VGG(nn.Module):
    def __init__(self, num_classes=12, model_type='VGG16'):
        super(VGG, self).__init__()
        self.features = self._make_layers(cfg[model_type])
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )
    
    def forward(self, x):
        out = None
        out = self.features(x)
        interface.RecordFeature(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.classifier(out)
        interface.RecordClass(out)
        
        return out
    
    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        return nn.Sequential(*layers)


class IRE_Interface:
    def __init__(self):
        self.labels = []
        self.classes = []
        self.classes_validate = []
        self.classes_test = []
        self.epoch_cur = 0
        self.state = 0  # 0:training 1:validate 2:testing
        self.flag = False

    def RecordLabel(self,label_list):
        self.labels = label_list.cpu().clone().numpy().tolist().copy()
        # print('record label:' + str(len(self.labels)))
        self.flag = True
        return

    def RecordFeature(self,features):
        if ((self.epoch_cur != EPOCHS - 1) and (self.state < 2)) or (not self.flag):
            return
        #print('record feature:')
        #print(features)

        for i in range(len(self.labels)):
            IRE.Training(features[i],self.labels[i],self.state)

        return

    def RecordClass(self,classes):
        if ((self.epoch_cur != EPOCHS - 1) and (self.state < 2)) or (not self.flag):
            return
        #print('record class:')
        #print(classes)
        classes = classes.cpu().detach().clone().numpy()

        for single_class in classes:
            if self.state == 0:
                self.classes.append(single_class)
            elif self.state == 1:
                self.classes_validate.append(single_class)
            elif self.state == 2:
                self.classes_test.append(single_class)

        self.flag = False
        return

    def Save_Info(self):
        classes = np.array(self.classes)
        classes_validate = np.array(self.classes_validate)
        classes_test = np.array(self.classes_test)
        features,features_validate,features_test,labels,labels_validate,labels_test = IRE.Get_Info()

        np.save(PATH_TO_SAVE_DATA + 'classes',classes)
        np.save(PATH_TO_SAVE_DATA + 'classes_validate',classes_validate)
        np.save(PATH_TO_SAVE_DATA + 'classes_test',classes_test)
        np.save(PATH_TO_SAVE_DATA + 'features',features)
        np.save(PATH_TO_SAVE_DATA + 'features_validate',features_validate)
        np.save(PATH_TO_SAVE_DATA + 'features_test',features_test)
        np.save(PATH_TO_SAVE_DATA + 'labels',labels)
        np.save(PATH_TO_SAVE_DATA + 'labels_validate',labels_validate)
        np.save(PATH_TO_SAVE_DATA + 'labels_test',labels_test)


if __name__ == '__main__':

    # Hyperparameters
    cfg = {
        'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
        'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
        'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
        'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
    }

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
    TRAIN_NUMS = 1638   # for training and validation set

    CUDA = True

    PATH_TO_SAVE_DATA = "./"

    TRAIN_PATH = "./dataset/crop_image/train/"
    TEST_PATH = "./dataset/crop_image/test/"
    MODEL_PATH = "./model/model.pth"

    IRE = IRE()
    interface = IRE_Interface()

    # Get image list and label
    # Image name
    name_file = open(TRAIN_PATH + 'train.txt', 'r')
    train_image_list = name_file.readlines()
    name_file.close()

    name_file = open(TEST_PATH + 'test.txt', 'r')
    test_image_list = name_file.readlines()
    name_file.close()

    # image labels
    label_file = open(TRAIN_PATH + 'label.txt', 'r')
    train_label_list = label_file.readlines()
    label_file.close()

    label_file = open(TEST_PATH + 'label.txt', 'r')
    test_label_list = label_file.readlines()
    label_file.close()

    # strip and turn labels to int
    for i in range(len(train_image_list)):
        train_image_list[i] = train_image_list[i].strip()
        train_label_list[i] = int(train_label_list[i].strip())

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

    # make datasets and dataloaders
    train_dataset = Dataset(train_image_list, train_label_list, data_transform, train=True)
    test_dataset = Dataset(test_image_list, test_label_list, data_transform, train=False)
    train_len = len(train_dataset)

    train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=False, sampler=SubsetRandomSampler(range(TRAIN_NUMS)))
    val_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=False, sampler=SubsetRandomSampler(range(TRAIN_NUMS, train_len)))
    test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # CUDA
    # testing CUDA is available or not
    if CUDA:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")
    print(device)

    # VGG Model
    # model = VGG.vgg('VGG16', pretrained=False)
    model = VGG(num_classes=12, model_type='VGG16')
    model.cuda()
    summary(model, (3, 224, 224))

    # define loss, optimizer and scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(params=model.parameters(), lr=1e-3, momentum=0.9, weight_decay=5e-4) # weight_decay can be smaller
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    # start training
    trainer = Trainer(criterion, optimizer, device)
    start_time = time.time()
    trainer.train_loop(model, train_loader, val_loader)
    trainer.test(model, test_loader)
    end_time = time.time()

    print("--- %s sec ---" % (end_time - start_time))
    # save model
    torch.save(model.state_dict(), MODEL_PATH)