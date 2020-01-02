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


# Training class
class Trainer:
    def __init__(self, criterion, optimizer, device):
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        
    def train_loop(self, model, train_loader,val_loader):
        for epoch in range(EPOCHS):
            print("---------------- Epoch {} ----------------".format(epoch+1))
            self._training_step(model, train_loader, epoch)
            
            self._validate(model, val_loader, epoch)
    
    def test(self, model, test_loader):
            print("---------------- Testing ----------------")
            self._validate(model, test_loader, 0, state="Testing")
            
    def _training_step(self, model, loader, epoch):
        model.train()
        
        for step, (X, y) in enumerate(loader):
            X, y = X.to(self.device), y.to(self.device)
            N = X.shape[0]
            
            self.optimizer.zero_grad()
            outs = model(X.float())
            loss = self.criterion(outs, y.long())
            
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
                
                outs = model(X.float())
                loss = self.criterion(outs, y.long())
                
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


# create dataset class
class Dataset(Dataset):
    def __init__(self, data_list, label_list, transform=None, train=True):       
        self.data_list = data_list
        self.label_list = label_list
        self.data_len = len(data_list)
        self.transform = transform
                
    def __len__(self):
        return self.data_len
    
    def __getitem__(self, index):
        return (self.data_list[index], self.label_list[index])


if __name__ == '__main__':
    MODEL_PATH = './model'

	# CUDA
    # testing CUDA is available or not
    CUDA = True
    if CUDA:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
	    device = torch.device("cpu")
    print(device)

    EPOCHS = 120
    BATCH_SIZE = 16
    PRINT_FREQ = 16
    #TRAIN_NUMS = 900
    TRAIN_NUMS = 1638

    input_data = np.load('./combine_data.npy')
    input_data = torch.from_numpy(input_data)
    input_label = np.load('./labels.npy')

    validate_data = np.load('./combine_data_validate.npy')
    validate_data = torch.from_numpy(validate_data)
    validate_label = np.load('./labels_validate.npy')

    test_data = np.load('./combine_data_test.npy')
    test_data = torch.from_numpy(test_data)
    test_label = np.load('./labels_test.npy')

    # make datasets and dataloaders
    train_dataset = Dataset(input_data, input_label, train=True)
    train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=False, sampler=SubsetRandomSampler(range(TRAIN_NUMS)))

    validate_dataset = Dataset(validate_data, validate_label, train=True)
    validate_loader = DataLoader(dataset=validate_dataset, batch_size=BATCH_SIZE, shuffle=False)

    test_dataset = Dataset(test_data, test_label, train=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    num_classes = 12
    model = nn.Sequential(
            nn.Linear(24, 256),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(256,256),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(256, num_classes),
        )
    model.cuda()
    summary(model, (1, 1, 24))

    # define loss, optimizer and scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(params=model.parameters(), lr=1e-3, momentum=0.9, weight_decay=5e-4) # weight_decay can be smaller
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    # start training
    trainer = Trainer(criterion, optimizer, device)
    start_time = time.time()
    trainer.train_loop(model, train_loader,validate_loader)
    trainer.test(model, test_loader)
    end_time = time.time()

    print("--- %s sec ---" % (end_time - start_time))
    # save model
    torch.save(model.state_dict(), MODEL_PATH + '/classfiler.pth')