import pickle
from typing import List
from model import StudentNet, TeacherNet
from TrainAugnet import give_dataloader
import torch
import torch.nn as nn
import numpy as np
from torchvision import datasets
from torchvision.transforms import transforms
from torch.utils.data import random_split, DataLoader
import torch.optim as optimizers
from tqdm import tqdm  # to be removed
from sklearn.metrics import accuracy_score
from loss import FSP, KL_div
import yaml
import sys

def TrainStudent(Network : torch.nn.Module, tNets : List[torch.nn.Module], trainDataloader: DataLoader, valDataloader: DataLoader, device : str, epochs : int, optim : optimizers, Network_path : str, logs_path : str):
    score = 0.
    loss_fn = nn.CrossEntropyLoss()
    Network_hist = {'loss': [], 'accuracy': [], 'val_loss':[], 'val_accuracy': []}
    fsp_loss = FSP()

    # train teacher
    for epoch in range(epochs):
        train_acc = 0.
        train_loss = 0.
        val_acc = 0.
        val_loss = 0.
        Network.train()
        for (x,t) in trainDataloader:
            x,t = x.to(device), t.to(device)
            fm_s0, fm_s1, fm_s2, fm_s3, pred_s = Network(x)
            FSP_loss_list = []
            kl_div_loss_list = []
           
            for tNet in tNets:
                fm_t0, fm_t1, fm_t2, fm_t3, pred_t = tNet(x)
                FSP_loss = fsp_loss(fm_s0, fm_s1, fm_t0, fm_t1) + fsp_loss(fm_s1, fm_s2, fm_t1, fm_t2) + fsp_loss(fm_s2, fm_s3, fm_t2, fm_t3)
                FSP_loss_list.append(FSP_loss)
                kl_div_loss_list.append(KL_div(pred_t,pred_s))

            FSP_loss = torch.stack(FSP_loss_list,dim = 0)
            kl_div_loss = torch.stack(kl_div_loss_list,dim = 0)
            FSP_loss = FSP_loss.sum()
            kl_div_loss = kl_div_loss.sum()

            k = len(tNets)
            FSP_loss /= k
            kl_div_loss /= k
            
            loss = loss_fn(pred_t, t) + FSP_loss + kl_div_loss
            optim.zero_grad()
            loss.backward()
            optim.step()
            train_loss += loss.item()
            train_acc += accuracy_score(t.tolist(), pred_t.argmax(dim=-1).tolist())
        train_loss /= len(trainDataloader)
        train_acc /= len(trainDataloader)

        Network.eval()
        with torch.no_grad():
            for (x,t) in valDataloader:
                x,t = x.to(device), t.to(device)
                fm_t0, fm_t1, fm_t2, fm_t3, pred_t = Network(x)
                loss = loss_fn(pred_t, t)
                val_loss += loss.item()
                val_acc += accuracy_score(t.tolist(), pred_t.argmax(dim=-1).tolist())
            val_loss /= len(valDataloader)
            val_acc /= len(valDataloader)

        if score < val_acc:                                                                                         
            score = val_acc                                                                                         
            torch.save(Network.state_dict(), Network_path)                                            
            print('-'*10)   
            print('update')  
            print('-'*10)                                                                                                                                                 
        
        print(f'epoch: {epoch+1}, train_loss: {train_loss:.3f}, train_acc: {train_acc:.3f}, val_loss: {val_loss:.3f}, val_acc: {val_acc:.3f}')
        print('-'*10)

        Network_hist['loss'].append(train_loss)
        Network_hist['accuracy'].append(train_acc)
        Network_hist['val_loss'].append(val_loss)
        Network_hist['val_accuracy'].append(val_acc)

    with open(logs_path, mode='wb') as f:
        pickle.dump(Network_hist, f)

if __name__ == "__main__":
    t1 = TeacherNet(3,10)
    t2 = TeacherNet(3,10)
    t3 = TeacherNet(3,10)
    ts = [t1,t2,t3]
    std = StudentNet(3,10)
    train_dataloader, val_dataloader, test_dataloader = give_dataloader("./data/cifar10","CIFAR10",32,0.8,2)


    TrainStudent(std,ts,train_dataloader,val_dataloader,"cpu",1,optimizers.Adam(std.parameters()),"./logs/std.pt","./logs/std.pickle")

    

