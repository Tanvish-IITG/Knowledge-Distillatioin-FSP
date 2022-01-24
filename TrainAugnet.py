import pickle
from typing import List
from model import StudentNet, TeacherNet
from AugNet import AugNet
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

# helps in debugging backward pass
torch.autograd.set_detect_anomaly(True)

def give_dataloader(data_dir,dataset_name,img_size,train_ratio,batch_size):

    train_transform = transforms.Compose([transforms.RandomCrop(img_size, padding=4),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor()]
                                    )
    
    test_transform = transforms.Compose([
                                    transforms.ToTensor()]
                                    )

    datatest_train = eval("datasets." + dataset_name)(root=data_dir,
                                    download=True,
                                    train=True,
                                    transform=train_transform)
    datatest_test = eval("datasets." + dataset_name)(root=data_dir,
                                    download=True,
                                    train=False,
                                    transform=test_transform)

    n_samples = len(datatest_train)
    n_train = int(n_samples * train_ratio)
    n_val = n_samples - n_train
    datatest_train, datatest_val = random_split(datatest_train, [n_train, n_val])

    train_dataloader = DataLoader(datatest_train,
                                batch_size=batch_size,
                                shuffle=True,
                                drop_last=True,
                                num_workers=8)
    val_dataloader = DataLoader(datatest_val,
                                batch_size=batch_size,
                                shuffle=False)
    test_dataloader = DataLoader(datatest_test,
                                batch_size=batch_size,
                                shuffle=False)

    return train_dataloader, val_dataloader, test_dataloader

def TrainTeacher(Network : torch.nn.Module, trainDataloader: DataLoader, valDataloader: DataLoader, device : str, epochs : int, optim : optimizers, Network_path : str, logs_path : str):
    score = 0.
    loss_fn = nn.CrossEntropyLoss()
    Network_hist = {'loss': [], 'accuracy': [], 'val_loss':[], 'val_accuracy': []}

    # train teacher
    for epoch in range(epochs):
        train_acc = 0.
        train_loss = 0.
        val_acc = 0.
        val_loss = 0.
        Network.train()
        for (x,t) in trainDataloader:
            x,t = x.to(device), t.to(device)
            fm_s0, fm_s1, fm_s2, fm_s3, pred_t = Network(x)
            loss = loss_fn(pred_t, t)
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
            FSP_loss = torch.tensor(0,dtype = torch.float32, requires_grad = True)
            kl_div_loss = torch.tensor(0,dtype = torch.float32, requires_grad = True)

            for tNet in tNets:
                fm_t0, fm_t1, fm_t2, fm_t3, pred_t = tNet(x)
                FSP_loss += fsp_loss(fm_s0, fm_s1, fm_t0, fm_t1)
                FSP_loss += fsp_loss(fm_s1, fm_s2, fm_t1, fm_t2)
                FSP_loss += fsp_loss(fm_s2, fm_s3, fm_t2, fm_t3)
                kl_div_loss += KL_div(pred_t,pred_s)

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




def main():
    # settings
    torch.manual_seed(123)
    np.random.seed(123)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    with open(sys.argv[1], "r") as yamlfile:
        config = yaml.load(yamlfile, Loader=yaml.FullLoader)
    print("Read successful")

    input_dim = config['network']['input_dim']
    output_dim = config['network']['output_dim']

    data_path = config['dataset']['path']
    dataset_name = config['dataset']['name']
    img_size = config['dataset']['img_size']
    ratio = config['dataset']['ratio']
    batch_size = config['dataset']['batch_size']




    train_dataloader, val_dataloader, test_dataloader = give_dataloader(data_path,dataset_name,img_size,ratio,batch_size)
    
    # model
    teachers = [TeacherNet(input_dim=input_dim, output_dim=output_dim).to(device),TeacherNet(input_dim=input_dim, output_dim=output_dim).to(device),TeacherNet(input_dim=input_dim, output_dim=input_dim).to(device)]
    student = StudentNet(input_dim=input_dim, output_dim=output_dim).to(device)


    optim_type = ["Adam","SGD","Adagrad"]
    optim_type = config["teachers_train"]['optim']
    network_path = config["teachers_train"]['network_path']
    log_path = config["teachers_train"]['log_path']
    epochs = [12,12,12]
    epochs = config["teachers_train"]['epochs']

    for i in range(3):
        optim = eval("optimizers." + optim_type[i])(teachers[i].parameters())
        TrainTeacher(teachers[i],train_dataloader,val_dataloader,device,epochs[i],optim,network_path + str(i+1) +".pt",log_path + str(i+1) + ".pickle")

    network_path = config["student_train"]['network_path']
    log_path = config["student_train"]['log_path']
    epoch = 12
    epoch = config["student_train"]['epoch']
    optim = config["student_train"]['optim']

    TrainStudent(student,teachers,train_dataloader,val_dataloader,device,epoch,eval("optimizers." + optim)(student.parameters()),network_path + ".pt", log_path + ".pickle")

    augnet = AugNet(student)

    network_path = config["augnet_train"]['network_path']
    log_path = config["augnet_train"]['log_path']
    epoch = 12
    epoch = config["augnet_train"]['epoch']
    optim = config["augnet_train"]['optim']
    TrainTeacher(augnet,train_dataloader,val_dataloader,device,epoch,eval("optimizers." + optim)(augnet.parameters()), network_path + ".pth", log_path + ".pickle")

if __name__ == '__main__':
    main()