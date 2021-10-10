import pickle
from model import StudentNet, TeacherNet
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from torchvision import datasets
from torchvision.transforms import transforms
from torch.utils.data import random_split, DataLoader
import torch.optim as optimizers
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from loss import FSP
torch.autograd.set_detect_anomaly(True)
def main():
    # settings
    torch.manual_seed(123)
    np.random.seed(123)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # dataseet(cifar10)
    data_dir = './data/cifar10'
    train_transform = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean = [0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])])
    
    test_transform = transforms.Compose([
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean = [0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])])

    cifar10_train = datasets.CIFAR10(root=data_dir,
                                    download=True,
                                    train=True,
                                    transform=train_transform)
    cifar10_test = datasets.CIFAR10(root=data_dir,
                                    download=True,
                                    train=False,
                                    transform=test_transform)

    n_samples = len(cifar10_train)
    n_train = int(n_samples * 0.8)
    n_val = n_samples - n_train
    cifar10_train, cifar10_val = random_split(cifar10_train, [n_train, n_val])

    train_dataloader = DataLoader(cifar10_train,
                                batch_size=256,
                                shuffle=True,
                                drop_last=True,
                                num_workers=8)
    val_dataloader = DataLoader(cifar10_val,
                                batch_size=256,
                                shuffle=False)
    test_dataloader = DataLoader(cifar10_test,
                                batch_size=256,
                                shuffle=False)
    
    # model
    teacher = TeacherNet(input_dim=3, output_dim=10).to(device)
    student = StudentNet(input_dim=3, output_dim=10).to(device)

    # paramter etc
    epochs = 12
    min_loss = float('inf')
    score = 0.
    fsp_loss = FSP()
    loss_fn = nn.CrossEntropyLoss()
    teacher_optim = optimizers.Adam(teacher.parameters())
    student_optim = optimizers.Adam(student.parameters())
    teacher_hist = {'loss': [], 'accuracy': [], 'val_loss':[], 'val_accuracy': []}
    student_hist = {'loss': [], 'accuracy': [], 'val_loss':[], 'val_accuracy': []}
    fsp_hist = {'loss': [], 'val_loss':[]}

    # train teacher
    for epoch in range(epochs):
        train_acc = 0.
        train_loss = 0.
        val_acc = 0.
        val_loss = 0.
        teacher.train()
        for (x,t) in tqdm(train_dataloader, leave=False):
            x,t = x.to(device), t.to(device)
            fm_s0, fm_s1, fm_s2, fm_s3, pred_t = teacher(x)
            loss = loss_fn(pred_t, t)
            teacher_optim.zero_grad()
            loss.backward()
            teacher_optim.step()
            train_loss += loss.item()
            train_acc += accuracy_score(t.tolist(), pred_t.argmax(dim=-1).tolist())
        train_loss /= len(train_dataloader)
        train_acc /= len(train_dataloader)

        teacher.eval()
        with torch.no_grad():
            for (x,t) in val_dataloader:
                x,t = x.to(device), t.to(device)
                fm_t0, fm_t1, fm_t2, fm_t3, pred_t = teacher(x)
                loss = loss_fn(pred_t, t)
                val_loss += loss.item()
                val_acc += accuracy_score(t.tolist(), pred_t.argmax(dim=-1).tolist())
            val_loss /= len(val_dataloader)
            val_acc /= len(val_dataloader)

        if score < val_acc:                                                                                         
            score = val_acc                                                                                         
            torch.save(teacher.state_dict(), './logs/teacher_param.pth')                                            
            print('-'*10)   
            print('update')  
            print('-'*10)                                                                                                                                                 
        
        print(f'epoch: {epoch+1}, train_loss: {train_loss:.3f}, train_acc: {train_acc:.3f}, val_loss: {val_loss:.3f}, val_acc: {val_acc:.3f}')
        print('-'*10)

        teacher_hist['loss'].append(train_loss)
        teacher_hist['accuracy'].append(train_acc)
        teacher_hist['val_loss'].append(val_loss)
        teacher_hist['val_accuracy'].append(val_acc)

    with open('./logs/teacher_hist.pickle', mode='wb') as f:
        pickle.dump(teacher_hist, f)

    # First FSP training
    fsp_loss = FSP()
    for epoch in range(epochs):
        train_loss = 0.
        val_loss = 0.
        student.train()
        teacher.eval()
        for (x, t) in tqdm(train_dataloader, leave=False):
            x, t = x.to(device), t.to(device)
            # extract pre-trained teacher's feature map
            with torch.no_grad():
                fm_t0, fm_t1, fm_t2, fm_t3, pred_t = teacher(x)
            # extract student's feature map
            fm_s0, fm_s1, fm_s2, fm_s3, pred_s = student(x)
            # calculate FSP loss
            loss = fsp_loss(fm_s0, fm_s1, fm_t0, fm_t1) + fsp_loss(fm_s1, fm_s2, fm_t1, fm_t2) + fsp_loss(fm_s2, fm_s3, fm_t2, fm_t3)
            student_optim.zero_grad()
            loss.backward(retain_graph=True)
            student_optim.step()
            train_loss += loss.item()
        train_loss /= len(train_dataloader)

        # validation step
        student.eval()
        with torch.no_grad():
            for (x, t) in tqdm(val_dataloader, leave=False):
                x, t = x.to(device), t.to(device)
                with torch.no_grad():
                    fm_t0, fm_t1, fm_t2, fm_t3, pred_t = teacher(x)
                    # extract student's feature map
                fm_s0, fm_s1, fm_s2, fm_s3, pred_s = student(x)
                loss = fsp_loss(fm_s0, fm_s1, fm_t0.detach(), fm_t1.detach()) + fsp_loss(fm_s1, fm_s2, fm_t1.detach(), fm_t2.detach()) + fsp_loss(fm_s2, fm_s3, fm_t2.detach(), fm_t3.detach())
                val_loss += loss.item()
            val_loss /= len(train_dataloader)
        
        if min_loss > val_loss:
            print('-' * 10)
            print('min_loss is updated')
            min_loss = val_loss
            torch.save(student.state_dict(), './logs/student_fsp_param.pth')
        
        fsp_hist['loss'].append(train_loss)
        fsp_hist['val_loss'].append(val_loss)
                
        with open('./logs/fsp_hist.pickle', mode='wb') as f:
            pickle.dump(fsp_hist, f)

        print(f'epoch: {epoch+1}, fsp_loss: {loss:.3f}, fsp_val_loss: {val_loss:.3f}')
        print('-'*10)
    
    # load best paramter
    student.load_state_dict(torch.load('./logs/student_fsp_param.pth'))
    
    min_loss = float('inf')
    # train for the original task
    for epoch in range(epoch):
        train_loss = 0.
        train_acc = 0.
        val_loss = 0.
        val_acc = 0.
        student.train()
        for (x,t) in tqdm(train_dataloader, leave=False):
            x, t = x.to(device), t.to(device)
            fm_s0, fm_s1, fm_s2, fm_s3, pred_s = student(x)
            loss = loss_fn(pred_s, t)
            student_optim.zero_grad()
            loss.backward(retain_graph=True)
            student_optim.step()
            train_loss += loss.item()
            train_acc += accuracy_score(t.tolist(), pred_s.argmax(dim=-1).tolist())
        train_loss /= len(train_dataloader)
        train_acc /= len(train_dataloader)

        # validation step
        student.eval()
        with torch.no_grad():
            for (x, t) in tqdm(val_dataloader, leave=False):
                x, t = x.to(device), t.to(device)
                fm_s0, fm_s1, fm_s2, fm_s3, pred_s = student(x)
                loss = loss_fn(pred_s, t)
                val_loss += loss.item()
                val_acc += accuracy_score(t.tolist(), pred_s.argmax(dim=-1).tolist())
            val_loss /= len(val_dataloader)
            val_acc /= len(val_dataloader)

        if min_loss > val_loss:
            print('-' * 10)
            print('min_loss is updated')
            min_loss = val_loss
            torch.save(student.state_dict(), './logs/student_param.pth')

        # store accuracy and loss
        student_hist['loss'].append(train_loss)
        student_hist['accuracy'].append(train_acc)
        student_hist['val_loss'].append(val_loss)
        student_hist['val_accuracy'].append(val_acc)

        print(f'epoch: {epoch+1}, train_loss: {train_loss:.3f}, train_acc: {train_acc:.3f}, val_loss: {val_loss:.3f}, val_acc: {val_acc:.3f}')
        print('-'*10)

    # store value of hist in logs
    with open('./logs/student_hist.pickle', mode='wb') as f:
        pickle.dump(student_hist, f)

    # test student performance
    student.load_state_dict(torch.load('./logs/student_param.pth'))
    student.eval()
    test_loss = 0.
    test_acc = 0.
    with torch.no_grad():
        for (x, t) in tqdm(test_dataloader, leave=False):
            x, t = x.to(device), t.to(device)
            fm_s0, fm_s1, fm_s2, fm_s3, pred_s = student(x)
            loss = loss_fn(pred_s, t)
            test_loss += loss.item()
            test_acc += accuracy_score(t.tolist(), pred_s.argmax(dim=-1).tolist())
        test_loss /= len(test_dataloader)
        test_acc /= len(test_dataloader)

    print(f'test_loss: {test_loss:.3f}, test_accuracy: {test_acc:.3f}')

if __name__ == '__main__':
    main()