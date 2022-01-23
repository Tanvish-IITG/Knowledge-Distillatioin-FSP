import torch
from torch.functional import Tensor
import torch.nn as nn
import torch.nn.functional as F

class FSP(nn.Module):
    def __init__(self):
        super(FSP, self).__init__()

    def calculate_fsp_matrix(self, fm1, fm2):
        if fm1.size(2) > fm2.size(2):
            fm1 = F.adaptive_avg_pool2d(fm1, (fm2.size(2), fm2.size(3)))
        # calculate FSP
        fm1 = fm1.view(fm1.size(0), fm1.size(1), -1) # [256, 16, 32, 32] -> [256, 16, 904]
        fm2 = fm2.view(fm2.size(0), fm2.size(1), -1).transpose(1,2) # [256, 32, 15, 15] -> [256, 32, 125]
        fsp = torch.bmm(fm1, fm2) / fm1.size(2)
        return fsp

    def forward(self, fm_s1, fm_s2, fm_t1, fm_t2):
        loss = F.mse_loss(self.calculate_fsp_matrix(fm_s1, fm_s2), self.calculate_fsp_matrix(fm_t1, fm_t2))
        return loss

def KL_div(teacher: Tensor, student : Tensor):
    teacher = F.softmax(teacher)
    student = F.softmax(student)
    return torch.mean(teacher * ( torch.log(teacher) - torch.log(student)) )

if __name__ == '__main__':
    input1 = torch.randn((2,64, 7, 7))
    input2 = torch.randn((2,64, 7, 7))
    input3 = torch.randn((2,64, 7, 7))
    input4 = torch.randn((2,64, 7, 7))
    loss = FSP()
    value = loss(input1, input2, input3, input4)
    print(value)