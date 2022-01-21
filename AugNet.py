import torch.nn as nn
import torch
from model import StudentNet, ResidualModule, MLP, StudentNet 

class AugNet(nn.Module):
    def __init__(self, stdnet : StudentNet):
        super().__init__()

        self.conv1 = stdnet.conv1
        self.bn = stdnet.bn
        self.relu = stdnet.relu

        self.residual_layer1_tb = stdnet.residual_layer1[0]
        self.residual_layer1 = self._make_resnet_block(3, 32, 32)
        self.residual_layer1_te = stdnet.residual_layer1[1]

        self.residual_layer2_tb = stdnet.residual_layer2[0]
        self.residual_layer2 = self._make_resnet_block(3, 64, 64)
        self.residual_layer2_te = stdnet.residual_layer2[1]


        self.residual_layer3_tb = stdnet.residual_layer3[0]
        self.residual_layer3 = self._make_resnet_block(3, 128, 128)
        self.residual_layer3_te = stdnet.residual_layer3[1]

        self.mlp = stdnet.mlp
        self.maxpool = nn.MaxPool2d(3, stride=2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn(x)
        fm0 = self.relu(x)
        
        x = self.maxpool(fm0)
        skip = self.residual_layer1_tb(x)
        x = self.residual_layer1(skip)
        fm1 = self.residual_layer1_te(x + skip)
        
        x = self.maxpool(fm1)
        skip = self.residual_layer2_tb(x)
        x = self.residual_layer2(skip)
        fm2 = self.residual_layer2_te(x + skip)
        
        x = self.maxpool(fm2)
        skip = self.residual_layer3_tb(x)
        x = self.residual_layer3(skip)
        fm3 = self.residual_layer3_te(x + skip)
        
        x = self.maxpool(fm3)
        x = x.view(x.shape[0], -1)
        x = self.mlp(x)
        return fm0, fm1, fm2, fm3, x

    def _make_resnet_block(self, n, input_dim, output_dim):
        layers = []
        for _ in range(n):
            layers.append(ResidualModule(input_dim, output_dim))
            input_dim = output_dim
        return nn.Sequential(*layers)

if __name__ == "__main__":
    stnet = StudentNet(3,10)
    augnet = AugNet(stnet)
    inp = torch.rand((1,3,32,32))
    out = augnet(inp)