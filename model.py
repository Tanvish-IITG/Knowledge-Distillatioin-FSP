import torch
import torch.nn as nn
from torchsummary import summary

class ResidualModule(nn.Module):
    def __init__(self, input_dim, output_dim, stride=1):
        super().__init__()

        self.relu = nn.ReLU(inplace=False)

        self.residual_layer = nn.Sequential(
            nn.Conv2d(input_dim, output_dim, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(output_dim),
            self.relu,
            nn.Conv2d(output_dim, output_dim, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(output_dim)
        )

        self.shortcut = nn.Sequential()

        if stride != 1 or input_dim != output_dim:
            self.shortcut = nn.Sequential(
                nn.Conv2d(input_dim, output_dim, kernel_size=3, padding=1, )
            )

    def forward(self, x):
        return self.relu(self.residual_layer(x) + self.shortcut(x))

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU(inplace=False)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# 32 layers
class TeacherNet(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()

        self.conv1 = nn.Conv2d(input_dim, 16, kernel_size=3, stride=1, padding=1)
        self.bn = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=False)
        self.residual_layer1 = self._make_teacher_layer(5, 16, 32)
        self.residual_layer2 = self._make_teacher_layer(5, 32, 64)
        self.residual_layer3 = self._make_teacher_layer(5, 64, 128)
        self.mlp = MLP(128, 64, output_dim)
        self.maxpool = nn.MaxPool2d(3, stride=2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn(x)
        fm0 = self.relu(x)
        
        x = self.maxpool(fm0)
        fm1 = self.residual_layer1(x)
        
        x = self.maxpool(fm1)
        fm2 = self.residual_layer2(x)
        
        x = self.maxpool(fm2)
        fm3 = self.residual_layer3(x)
        
        x = self.maxpool(fm3)
        x = x.view(x.shape[0], -1)
        x = self.mlp(x)
        return fm0, fm1, fm2, fm3, x

    

    def _make_teacher_layer(self, n, input_dim, output_dim):
        layers = []
        for _ in range(n):
            layers.append(ResidualModule(input_dim, output_dim))
            input_dim = output_dim
        return nn.Sequential(*layers)


# 14 layers 
class StudentNet(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()

        self.conv1 = nn.Conv2d(input_dim, 16, kernel_size=3, stride=1, padding=1)
        self.bn = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=False)
        self.residual_layer1 = self._make_student_layer(2, 16, 32)
        self.residual_layer2 = self._make_student_layer(2, 32, 64)
        self.residual_layer3 = self._make_student_layer(2, 64, 128)
        self.mlp = MLP(128, 64, output_dim)
        self.maxpool = nn.MaxPool2d(3, stride=2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn(x)
        fm0 = self.relu(x)
        
        x = self.maxpool(fm0)
        fm1 = self.residual_layer1(x)
        
        x = self.maxpool(fm1)
        fm2 = self.residual_layer2(x)
        
        x = self.maxpool(fm2)
        fm3 = self.residual_layer3(x)
        
        x = self.maxpool(fm3)
        x = x.view(x.shape[0], -1)
        x = self.mlp(x)
        return fm0, fm1, fm2, fm3, x

    def _make_student_layer(self, n, input_dim, output_dim):
        layers = []
        for _ in range(n):
            layers.append(ResidualModule(input_dim, output_dim))
            input_dim = output_dim
        return nn.Sequential(*layers)


if __name__ == '__main__':
    teacher = TeacherNet(input_dim=3, output_dim=10)
    student = StudentNet(input_dim=3, output_dim=10)
    x = torch.randn((256, 3, 32, 32))
    output = student(x)
    print(output.shape)
    #summary(teacher, (3,32,32))
    #summary(student, (3,32,32))