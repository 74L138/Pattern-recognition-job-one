import torch.nn as nn
# import test

class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        # torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        # torch.nn.MaxPool2d(kernel_size, stride, padding)
        # input 維度 [1, 16, 16]
        self.vgg = nn.Sequential(
            nn.Conv2d(1, 16, 3, 1, 1),  # [16, 28, 28]
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, 1, 1),  # [32, 28, 28]
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),  # [32, 14, 14]
            nn.Conv2d(32, 64, 3, 1, 1),  # [64, 14, 14]
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, 1, 1),  # [128, 14, 14]
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),  # [128, 7, 7]
        )
        #定义模型
        self.fc = nn.Sequential(
            nn.Linear(128*7*7, 512),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(1024, 11),
        )
#前向传播
    def forward(self, x):
        out = self.vgg(x)
        # print(out.shape)
        out = out.view(out.size()[0], -1)
        # print(out.shape)
        return self.fc(out)
