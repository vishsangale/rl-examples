from torch import nn
import torch.nn.functional as F


class DQN(nn.Module):
    def __init__(self, h, w, outputs):
        super(DQN, self).__init__()
        nr_channels = 3

        self.conv1 = nn.Conv2d(nr_channels, 16, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(16)

        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(32)

        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(32)

        def conv_size(size, kernel_size=5, stride=2):
            return (size - kernel_size) // stride + 1
        convh = conv_size(conv_size(conv_size(h)))
        convw = conv_size(conv_size(conv_size(w)))
        self.out = nn.Linear(convh*convw*32, outputs)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        return self.out(x.view(x.size(0), -1))
