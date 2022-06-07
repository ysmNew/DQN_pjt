import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import random

class Qnet(nn.Module):
    def __init__(self):
        super(Qnet, self).__init__()
        self.conv1 = nn.Conv2d(2, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.fc1 = nn.Linear(1920, 128)
        self.fc2 = nn.Linear(128, 4)

    def forward(self, x):           # input size: (N, 2, 10, 9)
        x = x.type(torch.float32)
        x = F.relu(self.conv1(x))               # (N, 2, 10, 9) -> (N, 32, 8, 7)
        x = F.relu(self.conv2(x))               # (N, 32, 8, 7) -> (N, 64, 6, 5)
        x = torch.flatten(x, 1)                 # (N, 64, 30)
        x = F.relu(self.fc1(x))                 # (N, 1920) -> (N, 128)
        x = self.fc2(x)                         # (N, 128) -> (N, 4)
        return x


class Dueling_Qnet(nn.Module):
    def __init__(self):
        super(Dueling_Qnet, self).__init__()
        self.conv1 = nn.Conv2d(2, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.fc1_adv = nn.Linear(1920, 128)
        self.fc1_val = nn.Linear(1920, 128)
        self.fc2_adv = nn.Linear(128, 4)
        self.fc2_val = nn.Linear(128, 1)

    def forward(self, x):           # input size: (N, 2, 10, 9)
        x = x.type(torch.float32)
        x = F.relu(self.conv1(x))               # (N, 2, 10, 9) -> (N, 32, 8, 7)
        x = F.relu(self.conv2(x))               # (N, 32, 8, 7) -> (N, 64, 6, 5)
        x = torch.flatten(x, 1)                 # (N, 64, 30)
# Q(s,a) = V(s) + A(s,a) - A.mean
        # A(s,a) 루트
        adv = F.relu(self.fc1_adv(x))           # (N, 1920) -> (N, 128)
        adv = F.relu(self.fc2_adv(adv))         # (N, 128) -> (N, 4)

        # V(s) 루트
        val = F.relu(self.fc1_val(x))           # (N, 1920) -> (N, 128)
        val = F.relu(self.fc2_val(val))         # (N, 128) -> (N, 1)

        # adv.mean (N)이므로 (N, 1)로 차원 추가 필요
        x = val+adv-adv.mean(1).unsqueeze(1)    # (N, 4) (브로드캐스팅)
        return x
