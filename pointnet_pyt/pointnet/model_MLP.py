from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F



class PointNetfeat(nn.Module):
    def __init__(self,num_layers=1, pooling='max'):
        super(PointNetfeat, self).__init__()
        self.fc1 = nn.Linear(3, 1024,bias=False)
            
        self.pooling = pooling

    def forward(self, x):
        emb = F.relu(self.fc1(x))
        if self.pooling == 'max':
            x = torch.max(emb, 1, keepdim=True)[0]
        elif self.pooling == 'mean':
            x = torch.mean(emb, 1, keepdim=True)
        x = x.view(-1, 1024)
        return x,emb

class PointNetCls(nn.Module):
    def __init__(self, k=2,num_layers=1, pooling='max'):
        super(PointNetCls, self).__init__()
        self.feat = PointNetfeat(num_layers=num_layers, pooling=pooling)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k)
        self.dropout = nn.Dropout(p=0.3)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.relu = nn.ReLU()

    def forward(self, x):
        x,emb = self.feat(x)
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.dropout(self.fc2(x))))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1), emb

if __name__ == '__main__':
    sim_data = Variable(torch.rand(32,3,2500))

    pointfeat = PointNetfeat()
    out, _, _ = pointfeat(sim_data)
    print('point feat', out.size())

    cls = PointNetCls(k = 5)
    out, _, _ = cls(sim_data)
    print('class', out.size())

    seg = PointNetDenseCls(k = 3)
    out, _, _ = seg(sim_data)
    print('seg', out.size())
