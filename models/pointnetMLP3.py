# based on: https://github.com/fxia22/pointnet.pytorch/blob/master/utils/train_classification.py
import torch.nn as nn
from pointnet_pyt.pointnet.model_MLP3 import PointNetCls
from all_utils import DATASET_NUM_CLASS

class PointNetMLP3(nn.Module):

    def __init__(self, dataset, task, return_emb=False):
        super().__init__()
        self.task = task
        self.return_emb = return_emb
        num_class = DATASET_NUM_CLASS[dataset]
        self.model =  PointNetCls(k=num_class, pooling='max')

    def forward(self, pc):
        pc = pc.to(next(self.parameters()).device).float()
        logit,emb = self.model(pc)

        out = {'logit': logit}
        if self.return_emb:
            return out,emb
        else:
            return out


class PointNetMLP3Mean(nn.Module):

    def __init__(self, dataset, task, return_emb=False):
        super().__init__()
        self.task = task
        self.return_emb = return_emb
        num_class = DATASET_NUM_CLASS[dataset]
        self.model =  PointNetCls(k=num_class, pooling='mean')

    def forward(self, pc):
        pc = pc.to(next(self.parameters()).device).float()
        logit,emb = self.model(pc)

        out = {'logit': logit}
        if self.return_emb:
            return out,emb
        else:
            return out

