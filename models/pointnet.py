# based on: https://github.com/fxia22/pointnet.pytorch/blob/master/utils/train_classification.py
import torch.nn as nn
from pointnet_pyt.pointnet.model import PointNetCls
from all_utils import DATASET_NUM_CLASS

class PointNet(nn.Module):

    def __init__(self, dataset, task, return_emb=False):
        super().__init__()
        self.task = task
        self.return_emb = return_emb
        num_class = DATASET_NUM_CLASS[dataset]
        if self.task == 'cls_trans':
            self.model =  PointNetCls(k=num_class, feature_transform=True, pooling='max')
        else:
            assert False

    def forward(self, pc, cls=None):
        pc = pc.to(next(self.parameters()).device)
        pc = pc.transpose(2, 1).float()
        if self.task == 'cls_trans':
            logit, _, trans_feat,emb = self.model(pc)
        else:
            assert False

        out = {'logit': logit, 'trans_feat': trans_feat}
        if self.return_emb:
            return out,emb
        else:
            return out


class PointNetMean(nn.Module):

    def __init__(self, dataset, task, return_emb=False):
        super().__init__()
        self.task = task
        self.return_emb = return_emb
        num_class = DATASET_NUM_CLASS[dataset]
        if self.task == 'cls_trans':
            self.model =  PointNetCls(k=num_class, feature_transform=True, pooling='mean')
        else:
            assert False

    def forward(self, pc, cls=None):
        pc = pc.to(next(self.parameters()).device)
        pc = pc.transpose(2, 1).float()
        if self.task == 'cls_trans':
            logit, _, trans_feat,emb = self.model(pc)
        else:
            assert False

        out = {'logit': logit, 'trans_feat': trans_feat}
        if self.return_emb:
            return out,emb
        else:
            return out

