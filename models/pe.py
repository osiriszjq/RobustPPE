# based on: https://github.com/fxia22/pointnet.pytorch/blob/master/utils/train_classification.py

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from all_utils import DATASET_NUM_CLASS

class PEMax(nn.Module):

    def __init__(self, dataset, task, pe_param=None, pool_param=None, return_emb=False):
        super().__init__()
        self.task = task
        self.return_emb = return_emb
        num_class = DATASET_NUM_CLASS[dataset]
        self.model =  PECls(k=num_class, pooling='max', pe_param=pe_param, pool_param=pool_param)


    def forward(self, pc, cls=None):
        pc = pc.to(next(self.parameters()).device)
        logit,emb = self.model(pc)
        out = {'logit': logit}
        if self.return_emb:
            return out,emb
        else:
            return out
    
    
class PEHist(nn.Module):

    def __init__(self, dataset, task, pe_param=None, pool_param=None, return_emb=False):
        super().__init__()
        self.task = task
        self.return_emb = return_emb
        num_class = DATASET_NUM_CLASS[dataset]
        self.model =  PECls(k=num_class, pooling='hist', pe_param=pe_param, pool_param=pool_param)

    def forward(self, pc, cls=None):
        pc = pc.to(next(self.parameters()).device)
        logit,emb = self.model(pc)
        out = {'logit': logit}
        if self.return_emb:
            return out,emb
        else:
            return out
    
class PERansac(nn.Module):

    def __init__(self, dataset, task, pe_param=None, pool_param=None, return_emb=False):
        super().__init__()
        self.task = task
        self.return_emb = return_emb
        num_class = DATASET_NUM_CLASS[dataset]
        self.model =  PECls(k=num_class, pooling='ransac', pe_param=pe_param, pool_param=pool_param)
        

    def forward(self, pc, cls=None):
        pc = pc.to(next(self.parameters()).device)
        logit,emb = self.model(pc)
        out = {'logit': logit}
        if self.return_emb:
            return out,emb
        else:
            return out
    
    
class PEMean(nn.Module):

    def __init__(self, dataset, task, pe_param=None, pool_param=None, return_emb=False):
        super().__init__()
        self.task = task
        self.return_emb = return_emb
        num_class = DATASET_NUM_CLASS[dataset]
        self.model =  PECls(k=num_class, pooling='mean', pe_param=pe_param, pool_param=pool_param)
        

    def forward(self, pc, cls=None):
        pc = pc.to(next(self.parameters()).device)
        logit,emb = self.model(pc)
        out = {'logit': logit}
        if self.return_emb:
            return out,emb
        else:
            return out
    
    
class PEMedian(nn.Module):

    def __init__(self, dataset, task, pe_param=None, pool_param=None, return_emb=False):
        super().__init__()
        self.task = task
        self.return_emb = return_emb
        num_class = DATASET_NUM_CLASS[dataset]
        self.model =  PECls(k=num_class, pooling='median', pe_param=pe_param, pool_param=pool_param)
        

    def forward(self, pc, cls=None):
        pc = pc.to(next(self.parameters()).device)
        logit,emb = self.model(pc)
        

        out = {'logit': logit}
        if self.return_emb:
            return out,emb
        else:
            return out

    

class PECls(nn.Module):
    def __init__(self, k=2, pooling='max', pe_param=None, pool_param=None):
        super(PECls, self).__init__()
        self.pooling = pooling
        self.pe_param = pe_param
        self.pool_param = pool_param
        self.feat = encoding_func_3D('RFF',pe_param)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k)
        self.dropout = nn.Dropout(p=0.3)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.relu = nn.ReLU()

    def forward(self, x):
        emb = self.feat(x)
        if self.pooling == 'max':
            x = torch.max(emb, 1, keepdim=False)[0]
        elif self.pooling == 'mean':
            x = torch.mean(emb, 1, keepdim=False)
        elif self.pooling == 'median':
            x = torch.median(emb, 1, keepdim=False)[0]
        elif self.pooling == 'hist':
            x = hist(emb,self.pool_param)
        elif self.pooling == 'ransac':
            x = ransac(emb,self.pool_param)
        else:
            assert False
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.dropout(self.fc2(x))))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1),emb
    
    

# simple 3D encoding. All method use 3 directions.
class encoding_func_3D:
    def __init__(self, name, param=None):
        self.name = name

        if name == 'none': self.dim=2
        elif name == 'basic': self.dim=4
        else:
            self.dim = param[1]
            if name == 'RFF':
                self.b = param[0]*torch.randn((int(param[1]/2),3)).cuda()
            elif name == 'rffb':
                self.b = param[0]
            elif name == 'LinF':
                self.b = torch.linspace(2.**0., 2.**param[0], steps=int(param[1]/6)).reshape(-1,1)
            elif name == 'LogF':
                self.b = 2.**torch.linspace(0., param[0], steps=int(param[1]/6)).reshape(-1,1)
            elif name == 'Gau':
                self.dic = torch.linspace(-1., 1, steps=int(param[1]/3)+1)[:-1].reshape(1,-1).cuda()
                self.sig = param[0]
            elif name == 'Gau':
                self.dic = torch.linspace(-1., 1, steps=int(param[1]/3)+1)[:-1].reshape(1,-1).cuda()
                self.sig = param[0]
            elif name == 'Tri':
                self.dic = torch.linspace(-1., 1, steps=int(param[1]/3)+1)[:-1].reshape(1,-1)
                if param[0] is None: self.d = 1/param[1]
                else: self.d = param[0]
            else:
                print('Undifined encoding!')
    def __call__(self, x):
        if self.name == 'none':
            return x
        elif self.name == 'basic':
            emb = torch.cat((torch.sin((2.*np.pi*x)),torch.cos((2.*np.pi*x))),-1)
            emb = emb/(emb.norm(dim=1).max())
            return emb
        elif (self.name == 'RFF')|(self.name == 'rffb'):
            emb = torch.cat((torch.sin((2.*np.pi*x) @ self.b.T),torch.cos((2.*np.pi*x) @ self.b.T)),-1)
            #emb = emb/(emb.norm(dim=1).max())
            return emb
        elif (self.name == 'LinF')|(self.name == 'LogF'):
            emb1 = torch.cat((torch.sin((2.*np.pi*x[:,:1]) @ self.b.T),torch.cos((2.*np.pi*x[:,:1]) @ self.b.T)),1)
            emb2 = torch.cat((torch.sin((2.*np.pi*x[:,1:2]) @ self.b.T),torch.cos((2.*np.pi*x[:,1:2]) @ self.b.T)),1)
            emb3 = torch.cat((torch.sin((2.*np.pi*x[:,2:3]) @ self.b.T),torch.cos((2.*np.pi*x[:,2:3]) @ self.b.T)),1)
            emb = torch.cat([emb1,emb2,emb3],-1)
            emb = emb/(emb.norm(dim=1).max())
            return emb
        elif self.name == 'Gau':
            emb1 = (-0.5*(x[...,:1]-self.dic)**2/(self.sig**2)).exp()
            emb2 = (-0.5*(x[...,1:2]-self.dic)**2/(self.sig**2)).exp()
            emb3 = (-0.5*(x[...,2:3]-self.dic)**2/(self.sig**2)).exp()
            emb = torch.cat([emb1,emb2,emb3],-1)
            #emb = emb/(emb.norm(dim=1).max())
            return emb
        elif self.name == 'Tri':
            emb1 = (1-(x[:,:1]-self.dic).abs()/self.d)
            emb1 = emb1*(emb1>0)
            emb2 = (1-(x[:,1:2]-self.dic).abs()/self.d)
            emb2 = emb2*(emb2>0)
            emb3 = (1-(x[:,2:3]-self.dic).abs()/self.d)
            emb3 = emb3*(emb3>0)
            emb = torch.cat([emb1,emb2,emb3],-1)
            emb = emb/(emb.norm(dim=1).max())
            return emb
    
    
def hist(x,nbins):
    x = x.clamp(-1+1e-3,1)
    x = x+1
    x = x/2.0
    x = x*nbins
    x = x.ceil()
    x = x/nbins*2-1
    x = x-1.0/nbins
    x = x.mode(1)[0]
    return x
