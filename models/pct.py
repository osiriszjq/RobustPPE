import torch.nn as nn
from PCT_Pytorch.model import Pct as Pct_original
from PCT_Pytorch.model import Pctes as Pctes_original
from PCT_Pytorch.model import Pcte as Pcte_original
from PCT_Pytorch.model import Pctc as Pctc_original
from PCT_Pytorch.model import Pctcmean as Pctcmean_original
from PCT_Pytorch.model import peat as peat_original
from PCT_Pytorch.model import peatmean as peatmean_original
from all_utils import DATASET_NUM_CLASS

class Pct(nn.Module):

    def __init__(self, task, dataset):
        super().__init__()
        self.task = task
        self.dataset = dataset

        if task == "cls":
            num_classes = DATASET_NUM_CLASS[dataset]
            # default arguments
            class Args:
                def __init__(self):
                    self.dropout = 0.5
            args = Args()
            self.model = Pct_original(args, output_channels=num_classes)

        else:
            assert False

    def forward(self, pc, cls=None):
        pc = pc.to(next(self.parameters()).device)
        pc = pc.permute(0, 2, 1).contiguous()
        if self.task == 'cls':
            assert cls is None
            logit = self.model(pc)
            out = {'logit': logit}
        else:
            assert False

        return out


class Pctes(nn.Module):

    def __init__(self, task, dataset, pe_param=None):
        super().__init__()
        self.task = task
        self.dataset = dataset

        if task == "cls":
            num_classes = DATASET_NUM_CLASS[dataset]
            # default arguments
            class Args:
                def __init__(self):
                    self.dropout = 0.5
            args = Args()
            self.model = Pctes_original(args, output_channels=num_classes,pe_param=pe_param)

        else:
            assert False

    def forward(self, pc, cls=None):
        pc = pc.to(next(self.parameters()).device)
        pc = pc.permute(0, 2, 1).contiguous()
        if self.task == 'cls':
            assert cls is None
            logit = self.model(pc)
            out = {'logit': logit}
        else:
            assert False

        return out
    

class Pcte(nn.Module):

    def __init__(self, task, dataset, pe_param=None):
        super().__init__()
        self.task = task
        self.dataset = dataset

        if task == "cls":
            num_classes = DATASET_NUM_CLASS[dataset]
            # default arguments
            class Args:
                def __init__(self):
                    self.dropout = 0.5
            args = Args()
            self.model = Pcte_original(args, output_channels=num_classes,pe_param=pe_param)

        else:
            assert False

    def forward(self, pc, cls=None):
        pc = pc.to(next(self.parameters()).device)
        pc = pc.permute(0, 2, 1).contiguous()
        if self.task == 'cls':
            assert cls is None
            logit = self.model(pc)
            out = {'logit': logit}
        else:
            assert False

        return out
    
    
class Pctc(nn.Module):

    def __init__(self, task, dataset):
        super().__init__()
        self.task = task
        self.dataset = dataset

        if task == "cls":
            num_classes = DATASET_NUM_CLASS[dataset]
            # default arguments
            class Args:
                def __init__(self):
                    self.dropout = 0.5
            args = Args()
            self.model = Pctc_original(args, output_channels=num_classes)

        else:
            assert False

    def forward(self, pc, cls=None):
        pc = pc.to(next(self.parameters()).device)
        pc = pc.permute(0, 2, 1).contiguous()
        if self.task == 'cls':
            assert cls is None
            logit = self.model(pc)
            out = {'logit': logit}
        else:
            assert False

        return out
    
    
    
class Pctcmean(nn.Module):

    def __init__(self, task, dataset):
        super().__init__()
        self.task = task
        self.dataset = dataset

        if task == "cls":
            num_classes = DATASET_NUM_CLASS[dataset]
            # default arguments
            class Args:
                def __init__(self):
                    self.dropout = 0.5
            args = Args()
            self.model = Pctcmean_original(args, output_channels=num_classes)

        else:
            assert False

    def forward(self, pc, cls=None):
        pc = pc.to(next(self.parameters()).device)
        pc = pc.permute(0, 2, 1).contiguous()
        if self.task == 'cls':
            assert cls is None
            logit = self.model(pc)
            out = {'logit': logit}
        else:
            assert False

        return out
    
    
    
class peat(nn.Module):

    def __init__(self, task, dataset, pe_param=None):
        super().__init__()
        self.task = task
        self.dataset = dataset

        if task == "cls":
            num_classes = DATASET_NUM_CLASS[dataset]
            # default arguments
            class Args:
                def __init__(self):
                    self.dropout = 0.5
            args = Args()
            self.model = peat_original(args, output_channels=num_classes,pe_param=pe_param)

        else:
            assert False

    def forward(self, pc, cls=None):
        pc = pc.to(next(self.parameters()).device)
        pc = pc.permute(0, 2, 1).contiguous()
        if self.task == 'cls':
            assert cls is None
            logit = self.model(pc)
            out = {'logit': logit}
        else:
            assert False

        return out
    
    
    
class peatmean(nn.Module):

    def __init__(self, task, dataset, pe_param=None):
        super().__init__()
        self.task = task
        self.dataset = dataset

        if task == "cls":
            num_classes = DATASET_NUM_CLASS[dataset]
            # default arguments
            class Args:
                def __init__(self):
                    self.dropout = 0.5
            args = Args()
            self.model = peatmean_original(args, output_channels=num_classes,pe_param=pe_param)

        else:
            assert False

    def forward(self, pc, cls=None):
        pc = pc.to(next(self.parameters()).device)
        pc = pc.permute(0, 2, 1).contiguous()
        if self.task == 'cls':
            assert cls is None
            logit = self.model(pc)
            out = {'logit': logit}
        else:
            assert False

        return out
