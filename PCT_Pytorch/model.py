import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .util import sample_and_group 


class Local_op(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Local_op, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.bn2 = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        b, n, s, d = x.size()  # torch.Size([32, 512, 32, 6]) 
        x = x.permute(0, 1, 3, 2)   
        x = x.reshape(-1, d, s) 
        batch_size, _, N = x.size()
        x = F.relu(self.bn1(self.conv1(x))) # B, D, N
        x = F.relu(self.bn2(self.conv2(x))) # B, D, N
        x = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        x = x.reshape(b, n, -1).permute(0, 2, 1)
        return x


class Pct(nn.Module):
    def __init__(self, args, output_channels=40):
        super(Pct, self).__init__()
        self.args = args
        # self.conv1 = nn.Conv1d(3, 64, kernel_size=1, bias=False)
        # self.conv2 = nn.Conv1d(64, 64, kernel_size=1, bias=False)
        # self.bn1 = nn.BatchNorm1d(64)
        # self.bn2 = nn.BatchNorm1d(64)
        # self.gather_local_0 = Local_op(in_channels=128, out_channels=128)
        # self.gather_local_1 = Local_op(in_channels=256, out_channels=256)

        # self.pt_last = Point_Transformer_Last(args)

        # self.conv_fuse = nn.Sequential(nn.Conv1d(1280, 1024, kernel_size=1, bias=False),
        #                             nn.BatchNorm1d(1024),
        #                             nn.LeakyReLU(negative_slope=0.2))

        self.feat = Pct_feat(args)


        self.linear1 = nn.Linear(1024, 512, bias=False)
        self.bn6 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(p=args.dropout)
        self.linear2 = nn.Linear(512, 256)
        self.bn7 = nn.BatchNorm1d(256)
        self.dp2 = nn.Dropout(p=args.dropout)
        self.linear3 = nn.Linear(256, output_channels)

    def forward(self, x):
        # xyz = x.permute(0, 2, 1)
        # batch_size, _, _ = x.size()
        # # B, D, N
        # x = F.relu(self.bn1(self.conv1(x)))
        # # B, D, N
        # x = F.relu(self.bn2(self.conv2(x)))
        # x = x.permute(0, 2, 1)
        # new_xyz, new_feature = sample_and_group(npoint=512, radius=0.15, nsample=32, xyz=xyz, points=x)         
        # feature_0 = self.gather_local_0(new_feature)
        # feature = feature_0.permute(0, 2, 1)
        # new_xyz, new_feature = sample_and_group(npoint=256, radius=0.2, nsample=32, xyz=new_xyz, points=feature) 
        # feature_1 = self.gather_local_1(new_feature)


        # x = self.pt_last(feature_1)
        # x = torch.cat([x, feature_1], dim=1)
        # x = self.conv_fuse(x)
        batch_size, _, _ = x.size()
        x = self.feat(x)
        x = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        x = F.leaky_relu(self.bn6(self.linear1(x)), negative_slope=0.2)
        x = self.dp1(x)
        x = F.leaky_relu(self.bn7(self.linear2(x)), negative_slope=0.2)
        x = self.dp2(x)
        x = self.linear3(x)

        return x
    

class Pct_feat(nn.Module):
    def __init__(self, args):
        super(Pct_feat, self).__init__()
        self.args = args
        self.conv1 = nn.Conv1d(3, 64, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(64, 64, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.gather_local_0 = Local_op(in_channels=128, out_channels=128)
        self.gather_local_1 = Local_op(in_channels=256, out_channels=256)

        self.pt_last = Point_Transformer_Last(args)

        self.conv_fuse = nn.Sequential(nn.Conv1d(1280, 1024, kernel_size=1, bias=False),
                                    nn.BatchNorm1d(1024),
                                    nn.LeakyReLU(negative_slope=0.2))

    def forward(self, x):
        xyz = x.permute(0, 2, 1)
        # B, D, N
        x = F.relu(self.bn1(self.conv1(x)))
        # B, D, N
        x = F.relu(self.bn2(self.conv2(x)))
        x = x.permute(0, 2, 1)
        new_xyz, new_feature = sample_and_group(npoint=512, radius=0.15, nsample=32, xyz=xyz, points=x)         
        feature_0 = self.gather_local_0(new_feature)
        feature = feature_0.permute(0, 2, 1)
        new_xyz, new_feature = sample_and_group(npoint=256, radius=0.2, nsample=32, xyz=new_xyz, points=feature) 
        feature_1 = self.gather_local_1(new_feature)


        x = self.pt_last(feature_1)
        x = torch.cat([x, feature_1], dim=1)
        x = self.conv_fuse(x)

        return x


class Pctc(nn.Module):
    def __init__(self, args, output_channels=40):
        super(Pctc, self).__init__()
        self.args = args
        self.feat = Pct_feat(args)


        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, output_channels)
        self.dropout = nn.Dropout(p=0.3)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.relu = nn.ReLU()

    def forward(self, x):
        batch_size, _, _ = x.size()
        x = self.feat(x)
        x = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.dropout(self.fc2(x))))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)
    

class Pctcmean(nn.Module):
    def __init__(self, args, output_channels=40):
        super(Pctcmean, self).__init__()
        self.args = args
        self.feat = Pct_feat(args)


        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, output_channels)
        self.dropout = nn.Dropout(p=0.3)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.relu = nn.ReLU()

    def forward(self, x):
        batch_size, _, _ = x.size()
        x = self.feat(x)
        x = x.mean(dim=-1).view(batch_size, -1)
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.dropout(self.fc2(x))))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)


class Pctes(nn.Module):
    def __init__(self, args, output_channels=40, pe_param=None):
        super(Pctes, self).__init__()
        self.args = args
        self.conv1 = nn.Conv1d(3, 64, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(64, 64, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.gather_local_0 = Local_op(in_channels=128, out_channels=128)
        self.gather_local_1 = Local_op(in_channels=256, out_channels=256)

        self.pt_last = Point_Transformer_Last(args)

        self.conv_fuse = nn.Sequential(nn.Conv1d(1280, 1024, kernel_size=1, bias=False),
                                    nn.BatchNorm1d(1024),
                                    nn.LeakyReLU(negative_slope=0.2))


        self.linear1 = nn.Linear(1024, 512, bias=False)
        self.bn6 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(p=args.dropout)
        self.linear2 = nn.Linear(512, 256)
        self.bn7 = nn.BatchNorm1d(256)
        self.dp2 = nn.Dropout(p=args.dropout)
        self.linear3 = nn.Linear(256, output_channels)

        self.feat = encoding_func_3D('RFF',pe_param)

    def forward(self, x):
        xyz = x.permute(0, 2, 1)
        batch_size, _, _ = x.size()
        
        x = self.feat(xyz)

        new_xyz, new_feature = sample_and_group(npoint=512, radius=0.15, nsample=32, xyz=xyz, points=x)         
        feature_0 = self.gather_local_0(new_feature)
        feature = feature_0.permute(0, 2, 1)
        new_xyz, new_feature = sample_and_group(npoint=256, radius=0.2, nsample=32, xyz=new_xyz, points=feature) 
        feature_1 = self.gather_local_1(new_feature)

        x = self.pt_last(feature_1)
        x = torch.cat([x, feature_1], dim=1)
        x = self.conv_fuse(x)
        # x = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        x = x.mean(dim=-1).view(batch_size, -1)
        x = F.leaky_relu(self.bn6(self.linear1(x)), negative_slope=0.2)
        x = self.dp1(x)
        x = F.leaky_relu(self.bn7(self.linear2(x)), negative_slope=0.2)
        x = self.dp2(x)
        x = self.linear3(x)

        return x


class Pcte(nn.Module):
    def __init__(self, args, output_channels=40, pe_param=None):
        super(Pcte, self).__init__()
        self.args = args
        self.conv1 = nn.Conv1d(3, 64, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(64, 64, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.gather_local_0 = Local_op(in_channels=128, out_channels=128)
        self.gather_local_1 = Local_op(in_channels=256, out_channels=256)

        self.pt_last = Point_Transformer_Last(args)

        self.conv_fuse = nn.Sequential(nn.Conv1d(1280, 1024, kernel_size=1, bias=False),
                                    nn.BatchNorm1d(1024),
                                    nn.LeakyReLU(negative_slope=0.2))


        self.linear1 = nn.Linear(1024, 512, bias=False)
        self.bn6 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(p=args.dropout)
        self.linear2 = nn.Linear(512, 256)
        self.bn7 = nn.BatchNorm1d(256)
        self.dp2 = nn.Dropout(p=args.dropout)
        self.linear3 = nn.Linear(256, output_channels)

        self.feat = encoding_func_3D('RFF',pe_param)

    def forward(self, x):
        xyz = x.permute(0, 2, 1)
        batch_size, _, _ = x.size()
        
        feature_1 = self.feat(xyz)
        feature_1 = feature_1.permute(0,2,1)

        x = self.pt_last(feature_1)
        x = torch.cat([x, feature_1], dim=1)
        x = self.conv_fuse(x)
        # x = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        x = x.mean(dim=-1).view(batch_size, -1)
        x = F.leaky_relu(self.bn6(self.linear1(x)), negative_slope=0.2)
        x = self.dp1(x)
        x = F.leaky_relu(self.bn7(self.linear2(x)), negative_slope=0.2)
        x = self.dp2(x)
        x = self.linear3(x)

        return x


class peat(nn.Module):
    def __init__(self, args, output_channels=40, pe_param=None):
        super(peat, self).__init__()
        self.args = args
        self.conv1 = nn.Conv1d(3, 64, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(64, 64, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.gather_local_0 = Local_op(in_channels=128, out_channels=128)
        self.gather_local_1 = Local_op(in_channels=256, out_channels=256)

        self.feat = Point_Transformer_Simple(args)

        self.conv_fuse = nn.Sequential(nn.Conv1d(1280, 1024, kernel_size=1, bias=False),
                                    nn.BatchNorm1d(1024),
                                    nn.LeakyReLU(negative_slope=0.2))


        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, output_channels)
        self.dropout = nn.Dropout(p=0.3)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.relu = nn.ReLU()

        self.pe = encoding_func_3D('RFF',pe_param)

    def forward(self, x):
        xyz = x.permute(0, 2, 1)
        batch_size, _, _ = x.size()
        
        feature_1 = self.pe(xyz)
        feature_1 = feature_1.permute(0,2,1)

        x = self.feat(feature_1)
        x = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.dropout(self.fc2(x))))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)

class peatmean(nn.Module):
    def __init__(self, args, output_channels=40, pe_param=None):
        super(peatmean, self).__init__()
        self.args = args
        self.conv1 = nn.Conv1d(3, 64, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(64, 64, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.gather_local_0 = Local_op(in_channels=128, out_channels=128)
        self.gather_local_1 = Local_op(in_channels=256, out_channels=256)

        self.feat = Point_Transformer_Simple(args)

        self.conv_fuse = nn.Sequential(nn.Conv1d(1280, 1024, kernel_size=1, bias=False),
                                    nn.BatchNorm1d(1024),
                                    nn.LeakyReLU(negative_slope=0.2))


        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, output_channels)
        self.dropout = nn.Dropout(p=0.3)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.relu = nn.ReLU()

        self.pe = encoding_func_3D('RFF',pe_param)

    def forward(self, x):
        xyz = x.permute(0, 2, 1)
        batch_size, _, _ = x.size()
        
        feature_1 = self.pe(xyz)
        feature_1 = feature_1.permute(0,2,1)

        x = self.feat(feature_1)
        x = x.mean(dim=-1).view(batch_size, -1)
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.dropout(self.fc2(x))))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)

class Point_Transformer_Last(nn.Module):
    def __init__(self, args, channels=256):
        super(Point_Transformer_Last, self).__init__()
        self.args = args
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=1, bias=False)

        self.bn1 = nn.BatchNorm1d(channels)
        self.bn2 = nn.BatchNorm1d(channels)

        self.sa1 = SA_Layer(channels)
        self.sa2 = SA_Layer(channels)
        self.sa3 = SA_Layer(channels)
        self.sa4 = SA_Layer(channels)

    def forward(self, x):
        # 
        # b, 3, npoint, nsample  
        # conv2d 3 -> 128 channels 1, 1
        # b * npoint, c, nsample 
        # permute reshape
        batch_size, _, N = x.size()

        # B, D, N
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x1 = self.sa1(x)
        x2 = self.sa2(x1)
        x3 = self.sa3(x2)
        x4 = self.sa4(x3)
        x = torch.cat((x1, x2, x3, x4), dim=1)

        return x
    
    
class Point_Transformer_Simple(nn.Module):
    def __init__(self, args, channels=256):
        super(Point_Transformer_Simple, self).__init__()
        self.args = args

        self.sa1 = SA_Layer(channels)
        self.sa2 = SA_Layer(channels)
        self.sa3 = SA_Layer(channels)
        self.sa4 = SA_Layer(channels)

    def forward(self, x):
        # 
        # b, 3, npoint, nsample  
        # conv2d 3 -> 128 channels 1, 1
        # b * npoint, c, nsample 
        # permute reshape

        # B, D, N
        x1 = self.sa1(x)
        x2 = self.sa2(x1)
        x3 = self.sa3(x2)
        x4 = self.sa4(x3)
        x = torch.cat((x1, x2, x3, x4), dim=1)

        return x
    

class SA_Layer(nn.Module):
    def __init__(self, channels):
        super(SA_Layer, self).__init__()
        self.q_conv = nn.Conv1d(channels, channels // 4, 1, bias=False)
        self.k_conv = nn.Conv1d(channels, channels // 4, 1, bias=False)
        self.q_conv.weight = self.k_conv.weight
        self.q_conv.bias = self.k_conv.bias

        self.v_conv = nn.Conv1d(channels, channels, 1)
        self.trans_conv = nn.Conv1d(channels, channels, 1)
        self.after_norm = nn.BatchNorm1d(channels)
        self.act = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        # b, n, c
        x_q = self.q_conv(x).permute(0, 2, 1)
        # b, c, n
        x_k = self.k_conv(x)
        x_v = self.v_conv(x)
        # b, n, n
        energy = torch.bmm(x_q, x_k)

        attention = self.softmax(energy)
        attention = attention / (1e-9 + attention.sum(dim=1, keepdim=True))
        # b, c, n
        x_r = torch.bmm(x_v, attention)
        x_r = self.act(self.after_norm(self.trans_conv(x - x_r)))
        x = x + x_r
        return x
    


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

