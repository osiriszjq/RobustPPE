###  Generate Various Common Corruptions ###
from operator import index
import os
import h5py
import argparse
import numpy as np
from convert import *
from util import *


### Transformation ###
### Noise ###
'''
Add Uniform noise to point cloud 
'''
def uniform_noise(pointcloud, severity):
    #TODO
    N, C = pointcloud.shape
    c = [0.01, 0.02, 0.03, 0.04, 0.05,0.06,0.07,0.08,0.09,0.1][severity-1]
    jitter = np.random.uniform(-c,c,(N, C))
    new_pc = (pointcloud + jitter).astype('float32')
    new_pc = np.clip(new_pc,-1,1)
    return new_pc

'''
Add Gaussian noise to point cloud 
'''
def gaussian_noise(pointcloud, severity):
    N, C = pointcloud.shape
    c = [0.01, 0.02, 0.03, 0.04, 0.05,0.06,0.07,0.08,0.09,0.1][severity-1]
    jitter = np.random.normal(size=(N, C)) * c
    new_pc = (pointcloud + jitter).astype('float32')
    new_pc = np.clip(new_pc,-1,1)
    return new_pc

'''
Add impulse noise
'''
def impulse_noise(pointcloud, severity):
    N, C = pointcloud.shape
    c = [N//45, N//40, N//35, N//30, N//20, N//15, N//10, N//5, N//2, N//1][severity-1]
    index = np.random.choice(ORIG_NUM, c, replace=False)
    pointcloud[index] += np.random.choice([-1,1], size=(c,C)) * 0.1
    pointcloud = np.clip(pointcloud,-1,1)
    return pointcloud
    #return normalize(pointcloud)


### Outliers ###
'''
Add background outliers to point cloud 
'''
def background_noise(pointcloud, severity):
    N, C = pointcloud.shape
    c = [N//45, N//40, N//35, N//30, N//20, N//15, N//10, N//5, N//2, N//1][severity-1]
    jitter = np.random.uniform(-1,1,(c, C))
    new_pc = np.concatenate((pointcloud,jitter),axis=0).astype('float32')
    new_pc = np.clip(new_pc,-1,1)
    return new_pc
    #return normalize(new_pc)

'''
Add 10% ball outliers to point cloud 
'''
def ball_noise_1(pointcloud, severity, p=10):
    N, C = pointcloud.shape
    c = N//p
    r = [0.2,0.4,0.6,0.8,1.0,1.2,1.4,1.6,1.8,2.0][severity-1]
    jitter = np.random.normal(size=(c, C))
    jitter = jitter/(np.sqrt((jitter**2).sum(-1,keepdims=True)))
    jitter = jitter*r
    new_pc = np.concatenate((pointcloud,jitter),axis=0).astype('float32')
    new_pc = np.clip(new_pc,-1,1)
    return new_pc
    #return normalize(new_pc)

'''
Add 50% ball outliers to point cloud 
'''
def ball_noise_2(pointcloud, severity, p=2):
    N, C = pointcloud.shape
    c = N//p
    r = [0.2,0.4,0.6,0.8,1.0,1.2,1.4,1.6,1.8,2.0][severity-1]
    jitter = np.random.normal(size=(c, C))
    jitter = jitter/(np.sqrt((jitter**2).sum(-1,keepdims=True)))
    jitter = jitter*r
    new_pc = np.concatenate((pointcloud,jitter),axis=0).astype('float32')
    new_pc = np.clip(new_pc,-1,1)
    return new_pc
    #return normalize(new_pc)

'''
Add 100% ball outliers to point cloud 
'''
def ball_noise_3(pointcloud, severity, p=1):
    N, C = pointcloud.shape
    c = N//p
    r = [0.2,0.4,0.6,0.8,1.0,1.2,1.4,1.6,1.8,2.0][severity-1]
    jitter = np.random.normal(size=(c, C))
    jitter = jitter/(np.sqrt((jitter**2).sum(-1,keepdims=True)))
    jitter = jitter*r
    new_pc = np.concatenate((pointcloud,jitter),axis=0).astype('float32')
    new_pc = np.clip(new_pc,-1,1)
    return new_pc
    #return normalize(new_pc)

'''
Upsampling
'''
def upsampling(pointcloud, severity):
    N, C = pointcloud.shape
    c = [N//10, N//9, N//8, N//7, N//6, N//5, N//4, N//3, N//2, N][severity-1]
    index = np.random.choice(ORIG_NUM, c, replace=False)
    add = pointcloud[index] + np.random.uniform(-0.05,0.05,(c, C))
    new_pc = np.concatenate((pointcloud,add),axis=0).astype('float32')
    new_pc = np.clip(new_pc,-1,1)
    return new_pc
    #return normalize(new_pc)
    

### Density ###
'''
Density-based up-sampling the point cloud
'''
def density_inc(pointcloud, severity):
    N, C = pointcloud.shape
    c = [(1,100), (2,100), (3,100), (4,100), (5,100),(6,100), (7,100), (8,100), (9,100), (10,100)][severity-1]
    # idx = np.random.choice(N,c[0])
    temp = []
    for _ in range(c[0]):
        i = np.random.choice(pointcloud.shape[0],1)
        picked = pointcloud[i]
        dist = np.sum((pointcloud - picked)**2, axis=1, keepdims=True)
        idx = np.argpartition(dist, c[1], axis=0)[:c[1]]
        # idx_2 = np.random.choice(c[1],int((3/4) * c[1]),replace=False)
        # idx = idx[idx_2]
        temp.append(pointcloud[idx.squeeze()])
        pointcloud = np.delete(pointcloud, idx.squeeze(), axis=0)
    
    idx = np.random.choice(pointcloud.shape[0],1024 - c[0] * c[1])
    temp.append(pointcloud[idx.squeeze()])

    pointcloud = np.concatenate(temp)
    # print(pointcloud.shape)
    return pointcloud

'''
Density-based sampling the point cloud
'''
def density(pointcloud, severity):
    N, C = pointcloud.shape
    c = [(1,100), (2,100), (3,100), (4,100), (5,100),(6,100), (7,100), (8,100), (9,100), (10,100)][severity-1]
    for _ in range(c[0]):
        i = np.random.choice(pointcloud.shape[0],1)
        picked = pointcloud[i]
        dist = np.sum((pointcloud - picked)**2, axis=1, keepdims=True)
        idx = np.argpartition(dist, c[1], axis=0)[:c[1]]
        idx_2 = np.random.choice(c[1],int((3/4) * c[1]),replace=False)
        idx = idx[idx_2]
        pointcloud = np.delete(pointcloud, idx.squeeze(), axis=0)
        # pointcloud[idx.squeeze()] = 0
    # print(pointcloud.shape)
    return pointcloud


def load_data():
    os.makedirs(folder_name,exist_ok = True)
    modelnet40_dir = "./data/modelnet40_ply_hdf5_2048/"
    modelnet40_test_file = os.path.join(modelnet40_dir, "test_files.txt")
    with open(modelnet40_test_file, "r") as f:
        modelnet40_test_paths = [l.strip() for l in f.readlines()]

    data   = []
    labels = []
    for modelnet40_test_path in modelnet40_test_paths:
        test_h5 = h5py.File(modelnet40_test_path, "r")

        data.append(test_h5["data"][:])
        labels.append(test_h5["label"][:])

    data   = np.concatenate(data)
    labels = np.concatenate(labels)

    np.save(folder_name+"/label.npy", labels)

    return data, labels


def save_data(data,corruption,severity):

    if not MAP[corruption]:
        np.save(folder_name+"/data_" + corruption + ".npy", data)
        return
        
    new_data = []
    for i in range(data.shape[0]):
        if corruption in ['occlusion', 'lidar']:
            new_data.append(MAP[corruption](severity))
        else:
            new_data.append(MAP[corruption](data[i],severity))
    new_data = np.stack(new_data,axis=0)
    np.save(folder_name+"/data_" + corruption + "_" + str(severity) + ".npy", new_data)


MAP = {
    'ball_l': ball_noise_1,
    'ball_m': ball_noise_2,
    'ball_h': ball_noise_3,
    'uniform': uniform_noise,
    'gaussian': gaussian_noise,
    'background': background_noise,
    'impulse': impulse_noise,
    'upsampling': upsampling,
    #    'density': density,
    #    'density_inc': density_inc,
    'original': None,
}

ORIG_NUM = 1024

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--iid', type=int, default="1")
    cmd_args = parser.parse_args()
    iid = cmd_args.iid
    folder_name = "./data/modelnet40_c_f"+str(iid)
    np.random.seed(2020+iid)
    data, labels = load_data()
    for cor in MAP.keys():
        for sev in [1,2,3,4,5,6,7,8,9,10]:
            if cor == 'density_inc':
                ORIG_NUM = 2048
            else:
                ORIG_NUM = 1024
            index = np.random.choice(data.shape[1],ORIG_NUM,replace=False)
            save_data(data[:,index,:], cor, sev)
            print("Done with Corruption: {} with Severity: {}".format(cor,sev))

