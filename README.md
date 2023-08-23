# Robust Point Cloud Processing through Positional Embedding 
### [Project Page](https://osiriszjq.github.io/RobustPPE) | [Paper]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)


[Jianqiao Zheng](https://github.com/osiriszjq/),
[Xueqian Li](https://lilac-lee.github.io/),
[Sameera Ramasinghe](https://scholar.google.pl/citations?user=-j0m9aMAAAAJ&hl=en),
[Simon Lucey](https://www.adelaide.edu.au/directory/simon.lucey)<br>
The University of Adelaide


This is the official implementation of the paper "Robust Point Cloud Processing through Positional Embedding". This codebase is based on [Benchmarking Robustness of 3D Point Cloud Recognition against Common Corruptions](https://github.com/jiachens/ModelNet40-C) by Jiachen Sun et al., and we thank the authors for their great contributions.


#### Illustration of different methods to extend 1D encoding
![Illustration of different methods to extend 1D encoding](imgs/method.pdf)


## Getting Started

The environment is same as [Benchmarking Robustness of 3D Point Cloud Recognition against Common Corruptions](https://github.com/jiachens/ModelNet40-C). The core steps are listed below.
#### Install Libraries
Install [Anaconda](https://anaconda.org/) and create a virtual environment.
```
conda create --name modelnetc python=3.7.5
```

Activate the virtual environment and install the libraries.
```
conda activate modelnetc
pip install -r requirements.txt
conda install sed  # for downloading data and pretrained models
```

For PointNet++, we need to install custom CUDA modules. Make sure you have access to a GPU during this step. You might need to set the appropriate `TORCH_CUDA_ARCH_LIST` environment variable depending on your GPU model. The following command should work for most cases `export TORCH_CUDA_ARCH_LIST="6.0;6.1;6.2;7.0;7.5"`. However, if the install fails, check if `TORCH_CUDA_ARCH_LIST` is correctly set. More details could be found [here](https://en.wikipedia.org/wiki/CUDA#GPUs_supported).

Third-party modules `pointnet2_pyt`, `PCT_Pytorch`, `emd`, and `PyGeM` can be installed by the following script.

```
./setup.sh
```

#### Datasets
To download ModelNet40 execute the following command. This will download the ModelNet40 point cloud dataset released with pointnet++ as well as the validation splits used in our work.
```
./download.sh modelnet40
```
To generate the ModelNet40-C dataset, please run:
```
python data/process.py
python data/generate_c.py
```
NOTE that the generation needs a monitor connected since Open3D library does not support background rendering. 

ModelNet40-C can be found in [Benchmarking Robustness of 3D Point Cloud Recognition against Common Corruptions](https://github.com/jiachens/ModelNet40-C). If you download ModelNet40-C directly, please fill this [Google form](https://docs.google.com/forms/d/e/1FAIpQLSdrzt8EtQdjGMlwIwWAzb39KzzVzijpK6-sPEaps07MjQwGGQ/viewform?usp=sf_link) as they required when you execute the following command.
```
./download.sh modelnet40_c
```
You can download our modified ModelNet40-C from [Google Drive](https://drive.google.com/drive/folders/106mMblD3HP93vhauG4PMirx01WP2j24y?usp=sharing).

 
## Running Experiments

#### Training and Config files
To train or test any model, we use the `main.py` script. The format for running this script is as follows. 
```
python main.py --exp-config <path to the config>
```
The train command we use is in `train.sh`. We only use dataset from DGCNN with simple augmetation (wihch is usually considered as "unaugmented").


#### Corruption test
To test a pretrained model with different corruptions, use command in the following format.

```
python main.py --entry test --model-path <cor_exp/runs>/<cfg_name>/<model_name>.pth --exp-config configs/corruptions/<cfg_name>.yaml
```

The evaluation commands we use is in the `eval_cor_modelnet40c.sh`, ``eval_cor_modelnet40c_our.sh`, `eval_tent_cutmix.sh` scripts.

## Citation
Please cite our paper and SimpleView if you use our benchmark and analysis results. Thank you!
```
```