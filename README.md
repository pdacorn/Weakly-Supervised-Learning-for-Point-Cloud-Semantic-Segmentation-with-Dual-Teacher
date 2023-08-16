# DCL
This code and framework are  implemented on [PointNeXt](https://github.com/guochengqian/PointNeXt)

## Environment and Datasets
This codebase was tested with the following environment configurations.

* Ubuntu 22.04
* Python 3.7
* CUDA 11.3
* Pytorch 1.10.1

Please refer to PointNeXt to install other required packages and download datasets.

## Usage

1. To accelerate and stabilize the training process, the first step is to train using only labeled data. After this step, the model is saved in the log folder. 

````
run ./pre_supervised/pre_supervised.py
````

2. Training all data based on pre_supervised model, and set the model folder path in cfg_s3dis.yaml file.
````
run ./DCL/main.py
````

## Acknowledgement
The code is built on PointNeXt. We thank the authors for sharing the codes.

