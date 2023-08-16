# DCL
This is the PyTorch implementation for 《Weakly Supervised Learning for Point Cloud Semantic Segmentation with Dual Teacher》.

This code and framework are  implemented on [PointNeXt](https://github.com/guochengqian/PointNeXt).

>Point cloud semantic segmentation has achieved
considerable progress in the past decade. To alleviate expensive
data annotation efforts, weakly supervised learning methods
are preferable, and traditional approaches are typically based
on siamese neural networks. To enhance the feature learning
capability, in this work, we introduce a dual-teacher-guided
contrastive learning framework for weakly supervised point
cloud semantic segmentation. A dual-teacher framework can
reduce sub-network coupling and facilitate feature learning. In
addition, a cross-validation approach can filter out low-quality
samples, and a pseudo-label correction module can improve the
quality of pseudo-labels. Cleaned unlabeled data are used to
construct contrastive loss based on the prototypes of each class,
which further boost the segmentation performance. Extensive
experimental results conducted on the S3DIS, ScanNet-v2, and
SemanticKITTI datasets demonstrate that our proposed DCL
outperforms state-of-the-art methods.

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

## Citation
````
@article{yao2023weakly,
  title={Weakly Supervised Learning for Point Cloud Semantic Segmentation with Dual Teacher},
  author={Yao, Baochen and Xiao, Hui and Zhuang, Jiayan and Peng, Chengbin},
  journal={IEEE Robotics and Automation Letters},
  year={2023},
  publisher={IEEE}
}
````


## Acknowledgement
The code is built on PointNeXt. We thank the authors for sharing the codes.

