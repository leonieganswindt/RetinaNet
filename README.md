# RetinaNet_Mating

Object-Detector for finding Mating-Events on microscopic pictures from mitochondria of yeast cells.

Based on:

[RetinaNet: Focal loss for Dense Object Detection](https://arxiv.org/abs/1708.02002)

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.

### Prerequisites

 * Python 3.3+
 * PyTorch 1.0.1+
 * Tensorflow 1.9
 * Torchvision 0.2.1 
 * TensorboardX
 * CUDA 10.0
 * Cell-Dataset: [download here]()

### Installing

Compile NMS:

```
cd lib/nms
export PATH=/usr/local/cuda-10.0/bin:$PATH
CUDAHOSTCXX=/usr/bin/gcc-6 python setup.py build_ext --inplace
```

Modify the 'coco_baseline' section in the Config-File: ```config.py```

When training on the yeast cell data:

 * Change the path for the basemodel
 * Change the path for the dataset
 * Change the path for the log dir (where to save the trained weights)
 * Change the path for the tb_dump_dir (where to save the path to initialize the Tensorboard)

When training on own data:

 * Change the path for the basemodel
 * Change the path for the dataset
 * Change the path for the log dir (where to save the trained weights)
 * Change the path for the tb_dump_dir (where to save the path to initialize the Tensorboard)
 * Change categories number and names
 * Adjust Anchor sizes (and so on)


## Running the training

Run from the command line with:

```
python train.py -d 2 -b 4 -e coco_baseline -ds COCO
```

### Running the prediction

Run from the command line after finishing the training:

```
python test.py -m log/coco_baseline/model_dump/epochxxx.pth -ds COCO -e coco_baseline
```

### Predict an image

Open the Jupyter Notebook:

```
'Retinanet_Visualize.ipynb'
```

and follow the instructions:

 - Change the path to the image and where to store the image
 - Run the Notebook


## Authors

* **Leonie Ganswindt** 


## Acknowledgments

* Used code: https://github.com/wondervictor/RetinaNet


