# A fork of PointConv for tensorflow 2.8 and CUDA 11.2

## What are the differences from [dgriffiths3](https://github.com/dgriffiths3/pointconv-tensorflow2) repo?

1. This implementation works with tensorlfow 2.8 and cuda 11.2 (Nvidia 3080 GPUs only support CUDA >= 11.2).
2. I have also added a simpler import method: `from pointconv import PointConvSA`.

## Setup

1. Clone this repository
2. `cd` into pointconv-tensorflow2
3. Compile ops by running `./tf_ops/compile_ops.sh PYTHON_INTERPRETER_PATH` (ensure the `CUDA_ROOT` points correctly to cuda 11.2)
4. Install pointconv package with `python -m pip install -e .` 

_____
This repository containts implementations of the PointConv (Wu et al, 2019) feature encoder and feature decoder layers as `tf.keras.layers` classes. This allows for PointConv layers to be used as part of the standard `tf.keras` api. The repository does not aim to be an exact implementation of the original repostiroy, rather a useful tool for building custom models or simple backend encoders for unordered point sets. For more details regarding the technical details check out the [original paper](https://arxiv.org/abs/1811.07246) and [github page](https://github.com/DylanWusee/pointconv). The implementation also matches the style of the [PointNet++ keras layers](https://github.com/dgriffiths3/pointnet2-tensorflow2).

## Usage

The layers follow the standard `tf.keras.layers` api. To import in your own project, copy the `pointconv` and `tf_ops` folders and set a relative path to find the layers. Here is an example of how a simple PointConv SetAbstraction model can be built using `tf.keras.Model()`.

```
from tensorflow import keras
from pointconv import PointConvSA

class MyModel(keras.Model):

  def __init__(self, batch_size):
    super(MyModel, self).__init__()

        self.layer1 = PointConvSA(npoint=512, radius=0.1, sigma=0.1, K=32, mlp=[64, 64, 128], bn=True)
        self.layer2 = PointConvSA(npoint=128, radius=0.2, sigma=0.2, K=32, mlp=[128, 128, 256], bn=True)
        self.layer2 = PointConvSA(npoint=1, radius=0.8, sigma=0.4, K=32, mlp=[256, 512, 1024], group_all=True bn=True)

        # To make a classifier, just add some fully-connected layers

        self.fn1 = keras.layers.Dense(512)
        self.fn2 = keras.layers.Dense(256)
        self.fn3 = keras.layers.Dense(n_classes, tf.nn.softmax)
    
  def call(input):

    xyz, points = self.layer1(input, None, training=training)
    xyz, points = self.layer2(xyz, points, training=training)
    xyz, points = self.layer3(xyz, points, training=training)

    net = tf.reshape(points, (self.batch_size, -1))

    net = self.dense1(net)
    net = self.dense2(net)
    pred = self.dense3(net)

    return pred
```

## Examples

Refer to [dgriffiths3](https://github.com/dgriffiths3/pointconv-tensorflow2) repository for use cases.

## Note

If you use these layers in your project remember to cite the original authors:

```
@article{wu2018pointconv,
  title={PointConv: Deep Convolutional Networks on 3D Point Clouds},
  author={Wu, Wenxuan and Qi, Zhongang and Fuxin, Li},
  journal={arXiv preprint arXiv:1811.07246},
  year={2018}
}
```
