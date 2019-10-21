#  Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserve.
#
#Licensed under the Apache License, Version 2.0 (the "License");
#you may not use this file except in compliance with the License.
#You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
#Unless required by applicable law or agreed to in writing, software
#distributed under the License is distributed on an "AS IS" BASIS,
#WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#See the License for the specific language governing permissions and
#limitations under the License.
"""
Contains PointNet++ SSG/MSG semantic segmentation models
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

import paddle.fluid as fluid
from paddle.fluid.param_attr import ParamAttr
from paddle.fluid.initializer import Constant
from .pointnet2_modules import *
# from pointnet2_modules import *

__all__ = ["PointNet2SemSegSSG", "PointNet2SemSegMSG"]


class PointNet2SemSeg(object):
    def __init__(self, num_classes, num_points, use_xyz=True):
        self.num_classes = num_classes
        self.num_points = num_points
        self.use_xyz = use_xyz
        self.feed_vars = []
        self.out_feature = None
        self.pyreader = None
        self.model_config()

    def model_config(self):
        self.SA_confs = []
        self.FP_confs = []

    def build_input(self):
        self.xyz = fluid.layers.data(name='xyz', shape=[self.num_points, 3], dtype='float32', lod_level=0)
        self.feature = fluid.layers.data(name='feature', shape=[self.num_points, 6], dtype='float32', lod_level=0)
        self.label = fluid.layers.data(name='label', shape=[self.num_points, 1], dtype='int64', lod_level=0)
        self.pyreader = fluid.io.PyReader(
                feed_list=[self.xyz, self.feature, self.label],
                capacity=64,
                use_double_buffer=True,
                iterable=False)
        self.feed_vars = [self.xyz, self.feature, self.label]

    def build_model(self, bn_momentum=0.9):
        self.build_input()

        xyzs, features = [self.xyz], [self.feature]
        for i, SA_conf in enumerate(self.SA_confs):
            xyzi, featurei = pointnet_sa_module(
                    xyz=xyzs[i],
                    feature=features[i],
                    bn_momentum=bn_momentum,
                    use_xyz=self.use_xyz,
                    name="sa_{}".format(i),
                    **SA_conf)
            xyzs.append(xyzi)
            features.append(featurei)
        for i in range(-1, -(len(self.FP_confs) + 1), -1):
            features[i - 1] = pointnet_fp_module(
                    unknown=xyzs[i - 1],
                    known=xyzs[i],
                    unknown_feats=features[i - 1],
                    known_feats=features[i],
                    bn_momentum=bn_momentum,
                    name="fp_{}".format(i),
                    **self.FP_confs[i])

        out = fluid.layers.transpose(features[0], perm=[0, 2, 1])
        out = fluid.layers.unsqueeze(out, axes=[-1])
        out = conv_bn(out, out_channels=128, bn=True, bn_momentum=bn_momentum, name="output_1")
        # out = fluid.layers.dropout(out, 0.5)
        out = conv_bn(out, out_channels=self.num_classes, bn=False, act=None, name="output_2")
        print(out)
        out = fluid.layers.squeeze(out, axes=[-1])
        out = fluid.layers.transpose(out, perm=[0, 2, 1])
        pred = fluid.layers.softmax(out)

        # calc loss
        self.loss = fluid.layers.cross_entropy(pred, self.label)
        self.loss = fluid.layers.reduce_mean(self.loss)

        # calc acc
        pred = fluid.layers.reshape(pred, shape=[-1, self.num_classes])
        label = fluid.layers.reshape(self.label, shape=[-1, 1])
        self.acc1 = fluid.layers.accuracy(pred, label, k=1)

    def get_feeds(self):
	return self.feed_vars

    def get_outputs(self):
        return {"loss": self.loss, "accuracy": self.acc1}

    def get_pyreader(self):
        return self.pyreader


class PointNet2SemSegSSG(PointNet2SemSeg):
    def __init__(self, num_classes, use_xyz=True):
        super(PointNet2SemSegSSG, self).__init__(num_classes, use_xyz)

    def model_config(self):
        self.SA_confs = [
            {
                "npoint": 1024,
                "radiuss": [0.1],
                "nsamples": [32],
                "mlps": [[32, 32, 64]],
            },
            {
                "npoint": 256,
                "radiuss": [0.2],
                "nsamples": [32],
                "mlps": [[64, 64, 128]],
            },
            {
                "npoint": 64,
                "radiuss": [0.4],
                "nsamples": [32],
                "mlps": [[128, 128, 256]],
            },
            {
                "npoint": 16,
                "radiuss": [0.8],
                "nsamples": [32],
                "mlps": [[256, 256, 512]],
            },
        ]

        self.FP_confs = [
            {"mlp": [128, 128, 128]},
            {"mlp": [256, 128]},
            {"mlp": [256, 256]},
            {"mlp": [256, 256]},
        ]


class PointNet2SemSegMSG(PointNet2SemSeg):
    def __init__(self, num_classes, use_xyz=True):
        super(PointNet2SemSegMSG, self).__init__(num_classes, use_xyz)

    def model_config(self):
        self.SA_confs = [
            {
                "npoint": 1024,
                "radiuss": [0.05, 0.1],
                "nsamples": [16, 32],
                "mlps": [[16, 16, 32], [32, 32, 64]],
            },
            {
                "npoint": 256,
                "radiuss": [0.1, 0.2],
                "nsamples": [16, 32],
                "mlps": [[64, 64, 128], [64, 96, 128]],
            },
            {
                "npoint": 64,
                "radiuss": [0.2, 0.4],
                "nsamples": [16, 32],
                "mlps": [[128, 196, 256], [128, 196, 256]],
            },
            {
                "npoint": 16,
                "radiuss": [0.4, 0.8],
                "nsamples": [16, 32],
                "mlps": [[256, 256, 512], [256, 384, 512]],
            },
        ]

        self.FP_confs = [
            {"mlp": [128, 128]},
            {"mlp": [256, 256]},
            {"mlp": [512, 512]},
            {"mlp": [512, 512]},
        ]


if __name__ == "__main__":
    num_classes = 13
    num_points = 32
    # model = PointNet2SemSegSSG(num_classes)
    model = PointNet2SemSegMSG(num_classes, num_points)
    model.build_model()
    outs = model.get_outputs()
    print(outs)

    place = fluid.CUDAPlace(0)
    exe = fluid.Executor(place)
    exe.run(fluid.default_startup_program())

    np.random.seed(2333)
    xyz_np = np.random.uniform(-100, 100, (8, 32, 3)).astype('float32')
    feature_np = np.random.uniform(-100, 100, (8, 32, 6)).astype('float32')
    label_np = np.random.uniform(0, num_classes, (8, 32, 1)).astype('int64')
    # print("xyz", xyz_np)
    # print("feaure", feature_np)
    # print("label", label_np)
    # ret = exe.run(fetch_list=[out.name for out in outs], feed={'xyz': xyz_np, 'feature': feature_np, 'label': label_np})
    for param in model.parameters():
        print(param.name)
    ret = exe.run(fetch_list=["conv2d_33.tmp_1", outs['loss'].name, outs['accuracy'].name], feed={'xyz': xyz_np, 'feature': feature_np, 'label': label_np})
    print("ret0", ret[0].shape, ret[0])
    print("ret1-", ret[1:])
    # ret[0].tofile("out.data")
