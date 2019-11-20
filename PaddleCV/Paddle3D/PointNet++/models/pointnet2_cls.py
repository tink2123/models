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
Contains PointNet++ classification models
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

import paddle.fluid as fluid
from paddle.fluid.param_attr import ParamAttr
from paddle.fluid.initializer import Constant
from .pointnet2_modules import *
#from pointnet2_modules import *
__all__ = ["PointNet2ClsSSG", "PointNet2ClsMSG"]


class PointNet2Cls(object):
    def __init__(self, num_classes, num_points, use_xyz=True):
        self.num_classes = num_classes
        self.num_points = num_points
        self.use_xyz = use_xyz
        self.out_feature = None
        self.pyreader = None
        self.model_config()

    def model_config(self):
        self.SA_confs = []

    def build_input(self):
        self.xyz = fluid.layers.data(name='xyz', shape=[self.num_points, 3], dtype='float32', lod_level=0)
        self.label = fluid.layers.data(name='label', shape=[1], dtype='int64', lod_level=0)
        #self.pyreader = fluid.io.PyReader(
        #        feed_list=[self.xyz, self.label],
        #        capacity=64,
        #        use_double_buffer=True,
        #        iterable=False)
        #self.feed_vars = [self.xyz, self.label]

    def build_model(self, bn_momentum=0.99):
        self.build_input()

        xyz, feature = self.xyz, None
        #print("feature:",feature)
        for i, SA_conf in enumerate(self.SA_confs):
            xyz, feature = pointnet_sa_module(
                    xyz=xyz,
                    feature=feature,
                    bn_momentum=bn_momentum,
                    use_xyz=self.use_xyz,
                    name="sa_{}".format(i),
                    **SA_conf)
        out = fluid.layers.transpose(feature, perm=[0, 2, 1])
        out = fluid.layers.squeeze(out,axes=[-1])
        #print("after sa module:",out.shape)
	out = fc_bn(out,out_channels=512, bn=True, bn_momentum=bn_momentum, name="fc_1")
        out = fluid.layers.dropout(out, 0.5, dropout_implementation="upscale_in_train")
        out = fc_bn(out,out_channels=256, bn=True, bn_momentum=bn_momentum, name="fc_2")
        out = fluid.layers.dropout(out, 0.5, dropout_implementation="upscale_in_train")
        out = fc_bn(out,out_channels=self.num_classes, act=None, name="fc_3")

        out = fluid.layers.softmax(out)

        # calc loss softmax
        self.loss = fluid.layers.cross_entropy(out, self.label)
        self.loss = fluid.layers.reduce_mean(self.loss)


        # sigmoid loss
	#label_onehot = fluid.layers.one_hot(self.label,depth=self.num_classes)
	#label_float = fluid.layers.cast(label_onehot,dtype="float32")
	#self.loss = fluid.layers.sigmoid_cross_entropy_with_logits(out,label_float)
        #self.loss = fluid.layers.reduce_mean(self.loss)

        # calc acc
        pred = fluid.layers.reshape(out, shape=[-1, self.num_classes])
        label = fluid.layers.reshape(self.label, shape=[-1, 1])
        self.acc1 = fluid.layers.accuracy(pred, label, k=1)

    def get_feeds(self):
        return self.feed_vars

    def get_outputs(self):
        return self.loss, self.acc1
        #return {"loss": self.loss, "accuracy": self.acc1}

    def get_pyreader(self):
        return self.pyreader


class PointNet2ClsSSG(PointNet2Cls):
    def __init__(self, num_classes, num_points, use_xyz=True):
        super(PointNet2ClsSSG, self).__init__(num_classes, num_points, use_xyz)

    def model_config(self):
        self.SA_confs = [
            {
                "npoint": 512,
                "radiuss": [0.2],
                "nsamples": [64],
                "mlps": [[64, 64, 128]],
            },
            {
                "npoint": 128,
                "radiuss": [0.4],
                "nsamples": [64],
                "mlps": [[128, 128, 256]],
            },
            {
                "npoint":None,
		"radiuss": [None],
		"nsamples":[None],
		"mlps": [[256, 512, 1024]],
            },
        ]


class PointNet2ClsMSG(PointNet2Cls):
    def __init__(self, num_classes, num_points, use_xyz=True):
        super(PointNet2ClsMSG, self).__init__(num_classes, num_points, use_xyz)

    def model_config(self):
        self.SA_confs = [
            {
                "npoint": 512,
                "radiuss": [0.1, 0.2, 0.4],
                "nsamples": [16, 32, 128],
                "mlps": [[32, 32, 64],
                         [64, 64, 128],
                         [64,96,128]],
            },
            {
                "npoint": 128,
                "radiuss": [0.2, 0.4, 0.8],
                "nsamples": [32, 64, 128],
                "mlps": [[64, 64, 128],
                         [128, 128, 256],
                         [128,128,256]],
            },
            {
                "npoint":None,
		"radiuss": [None],
		"nsamples":[None],
		"mlps": [[256, 512, 1024]],
            },
        ]

if __name__ == "__main__":
    num_classes = 13
    num_points = 32
    seed = 1333
    model = PointNet2ClsMSG(num_classes,num_points)
    model.build_model()
    loss,acc = model.get_outputs()
    opt = fluid.optimizer.AdamOptimizer(learning_rate=3e-2)
    #opt = fluid.optimizer.SGD(learning_rate=3e-2)
    opt.minimize(loss)

    place = fluid.CUDAPlace(0)
    exe = fluid.Executor(place)
    fluid.default_startup_program().random_seed = seed
    fluid.default_main_program().random_seed = seed
    exe.run(fluid.default_startup_program())
    # print param.name
    #for i,var in enumerate(fluid.default_startup_program().list_vars()):
    #    print(i,var.name)

    np.random.seed(1333)
    xyz_np = np.random.uniform(-100, 100, (8, 32, 3)).astype('float32')
    #feature_np = np.random.uniform(-100, 100, (8, 32, 6)).astype('float32')
    label_np = np.random.uniform(0, num_classes, (8, 1)).astype('int64')
    #print("xyz", xyz_np)
    #print("feaure", feature_np)
    print("label", label_np)
    for i in range(10):
        ret = exe.run(fetch_list=[loss.name], feed={'xyz': xyz_np,'label': label_np})
        #print("batch_norm_22.w_0:",ret[0])
        #print("fc weight:",ret[1])
        print("loss:",ret[-1])
        #print("output:",np.sum(ret[0]))
