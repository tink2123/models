# copyright (c) 2019 PaddlePaddle Authors. All Rights Reserve.
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
Contains PointNet++ MSG classification models
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

import paddle.fluid as fluid
from paddle.fluid.param_attr import ParamAttr
from paddle.fluid.initializer import Constant
from .pointnet2_modules_dy import *
#from pointnet2_modules_dy import *
from paddle.fluid.dygraph.base import to_variable

__all__ = ["PointNet2ClsMSG_dy", "PointNet2ClsSSG_dy"]


class PointNet2ClsMSG_dy(fluid.dygraph.Layer):
    def __init__(self, name_scope, num_classes, num_points, use_xyz=True):
        super(PointNet2ClsMSG_dy,self).__init__(name_scope)
        self.num_classes = num_classes
        self.num_points = num_points
        self.use_xyz = use_xyz
        self.sa_module_msg_0 = Pointnet_SA_Module_MSG(self.full_name(),
                                                  npoint=512,
                                                  radiuss=[0.1,0.2,0.4],
                                                  nsamples=[16,32,128],
                                                  mlps=[[32,32,64],[64,64,128],[64,96,128]])
        self.sa_module_msg_1 = Pointnet_SA_Module_MSG(self.full_name(),
                                                      npoint=128,
                                                      radiuss=[0.2,0.4,0.8],
                                                      nsamples=[32,64,128],
                                                      mlps=[[64,64,128],[128,128,256],[128,128,256]])
        self.sa_module = Pointnet_SA_Module_MSG(self.full_name(),radiuss=[None],nsamples=[None],mlps=[[256,512,1024]])
        self.fc_0 = FCBN(self.full_name(),out_channel=512,bn=True)
        self.fc_1 = FCBN(self.full_name(),out_channel=256,bn=True)
        self.fc_2 = FCBN(self.full_name(),out_channel=self.num_classes,bn=False,act=None)
    def forward(self, xyz,label):
        xyz,feature = self.sa_module_msg_0(xyz,feature=None)
        feature = fluid.layers.transpose(feature,perm=[0,2,1])
        # to fix transpose
        tmp = feature+0.0
        xyz,feature = self.sa_module_msg_1(xyz,feature=tmp)
        feature = fluid.layers.transpose(feature,perm=[0,2,1])
        xyz,feature = self.sa_module(xyz,feature=feature)
        out = squeeze(feature,axis=-1)
        # FC layer
        out = self.fc_0(out)
        out = fluid.layers.dropout(out,0.5,dropout_implementation="upscale_in_train")
        out = self.fc_1(out)
        out = fluid.layers.dropout(out,0.5,dropout_implementation="upscale_in_train")
        out = self.fc_2(out)

        pred = fluid.layers.softmax(out)

        loss = fluid.layers.cross_entropy(pred,label)
        loss = fluid.layers.reduce_mean(loss)

        # calc acc
        pred = fluid.layers.reshape(out,shape=[-1,self.num_classes])
        label = fluid.layers.reshape(label,shape=[-1,1])
        acc1 = fluid.layers.accuracy(pred,label,k=1)
        #acc2 = fluid.layers.accuracy(pred,label,k=5)
        return out,loss,acc1

    def set_bn_momentum(self, bn_momentum):
        self.sa_module_msg_0.set_bn_momentum(bn_momentum)
        self.sa_module_msg_1.set_bn_momentum(bn_momentum)
        self.sa_module.set_bn_momentum(bn_momentum)
    
        self.fc_0.set_bn_momentum(bn_momentum)
        self.fc_1.set_bn_momentum(bn_momentum)
        self.fc_2.set_bn_momentum(bn_momentum)


class PointNet2ClsSSG_dy(fluid.dygraph.Layer):
    def __init__(self, name_scope, num_classes, num_points, use_xyz=True):
        super(PointNet2ClsMSG_dy,self).__init__(name_scope)
        self.num_classes = num_classes
        self.num_points = num_points
        self.use_xyz = use_xyz
        self.sa_module_msg_0 = Pointnet_SA_Module_MSG(self.full_name(),
                                                  npoint=512,
                                                  radiuss=[0.2],
                                                  nsamples=[64],
                                                  mlps=[[64,64,128]])
        self.sa_module_msg_1 = Pointnet_SA_Module_MSG(self.full_name(),
                                                      npoint=128,
                                                      radiuss=[0.4],
                                                      nsamples=[64],
                                                      mlps=[[128,128,256]])
        self.sa_module = Pointnet_SA_Module_MSG(self.full_name(),radiuss=[None],nsamples=[None],mlps=[[256,512,1024]])
        self.fc_0 = FCBN(self.full_name(),out_channel=512,bn=True)
        self.fc_1 = FCBN(self.full_name(),out_channel=256,bn=True)
        self.fc_2 = FCBN(self.full_name(),out_channel=self.num_classes,bn=False,act=None)
    def forward(self, xyz,label):
        xyz,feature = self.sa_module_msg_0(xyz,feature=None)
        feature = fluid.layers.transpose(feature,perm=[0,2,1])
        # to fix transpose
        tmp = feature+0.0
        xyz,feature = self.sa_module_msg_1(xyz,feature=tmp)
        feature = fluid.layers.transpose(feature,perm=[0,2,1])
        xyz,feature = self.sa_module(xyz,feature=feature)
        out = squeeze(feature,axis=-1)
        # FC layer
        out = self.fc_0(out)
        out = fluid.layers.dropout(out,0.5,dropout_implementation="upscale_in_train")
        out = self.fc_1(out)
        out = fluid.layers.dropout(out,0.5,dropout_implementation="upscale_in_train")
        out = self.fc_2(out)

        pred = fluid.layers.softmax(out)

        loss = fluid.layers.cross_entropy(pred,label)
        loss = fluid.layers.reduce_mean(loss)

        # calc acc
        pred = fluid.layers.reshape(out,shape=[-1,self.num_classes])
        label = fluid.layers.reshape(label,shape=[-1,1])
        acc1 = fluid.layers.accuracy(pred,label,k=1)
        #acc2 = fluid.layers.accuracy(pred,label,k=5)
        return out,loss,acc1

    def set_bn_momentum(self, bn_momentum):
        self.sa_module_msg_0.set_bn_momentum(bn_momentum)
        self.sa_module_msg_1.set_bn_momentum(bn_momentum)
        self.sa_module.set_bn_momentum(bn_momentum)
    
        self.fc_0.set_bn_momentum(bn_momentum)
        self.fc_1.set_bn_momentum(bn_momentum)
        self.fc_2.set_bn_momentum(bn_momentum)


if __name__ == "__main__":
    num_classes = 13
    num_points = 32

    with fluid.dygraph.guard():
        pointnet_cls = PointNet2ClsMSG_dy("pointnet2_cls",num_classes=13, num_points=32)
        np.random.seed(1333)
        xyz_np = np.random.uniform(-100, 100, (8, 32, 3)).astype('float32')
        label_np = np.random.uniform(0, num_classes, (8, 1)).astype('int64')
        print("lable:",label_np)
        xyz = to_variable(xyz_np)
        label = to_variable(label_np)
        label.stop_gradient = True
        opt = fluid.optimizer.AdamOptimizer(learning_rate=3e-2)
        for i in range(10):
            out,loss,_ = pointnet_cls(xyz,label)
            loss.backward()
            opt.minimize(loss)
            #loss.clear_gradients()
            print("loss:",loss.numpy())
