#  Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserve.
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

from __future__ import division
from __future__ import print_function

import paddle.fluid as fluid
from paddle.fluid.param_attr import ParamAttr
from paddle.fluid.initializer import Constant
from paddle.fluid.initializer import Normal
from paddle.fluid.regularizer import L2Decay

import box_utils
from config.config_parser import ConfigPaser
from config.config import cfg


def conv_bn_layer(input,
                  ch_out,
                  filter_size,
                  stride,
                  padding,
                  act=None,
                  bn=False,
                  name=None,
                  is_train=True):
    if bn:
        out = fluid.layers.conv2d(
            input=input,
            num_filters=ch_out,
            filter_size=filter_size,
            stride=stride,
            padding=padding,
            act=None,
            param_attr=ParamAttr(initializer=fluid.initializer.Normal(0., 0.02),
                                 name=name + "_weights"),
            bias_attr=False,
            name=name + '.conv2d.output.1')

        bn_name = "bn" + name[4:]

        out = fluid.layers.batch_norm(input=out, 
                                      act=None, 
                                      is_test=not is_train,
                                      param_attr=ParamAttr(
                                            initializer=fluid.initializer.Normal(0., 0.02),
                                            name=bn_name + '_scale'),
                                      bias_attr=ParamAttr(
                                            initializer=fluid.initializer.Constant(0.0),
                                            name=bn_name + '_offset'),
                                      moving_mean_name=bn_name+'_mean',
                                      moving_variance_name=bn_name+'_var',
                                      name=bn_name+'.output')
    else:
        out = fluid.layers.conv2d(
            input=input,
            num_filters=ch_out,
            filter_size=filter_size,
            stride=stride,
            padding=padding,
            act=None,
            param_attr=ParamAttr(initializer=fluid.initializer.Normal(0., 0.02),
                                 name=name + "_weights"),
            bias_attr=ParamAttr(initializer=fluid.initializer.Constant(0.0),
                                 name=name + "_bias"),
            name=name + '.conv2d.output.1')

    if act == 'relu':
        out = fluid.layers.relu(x=out)
    if act == 'leaky':
        out = fluid.layers.leaky_relu(x=out, alpha=0.1)
    return out


class YOLOv3(object):
    def __init__(self, 
                model_cfg_path,
                is_train=True,
                use_pyreader=True,
                use_random=True):
        self.model_cfg_path = model_cfg_path
        self.config_parser = ConfigPaser(model_cfg_path)
        self.is_train = is_train
        self.use_pyreader = use_pyreader
        self.use_random = use_random
        self.outputs = []
        self.losses = []

    def build_model(self):
        model_defs = self.config_parser.parse()
        if model_defs is None:
            return None

        self.hyperparams = model_defs.pop(0)
        assert self.hyperparams['type'].lower() == "net", \
                "net config params should be given in the first segment named 'net'"
        self.img_height = self.hyperparams['height']
        self.img_width = self.hyperparams['width']

        self.build_input()

        out = self.image
        layer_outputs = []
        self.yolo_layer_defs = []
        self.yolo_anchors = []
        self.yolo_classes = []
        self.outputs = []
        for i, layer_def in enumerate(model_defs):
            if layer_def['type'] == 'convolutional':
                bn = layer_def.get('batch_normalize', 0)
                ch_out = int(layer_def['filters'])
                filter_size = int(layer_def['size'])
                stride = int(layer_def['stride'])
                padding = (filter_size - 1) // 2 if int(layer_def['pad']) else 0
                act = layer_def['activation']
                out = conv_bn_layer(
                        input=out,
                        ch_out=ch_out,
                        filter_size=filter_size,
                        stride=stride,
                        padding=padding,
                        act=act,
                        bn=bool(bn),
                        name="conv"+str(i),
                        is_train=self.is_train)

            elif layer_def['type'] == 'shortcut':
                layer_from = int(layer_def['from'])
                out = fluid.layers.elementwise_add(
                        x=out, 
                        y=layer_outputs[layer_from],
                        name="res"+str(i))

            elif layer_def['type'] == 'route':
                layers = map(int, layer_def['layers'].split(","))
                out = fluid.layers.concat(
                        input=[layer_outputs[i] for i in layers],
                        axis=1)

            elif layer_def['type'] == 'upsample':
                scale = float(layer_def['stride'])
                out = fluid.layers.resize_nearest(
                        input=out,
                        scale=scale,
                        name="upsample"+str(i))

            elif layer_def['type'] == 'maxpool':
                pool_size = int(layer_def['size'])
                pool_stride = int(layer_def['stride'])
                pool_padding = 0
                if pool_stride == 1 and pool_size == 2:
                    pool_padding = 1
                out = fluid.layers.pool2d(
                        input=out,
                        pool_type='max',
                        pool_size=pool_size,
                        pool_stride=pool_stride,
                        pool_padding=pool_padding)

            elif layer_def['type'] == 'yolo':
                self.yolo_layer_defs.append(layer_def)
                self.outputs.append(out)
                out.persistable = True

                anchor_idxs = map(int, layer_def['mask'].split(','))
                all_anchors = map(int, layer_def['anchors'].split(','))
                anchors = []
                for anchor_idx in anchor_idxs:
                    anchors.append(all_anchors[anchor_idx * 2])
                    anchors.append(all_anchors[anchor_idx * 2 + 1])
                self.yolo_anchors.append(anchors)
                class_num = int(layer_def['classes'])
                self.yolo_classes.append(class_num)

                if self.is_train:
                    ignore_thresh = float(layer_def['ignore_thresh'])
                    loss = fluid.layers.yolov3_loss(
                            x=out,
                            gtbox=self.gtbox,
                            gtlabel=self.gtlabel,
                            anchors=anchors,
                            class_num=class_num,
                            ignore_thresh=ignore_thresh,
                            input_size=int(self.hyperparams['height']),
                            name="yolo_loss"+str(i))
                    self.losses.append(fluid.layers.reduce_mean(loss))

            layer_outputs.append(out)

    def loss(self):
        return sum(self.losses)

    def get_pred(self):
        return self.outputs
    
    def get_yolo_anchors(self):
        return self.yolo_anchors

    def get_yolo_classes(self):
        return self.yolo_classes

    def build_input(self):
        self.image_shape = (3, int(self.hyperparams['height']), int(self.hyperparams['width']))
        self.image = fluid.layers.data(
                name='image', shape=self.image_shape, dtype='float32'
                )
        self.gtbox = fluid.layers.data(
                # name='gtbox', shape=[cfg.TRAIN.max_box_num, 4], dtype='float32', lod_level=1
                name='gtbox', shape=[10, 4], dtype='float32'
                )
        self.gtlabel = fluid.layers.data(
                name='gtlabel', shape=[10], dtype='int32'
                )
        self.im_shape = fluid.layers.data(
                name="im_shape", shape=[2], dtype='int32')
        self.im_id = fluid.layers.data(
                name="im_id", shape=[1], dtype='int32')
    
    def feeds(self):
        if not self.is_train:
            return [self.image, self.im_id, self.im_shape]
        return [self.image, self.gtbox, self.gtlabel]

    def get_hyperparams(self):
        return self.hyperparams

    def get_input_size(self):
        return int(self.hyperparams['height'])

        
