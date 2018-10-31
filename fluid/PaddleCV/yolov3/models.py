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


def conv_affine_layer(input,
                      ch_out,
                      filter_size,
                      stride,
                      padding,
                      act='relu',
                      bn=False
                      name=None):
    out = fluid.layers.conv2d(
        input=input,
        num_filters=ch_out,
        filter_size=filter_size,
        stride=stride,
        padding=padding,
        act=None,
        param_attr=ParamAttr(name=name + "_weights"),
        bias_attr=False,
        name=name + '.conv2d.output.1')

    if bn:
        bn_name = "bn" + name[3:]
        scale = fluid.layers.create_parameter(
            shape=[out.shape[1]],
            dtype=out.dtype,
            attr=ParamAttr(
                name=bn_name + '_scale', learning_rate=0.),
            default_initializer=Constant(1.))
        scale.stop_gradient = True
        bias = fluid.layers.create_parameter(
            shape=[out.shape[1]],
            dtype=out.dtype,
            attr=ParamAttr(
                bn_name + '_offset', learning_rate=0.),
            default_initializer=Constant(0.))
        bias.stop_gradient = True

        out = fluid.layers.affine_channel(x=out, scale=scale, bias=bias)

    if act == 'relu':
        out = fluid.layers.relu(x=out)
    if act == 'leaky':
        out = fluid.layers.leaky_relu(x=out)
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

    def build_model(self, self.image):
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
        self.outputs = []
        self.losses = []
        for i, layer_def in enumerate(model_defs):
            if layer_def['type'] == 'convolutional':
                bn = int(layer_def['batch_normalize'])
                ch_out = int(layer_def['filters'])
                filter_size = int(layer_def['size'])
                stride = int(layer_def['stride'])
                padding = (size - 1) // 2 if int(layer_def['pad']) else 0
                act = layer_def['activation']
                out = conv_affine_layer(
                        ch_out=ch_out,
                        filter_size=filter_size
                        stride=stride
                        padding=padding
                        act=act,
                        bn=bool(bn)
                        name="conv"+str(i))

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

                if is_train:
                    anchor_idxs = map(int, layer_def['mask'].split(','))
                    all_anchors = map(float, layer_def['anchors'].split(','))
                    anchors = []
                    for anchor_idx in anchor_idxs:
                        anchors.append(all_anchors[anchor_idx * 2])
                        anchors.append(all_anchors[anchor_idx * 2 + 1])
                    class_num = layer_def['class_num']
                    ignore_thresh = layer_def['ignore_thresh']
                    loss = fluid.layers.yolov3_loss(
                            x=out,
                            gtbox=self.gtbox,
                            gtlabel=self.gtlabel,
                            anchors=anchors,
                            class_num=class_num,
                            ignore_thresh=ignore_thresh
                            name="yolo_loss"+str(i))
                    self.losses.append(loss)

            layer_outputs.append(out)

    def loss(self):
        return sum(self.losses)

    def get_pred(self):
        all_pred_boxes = []
        all_pred_confs = []
        all_pred_labels = []
        for layer_def, output in zip(self.yolo_layer_defs, self.outputs):
            class_num = layer_def['class_num']
            all_anchors = map(float, layer_def['anchors'].split(','))
            anchor_idxs = map(int, layer_def['mask'].split(','))
            anchors = [[] for _ in range(len(anchor_idxs))]
            for i, anchor_idx in enumerate(anchor_idxs):
                anchors[i].append(all_anchors[anchor_idx * 2])
                anchors[i].append(all_anchors[anchor_idx * 2 + 1])
            pred_boxes, pred_confs, pred_labels = box_utils.get_yolo_detection(output, anchors, class_num, self.img_width, self.img_height)
            all_pred_boxes.append(pred_boxes)
            all_pred_confs.append(pred_confs)
            all_pred_labels.append(pred_labels)

        return (
            fluid.layers.concat(all_pred_boxes, axis=1)
            fluid.layers.concat(all_pred_confs, axis=1)
            fluid.layers.concat(all_pred_labels, axis=1)
            )

    def build_input(self):
        self.image_shape = (3, self.hyperparams['height'], self.hyperparams['width'])
        self.image = fluid.layers.data(
                name='image', shape=self.image_shape, dtype='float32'
                )
        self.gtbox = fluid.layers.data(
                name='gtbox', shape=[5], dtype='float32', lod_level=1
                )
        self.gtlabel = fluid.layers.data(
                name='gtlabel', shape=[-1], dtype='int32', lod_level=1
                )
    
    def feeds(self):
        if not self.is_train:
            return [self.image]
        return [self.image, self.gtbox, self.gtlabel]
    
    def get_hyperparams(self):
        return self.hyperparams

    def get_input_size(self):
        return int(self.hyperparams['height'])

        
