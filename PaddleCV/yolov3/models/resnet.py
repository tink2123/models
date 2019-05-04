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

import paddle.fluid as fluid
from paddle.fluid.param_attr import ParamAttr
from paddle.fluid.initializer import Constant
from paddle.fluid.regularizer import L2Decay
from config import cfg


def conv_bn_layer(input,
                  ch_out,
                  filter_size,
                  stride,
                  padding,
                  act='relu',
                  is_test=True,
                  name=None):
    conv1 = fluid.layers.conv2d(
        input=input,
        num_filters=ch_out,
        filter_size=filter_size,
        stride=stride,
        padding=padding,
        act=None,
        param_attr=ParamAttr(name=name + "_weights"),
        bias_attr=False,
        name=name + '.conv2d.output.1')
    if name == "conv1":
        bn_name = "bn_" + name
    else:
        bn_name = "bn" + name[3:]

    return fluid.layers.batch_norm(
        input=conv1,
        act=act,
        name=bn_name + '.output.1',
        param_attr=ParamAttr(name=bn_name + '_scale'),
        bias_attr=ParamAttr(bn_name + '_offset'),
        moving_mean_name=bn_name + '_mean',
        moving_variance_name=bn_name + '_variance',
        is_test=is_test)


def conv_affine_layer(input,
                      ch_out,
                      filter_size,
                      stride,
                      padding,
                      act='relu',
                      name=None):
    conv = fluid.layers.conv2d(
        input=input,
        num_filters=ch_out,
        filter_size=filter_size,
        stride=stride,
        padding=padding,
        act=None,
        param_attr=ParamAttr(name=name + "_weights"),
        bias_attr=False,
        name=name + '.conv2d.output.1')
    if name == "conv1":
        bn_name = "bn_" + name
    else:
        bn_name = "bn" + name[3:]

    scale = fluid.layers.create_parameter(
        shape=[conv.shape[1]],
        dtype=conv.dtype,
        attr=ParamAttr(
            name=bn_name + '_scale', learning_rate=0.),
        default_initializer=Constant(1.))
    scale.stop_gradient = True
    bias = fluid.layers.create_parameter(
        shape=[conv.shape[1]],
        dtype=conv.dtype,
        attr=ParamAttr(
            bn_name + '_offset', learning_rate=0.),
        default_initializer=Constant(0.))
    bias.stop_gradient = True

    out = fluid.layers.affine_channel(x=conv, scale=scale, bias=bias)
    if act == 'relu':
        out = fluid.layers.relu(x=out)
    return out


def shortcut(input, ch_out, stride, is_first, is_test=True,name=None):
    ch_in = input.shape[1]  # if args.data_format == 'NCHW' else input.shape[-1]
    if ch_in != ch_out or is_first == True:
        return conv_bn_layer(input, ch_out, 1, stride, 0, None, is_test=is_test, name=name)
    else:
        return input


def basicblock(input, ch_out, stride, is_first, is_test=True, name=None):
    short = shortcut(input, ch_out, stride, is_first=is_first, is_test=is_test, name=name+"_branch1")
    conv1 = conv_bn_layer(input, ch_out, 3, stride, 1, is_test=is_test, name=name+"_branch2a")
    conv2 = conv_bn_layer(conv1, ch_out, 3, 1, 1, act=None, is_test=is_test, name=name+"_branch2b")
    return fluid.layers.elementwise_add(x=short, y=conv2, act='relu', name=name+"_add")


def bottleneck(input, ch_out, stride, name):
    short = shortcut(input, ch_out * 4, stride, name=name + "_branch1")
    conv1 = conv_affine_layer(
        input, ch_out, 1, stride, 0, name=name + "_branch2a")
    conv2 = conv_affine_layer(conv1, ch_out, 3, 1, 1, name=name + "_branch2b")
    conv3 = conv_affine_layer(
        conv2, ch_out * 4, 1, 1, 0, act=None, name=name + "_branch2c")
    return fluid.layers.elementwise_add(
        x=short, y=conv3, act='relu', name=name + ".add.output.5")


def layer_warp(block_func, input, ch_out, count, stride, is_first, is_test=True, name=None):
    res_out = block_func(input, ch_out, stride, is_first=is_first, is_test=is_test, name=name + "a")
    for i in range(1, count):
        res_out = block_func(res_out, ch_out, 1, is_first=False, is_test=is_test, name=name + chr(ord("a") + i))
    return res_out


ResNet_cfg = {
    18: ([2, 2, 2, 2], basicblock),
    34: ([3, 4, 6, 3], basicblock),
    50: ([3, 4, 6, 3], bottleneck),
    101: ([3, 4, 23, 3], bottleneck),
    152: ([3, 8, 36, 3], bottleneck)
}

def add_ResNet_conv5_body(body_input,num_layers=34,is_test=True):
    stages, block_func = ResNet_cfg[num_layers]
    stages = stages[0:4]
    conv1 = conv_bn_layer(
        body_input, ch_out=64, filter_size=7, stride=2, padding=3, is_test=is_test, name="conv1")
    pool1 = fluid.layers.pool2d(
        input=conv1,
        pool_type='max',
        pool_size=3,
        pool_stride=2,
        pool_padding=1)
    res2 = layer_warp(block_func, pool1, 64, stages[0], 1, is_first=True, is_test=is_test, name="res2")
    res3 = layer_warp(block_func, res2, 128, stages[1], 2, is_first=False, is_test=is_test, name="res3")
    res4 = layer_warp(block_func, res3, 256, stages[2], 2, is_first=False, is_test=is_test, name="res4")
    res5 = layer_warp(block_func, res4, 512, stages[3], 2, is_first=False, is_test=is_test, name="res5")
    return [res5,res4,res3]
