# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from paddle import fluid
from paddle.fluid.param_attr import ParamAttr
from paddle.fluid.regularizer import L2Decay

from ppdet.core.workspace import register

__all__ = ['MobileNet','MobileNetV3']


@register
class MobileNetV3(object):
    """
    MobileNet v3
    """

    def __init__(self, scale=1.0, model_name='large'):
        self.scale = scale
        self.inplanes = 16
        if model_name == "large":
            self.cfg = [
                # k, exp, c,  se,     nl,  s,
                [3, 16, 16, False, 'relu', 1],
                [3, 64, 24, False, 'relu', 2],
                [3, 72, 24, False, 'relu', 1],
                [5, 72, 40, True, 'relu', 2],
                [5, 120, 40, True, 'relu', 1],
                [5, 120, 40, True, 'relu', 1],
                [3, 240, 80, False, 'hard_swish', 2],
                [3, 200, 80, False, 'hard_swish', 1],
                [3, 184, 80, False, 'hard_swish', 1],
                [3, 184, 80, False, 'hard_swish', 1],
                [3, 480, 112, True, 'hard_swish', 1],
                [3, 672, 112, True, 'hard_swish', 1],
                #[5, 672, 160, True, 'hard_swish', 2],
                #[5, 960, 160, True, 'hard_swish', 1],
                #[5, 960, 160, True, 'hard_swish', 1],
            ]
            self.cls_ch_squeeze = 960
            self.cls_ch_expand = 1280
        elif model_name == "small":
            self.cfg = [
                # k, exp, c,  se,     nl,  s,
                [3, 16, 16, True, 'relu', 2],
                [3, 72, 24, False, 'relu', 2],
                [3, 88, 24, False, 'relu', 1],
                [5, 96, 40, True, 'hard_swish', 2],
                [5, 240, 40, True, 'hard_swish', 1],
                [5, 240, 40, True, 'hard_swish', 1],
                [5, 120, 48, True, 'hard_swish', 1],
                [5, 144, 48, True, 'hard_swish', 1],
                #[5, 288, 96, True, 'hard_swish', 2],
                #[5, 576, 96, True, 'hard_swish', 1],
                #[5, 576, 96, True, 'hard_swish', 1],
            ]
            self.cls_ch_squeeze = 576
            self.cls_ch_expand = 1280
        else:
            raise NotImplementedError("mode[" + model_name +
                                      "_model] is not implemented!")

    #def net(self, input, class_dim=1000):
    def __call__(self, input):
        scale = self.scale
        inplanes = self.inplanes
        cfg = self.cfg
        cls_ch_squeeze = self.cls_ch_squeeze
        cls_ch_expand = self.cls_ch_expand

        #conv1
        conv = self.conv_bn_layer(
            input,
            filter_size=3,
            #num_filters=int(scale * inplanes),
            num_filters=inplanes if scale <= 1.0 else int(inplanes * scale), 
            stride=2,
            padding=1,
            num_groups=1,
            if_act=True,
            act='hard_swish',
            name='conv1')
        i = 0
        for layer_cfg in cfg:
            conv = self.residual_unit(
                input=conv,
                num_in_filter=inplanes,
                num_mid_filter=int(scale * layer_cfg[1]),
                num_out_filter=int(scale * layer_cfg[2]),
                act=layer_cfg[4],
                stride=layer_cfg[5],
                filter_size=layer_cfg[0],
                use_se=layer_cfg[3],
                name='conv' + str(i + 2))
            inplanes = int(scale * layer_cfg[2])
            i += 1
            print(layer_cfg[1])
            if layer_cfg[1] == 120:
            #if layer_cfg[1] == 88:
                print("define conv1")
                out1 = conv
            if layer_cfg[1] == 672:
            #if layer_cfg[1] == 144:
                print("define conv2")
                out2 = conv

        return out1,out2

    def conv_bn_layer(self,
                      input,
                      filter_size,
                      num_filters,
                      stride,
                      padding,
                      num_groups=1,
                      if_act=True,
                      act=None,
                      name=None,
                      use_cudnn=True):
        conv = fluid.layers.conv2d(
            input=input,
            num_filters=num_filters,
            filter_size=filter_size,
            stride=stride,
            padding=padding,
            groups=num_groups,
            act=None,
            use_cudnn=use_cudnn,
            param_attr=ParamAttr(name=name + '_weights'),
            bias_attr=False)
        bn_name = name + '_bn'
        bn = fluid.layers.batch_norm(
            input=conv,
            param_attr=ParamAttr(name=bn_name + "_scale"),
            bias_attr=ParamAttr(name=bn_name + "_offset"),
            moving_mean_name=bn_name + '_mean',
            moving_variance_name=bn_name + '_variance')
        if if_act:
            if act == 'relu':
                bn = fluid.layers.relu(bn)
            elif act == 'hard_swish':
                bn = self.hard_swish(bn)
        return bn

    def hard_swish(self, x):
        return x * fluid.layers.relu6(x + 3) / 6.

    def se_block(self, input, num_out_filter, ratio=4, name=None):
        num_mid_filter = int(num_out_filter // ratio)
        pool = fluid.layers.pool2d(
            input=input, pool_type='avg', global_pooling=True, use_cudnn=False)
        conv1 = fluid.layers.conv2d(
            input=pool,
            filter_size=1,
            num_filters=num_mid_filter,
            act='relu',
            param_attr=ParamAttr(name=name + '_1_weights'),
            bias_attr=ParamAttr(name=name + '_1_offset'))
        conv2 = fluid.layers.conv2d(
            input=conv1,
            filter_size=1,
            num_filters=num_out_filter,
            act='hard_sigmoid',
            param_attr=ParamAttr(name=name + '_2_weights'),
            bias_attr=ParamAttr(name=name + '_2_offset'))
        scale = fluid.layers.elementwise_mul(x=input, y=conv2, axis=0)
        return scale

    def residual_unit(self,
                      input,
                      num_in_filter,
                      num_mid_filter,
                      num_out_filter,
                      stride,
                      filter_size,
                      act=None,
                      use_se=False,
                      name=None):

        #first_conv = (num_out_filter != num_mid_filter)
        input_data = input
        #if first_conv:
        #    input = self.conv_bn_layer(
        #        input=input,
        #        filter_size=1,
        #        num_filters=num_mid_filter,
        #        stride=1,
        #        padding=0,
        #        if_act=True,
        #        act=act,
        #        name=name + '_expand')

        conv0 = self.conv_bn_layer(input=input,
                                   filter_size=1,
                                   num_filters=num_mid_filter,
                                   stride=1,
                                   padding=0,
                                   if_act=True,
                                   act=act,
                                   name=name + '_expand')

        conv1 = self.conv_bn_layer(
            input=conv0,
            filter_size=filter_size,
            num_filters=num_mid_filter,
            stride=stride,
            padding=int((filter_size - 1) // 2),
            if_act=True,
            act=act,
            num_groups=num_mid_filter,
            use_cudnn=True,
            name=name + '_depthwise')
        if use_se:
            conv1 = self.se_block(
                input=conv1, num_out_filter=num_mid_filter, name=name + '_se')

        conv2 = self.conv_bn_layer(
            input=conv1,
            filter_size=1,
            num_filters=num_out_filter,
            stride=1,
            padding=0,
            if_act=False,
            name=name + '_linear')
        if num_in_filter != num_out_filter or stride != 1:
            return conv2
        else:
            return fluid.layers.elementwise_add(x=input_data, y=conv2, act=None)

@register
class MobileNet(object):
    """
    MobileNet v1, see https://arxiv.org/abs/1704.04861

    Args:
        norm_type (str): normalization type, 'bn' and 'sync_bn' are supported
        norm_decay (float): weight decay for normalization layer weights
        conv_group_scale (int): scaling factor for convolution groups
        with_extra_blocks (bool): if extra blocks should be added
        extra_block_filters (list): number of filter for each extra block
    """
    __shared__ = ['norm_type', 'weight_prefix_name']

    def __init__(self,
                 norm_type='bn',
                 norm_decay=0.,
                 conv_group_scale=1,
                 conv_learning_rate=1.0,
                 with_extra_blocks=False,
                 extra_block_filters=[[256, 512], [128, 256], [128, 256],
                                      [64, 128]],
                 weight_prefix_name=''):
        self.norm_type = norm_type
        self.norm_decay = norm_decay
        self.conv_group_scale = conv_group_scale
        self.conv_learning_rate = conv_learning_rate
        self.with_extra_blocks = with_extra_blocks
        self.extra_block_filters = extra_block_filters
        self.prefix_name = weight_prefix_name

    def _conv_norm(self,
                   input,
                   filter_size,
                   num_filters,
                   stride,
                   padding,
                   num_groups=1,
                   act='relu',
                   use_cudnn=True,
                   name=None):
        parameter_attr = ParamAttr(
            learning_rate=self.conv_learning_rate,
            initializer=fluid.initializer.MSRA(),
            name=name + "_weights")
        conv = fluid.layers.conv2d(
            input=input,
            num_filters=num_filters,
            filter_size=filter_size,
            stride=stride,
            padding=padding,
            groups=num_groups,
            act=None,
            use_cudnn=use_cudnn,
            param_attr=parameter_attr,
            bias_attr=False)

        bn_name = name + "_bn"
        norm_decay = self.norm_decay
        bn_param_attr = ParamAttr(
            regularizer=L2Decay(norm_decay), name=bn_name + '_scale')
        bn_bias_attr = ParamAttr(
            regularizer=L2Decay(norm_decay), name=bn_name + '_offset')
        return fluid.layers.batch_norm(
            input=conv,
            act=act,
            param_attr=bn_param_attr,
            bias_attr=bn_bias_attr,
            moving_mean_name=bn_name + '_mean',
            moving_variance_name=bn_name + '_variance')

    def depthwise_separable(self,
                            input,
                            num_filters1,
                            num_filters2,
                            num_groups,
                            stride,
                            scale,
                            name=None):
        depthwise_conv = self._conv_norm(
            input=input,
            filter_size=3,
            num_filters=int(num_filters1 * scale),
            stride=stride,
            padding=1,
            num_groups=int(num_groups * scale),
            use_cudnn=False,
            name=name + "_dw")

        pointwise_conv = self._conv_norm(
            input=depthwise_conv,
            filter_size=1,
            num_filters=int(num_filters2 * scale),
            stride=1,
            padding=0,
            name=name + "_sep")
        return pointwise_conv

    def _extra_block(self,
                     input,
                     num_filters1,
                     num_filters2,
                     num_groups,
                     stride,
                     name=None):
        pointwise_conv = self._conv_norm(
            input=input,
            filter_size=1,
            num_filters=int(num_filters1),
            stride=1,
            num_groups=int(num_groups),
            padding=0,
            name=name + "_extra1")
        normal_conv = self._conv_norm(
            input=pointwise_conv,
            filter_size=3,
            num_filters=int(num_filters2),
            stride=2,
            num_groups=int(num_groups),
            padding=1,
            name=name + "_extra2")
        return normal_conv

    def __call__(self, input):
        scale = self.conv_group_scale

        blocks = []
        # input 1/1
        out = self._conv_norm(
            input, 3, int(32 * scale), 2, 1, name=self.prefix_name + "conv1")
        # 1/2
        out = self.depthwise_separable(
            out, 32, 64, 32, 1, scale, name=self.prefix_name + "conv2_1")
        out = self.depthwise_separable(
            out, 64, 128, 64, 2, scale, name=self.prefix_name + "conv2_2")
        # 1/4
        out = self.depthwise_separable(
            out, 128, 128, 128, 1, scale, name=self.prefix_name + "conv3_1")
        out = self.depthwise_separable(
            out, 128, 256, 128, 2, scale, name=self.prefix_name + "conv3_2")
        # 1/8
        blocks.append(out)
        out = self.depthwise_separable(
            out, 256, 256, 256, 1, scale, name=self.prefix_name + "conv4_1")
        out = self.depthwise_separable(
            out, 256, 512, 256, 2, scale, name=self.prefix_name + "conv4_2")
        # 1/16
        blocks.append(out)
        for i in range(5):
            out = self.depthwise_separable(
                out,
                512,
                512,
                512,
                1,
                scale,
                name=self.prefix_name + "conv5_" + str(i + 1))
        module11 = out

        out = self.depthwise_separable(
            out, 512, 1024, 512, 2, scale, name=self.prefix_name + "conv5_6")
        # 1/32
        out = self.depthwise_separable(
            out, 1024, 1024, 1024, 1, scale, name=self.prefix_name + "conv6")
        module13 = out
        blocks.append(out)
        if not self.with_extra_blocks:
            return blocks

        num_filters = self.extra_block_filters
        module14 = self._extra_block(module13, num_filters[0][0],
                                     num_filters[0][1], 1, 2,
                                     self.prefix_name + "conv7_1")
        module15 = self._extra_block(module14, num_filters[1][0],
                                     num_filters[1][1], 1, 2,
                                     self.prefix_name + "conv7_2")
        module16 = self._extra_block(module15, num_filters[2][0],
                                     num_filters[2][1], 1, 2,
                                     self.prefix_name + "conv7_3")
        module17 = self._extra_block(module16, num_filters[3][0],
                                     num_filters[3][1], 1, 2,
                                     self.prefix_name + "conv7_4")
        return module11, module13, module14, module15, module16, module17
