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

from ppdet.core.workspace import register

__all__ = ['BlazeNet']


@register
class BlazeNet(object):
    """
    BlazeFace, see https://arxiv.org/abs/1907.05047

    Args:
        blaze_filters (list): number of filter for each blaze block
        double_blaze_filters (list): number of filter for each double_blaze block
        with_extra_blocks (bool): whether or not extra blocks should be added
        lite_edition (bool): whether or not is blazeface-lite
    """

    def __init__(
            self,
            blaze_filters=[[24, 24], [24, 24], [24, 48, 2], [48, 48], [48, 48]],
            double_blaze_filters=[[48, 24, 96, 2], [96, 24, 96], [96, 24, 96],
                                  [96, 24, 96, 2], [96, 24, 96], [96, 24, 96]],
            with_extra_blocks=True,
            lite_edition=True):
            #lite_edition=False):
        super(BlazeNet, self).__init__()

        self.blaze_filters = blaze_filters
        self.double_blaze_filters = double_blaze_filters
        self.with_extra_blocks = with_extra_blocks
        self.lite_edition = lite_edition

    def __call__(self, input):
        if not self.lite_edition:
            conv1_num_filters = self.blaze_filters[0][0]
            conv = self._conv_norm(
                input=input,
                num_filters=conv1_num_filters,
                filter_size=3,
                stride=2,
                padding=1,
                act='hard_swish',
                name="conv1")

            for k, v in enumerate(self.blaze_filters):
                assert len(v) in [2, 3], \
                    "blaze_filters {} not in [2, 3]"
                if len(v) == 2:
                    conv = self.BlazeBlock(
                        conv, v[0], v[1], name='blaze_{}'.format(k))
                elif len(v) == 3:
                    conv = self.BlazeBlock(
                        conv,
                        v[0],
                        v[1],
                        stride=v[2],
                        name='blaze_{}'.format(k))

            layers = []
            for k, v in enumerate(self.double_blaze_filters):
                assert len(v) in [3, 4], \
                    "blaze_filters {} not in [3, 4]"
                if len(v) == 3:
                    conv = self.BlazeBlock(
                        conv,
                        v[0],
                        v[1],
                        double_channels=v[2],
                        #act = "hard_swish",
                        name='double_blaze_{}'.format(k))
                elif len(v) == 4:
                    layers.append(conv)
                    conv = self.BlazeBlock(
                        conv,
                        v[0],
                        v[1],
                        double_channels=v[2],
                        stride=v[3],
                        #act = "hard_swish",
                        name='double_blaze_{}'.format(k))
            layers.append(conv)

            if not self.with_extra_blocks:
                return layers[-1]
            print("feature1:",layers[-2].shape)
            print("feature2:",layers[-1].shape)

            # add fpn
            output1, output2 = self.fpn([layers[-2],layers[-1]],layers[-2].shape[1],"fpn")
            print("output1:",output1.shape)
            print("output2:",output2.shape)
            # add ssh
            feature1 = self.ssh(output1,out_channel=layers[-2].shape[1],name="ssh1")
            feature2 = self.ssh(output2,out_channel=layers[-1].shape[1],name="ssh2")
            print("ssh1:",feature1.shape)
            print("ssh2:",feature2.shape)
            return feature1, feature2
            #return layers[-2], layers[-1]
        else:
            conv1 = self._conv_norm(
                input=input,
                num_filters=24,
                filter_size=5,
                stride=2,
                padding=2,
                act='relu',
                name="conv1")
            conv2 = self.Blaze_lite(conv1, 24, 24, 1, 'conv2')
            conv3 = self.Blaze_lite(conv2, 24, 28, 1, 'conv3')
            conv4 = self.Blaze_lite(conv3, 28, 32, 2, 'conv4')
            conv5 = self.Blaze_lite(conv4, 32, 36, 1, 'conv5')
            conv6 = self.Blaze_lite(conv5, 36, 42, 1, 'conv6')
            conv7 = self.Blaze_lite(conv6, 42, 48, 2, 'conv7')
            in_ch = 48
            for i in range(5):
                conv7 = self.Blaze_lite(conv7, in_ch, in_ch + 8, 1,
                                        'conv{}'.format(8 + i))
                in_ch += 8
            assert in_ch == 88
            conv13 = self.Blaze_lite(conv7, 88, 96, 2, 'conv13')
            for i in range(4):
                conv13 = self.Blaze_lite(conv13, 96, 96, 1,
                                         'conv{}'.format(14 + i))

            return conv7, conv13

    def BlazeBlock(self,
                   input,
                   in_channels,
                   out_channels,
                   double_channels=None,
                   stride=1,
                   use_5x5kernel=True,
                   name=None):
        assert stride in [1, 2]
        use_pool = not stride == 1
        use_double_block = double_channels is not None
        #act = 'relu' if use_double_block else None
        act = "hard_swish" if use_double_block else None

        if use_5x5kernel:
            conv_dw = self._conv_norm(
                input=input,
                filter_size=5,
                num_filters=in_channels,
                stride=stride,
                padding=2,
                num_groups=in_channels,
                use_cudnn=False,
                name=name + "1_dw")
        else:
            conv_dw_1 = self._conv_norm(
                input=input,
                filter_size=3,
                num_filters=in_channels,
                stride=1,
                padding=1,
                num_groups=in_channels,
                use_cudnn=False,
                name=name + "1_dw_1")
            conv_dw = self._conv_norm(
                input=conv_dw_1,
                filter_size=3,
                num_filters=in_channels,
                stride=stride,
                padding=1,
                num_groups=in_channels,
                use_cudnn=False,
                name=name + "1_dw_2")
        #if use_se:
        #    conv_dw = self.se_block(
        #        input=conv_dw, num_out_filter=num_mid_filter, name=name + '_se')

        conv_pw = self._conv_norm(
            input=conv_dw,
            filter_size=1,
            num_filters=out_channels,
            stride=1,
            padding=0,
            act=act,
            name=name + "1_sep")

        if use_double_block:
            if use_5x5kernel:
                conv_dw = self._conv_norm(
                    input=conv_pw,
                    filter_size=5,
                    num_filters=out_channels,
                    stride=1,
                    padding=2,
                    use_cudnn=False,
                    name=name + "2_dw")
            else:
                conv_dw_1 = self._conv_norm(
                    input=conv_pw,
                    filter_size=3,
                    num_filters=out_channels,
                    stride=1,
                    padding=1,
                    num_groups=out_channels,
                    use_cudnn=False,
                    name=name + "2_dw_1")
                conv_dw = self._conv_norm(
                    input=conv_dw_1,
                    filter_size=3,
                    num_filters=out_channels,
                    stride=1,
                    padding=1,
                    num_groups=out_channels,
                    use_cudnn=False,
                    name=name + "2_dw_2")

            conv_pw = self._conv_norm(
                input=conv_dw,
                filter_size=1,
                num_filters=double_channels,
                stride=1,
                padding=0,
                name=name + "2_sep")

        # shortcut
        if use_pool:
            shortcut_channel = double_channels or out_channels
            shortcut_pool = self._pooling_block(input, stride, stride)
            channel_pad = self._conv_norm(
                input=shortcut_pool,
                filter_size=1,
                num_filters=shortcut_channel,
                stride=1,
                padding=0,
                name="shortcut" + name)
            return fluid.layers.elementwise_add(
                x=channel_pad, y=conv_pw, act='relu')
        return fluid.layers.elementwise_add(x=input, y=conv_pw, act='relu')

    def Blaze_lite(self, input, in_channels, out_channels, stride=1, name=None):
        assert stride in [1, 2]
        use_pool = not stride == 1
        ues_pad = not in_channels == out_channels
        conv_dw = self._conv_norm(
            input=input,
            filter_size=3,
            num_filters=in_channels,
            stride=stride,
            padding=1,
            num_groups=in_channels,
            name=name + "_dw")

        conv_pw = self._conv_norm(
            input=conv_dw,
            filter_size=1,
            num_filters=out_channels,
            stride=1,
            padding=0,
            name=name + "_sep")

        if use_pool:
            shortcut_pool = self._pooling_block(input, stride, stride)
        if ues_pad:
            conv_pad = shortcut_pool if use_pool else input
            channel_pad = self._conv_norm(
                input=conv_pad,
                filter_size=1,
                num_filters=out_channels,
                stride=1,
                padding=0,
                name="shortcut" + name)
            return fluid.layers.elementwise_add(
                x=channel_pad, y=conv_pw, act='relu')
        return fluid.layers.elementwise_add(x=input, y=conv_pw, act='relu')


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

    def fpn(self,input,out_channels,name=None):
        output1 = self._conv_norm(
            input = input[0],
            filter_size=1,
            padding=0,
            num_filters = out_channels,
            stride=1,
            act="leaky",
            name=name+"output1"
        )
        output2 = self._conv_norm(
            input=input[1],
            filter_size=1,
            num_filters=out_channels,
            padding=0,
            stride=1,
            act="leaky",
            name=name+"output2"
        )
        up2 = fluid.layers.resize_nearest(output2,out_shape=[output1.shape[2],output1.shape[3]])
        output1 = fluid.layers.elementwise_add(output1,up2)
        output1 = self._conv_norm(
            input=output1,
            filter_size=3,
            stride=1,
            padding=1,
            num_filters=out_channels,
            act="leaky",
            name=name+"merge1"
        )
        return output1, output2



    def ssh(self,input,out_channel,name=None):
        assert out_channel % 4 ==0
        # 3*3
        conv0 = self._conv_norm(
            input=input,
            filter_size=3,
            num_filters=out_channel//2,
            stride=1,
            padding=1,
            act=None,
            name = name + "ssh_conv3"
        )
        # 5*5_1
        conv1 = self._conv_norm(
            input=conv0,
            filter_size=3,
            num_filters=out_channel//4,
            stride=1,
            padding=1,
            act="leaky",
            name=name+"ssh_conv5_1"
        )
        # 5*5_2
        conv2 = self._conv_norm(
            input=conv1,
            filter_size=3,
            num_filters=out_channel // 4,
            stride=1,
            padding=1,
            act=None,
            name=name+"ssh_conv5_2"
        )
        # 7*7_2
        conv3 = self._conv_norm(
            input=conv2,
            filter_size=3,
            num_filters=out_channel // 4,
            stride=1,
            padding=1,
            act="leaky",
            name=name+"ssh_conv7_1"
        )
        # 7*7_3
        conv4 = self._conv_norm(
            input=conv3,
            filter_size=3,
            num_filters=out_channel // 4,
            stride=1,
            padding=1,
            act=None,
            name=name+"ssh_conv7_2"
        )
        concat = fluid.layers.concat([conv0,conv2,conv4],axis=1)
        return fluid.layers.relu(concat)

    def _conv_norm(
            self,
            input,
            filter_size,
            num_filters,
            stride,
            padding,
            num_groups=1,
            act='relu',  # None
            use_cudnn=True,
            name=None):
        parameter_attr = ParamAttr(
            learning_rate=0.1,
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
        bn_name = name + '_bn'
        bn = fluid.layers.batch_norm(
            input=conv,
            param_attr=ParamAttr(name=bn_name + "_scale"),
            bias_attr=ParamAttr(name=bn_name + "_offset"),
            moving_mean_name=bn_name + '_mean',
            moving_variance_name=bn_name + '_variance')

        if act == 'relu':
            bn = fluid.layers.relu(bn)
        elif act == 'hard_swish':
            bn = self.hard_swish(bn)
        elif act == "leaky":
            bn = fluid.layers.leaky_relu(bn,alpha=0.01)
        return bn


    def _pooling_block(self,
                       conv,
                       pool_size,
                       pool_stride,
                       pool_padding=0,
                       ceil_mode=True):
        pool = fluid.layers.pool2d(
            input=conv,
            pool_size=pool_size,
            pool_type='max',
            pool_stride=pool_stride,
            pool_padding=pool_padding,
            ceil_mode=ceil_mode)
        return pool

