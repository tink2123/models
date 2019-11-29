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

import paddle.fluid as fluid
from paddle.fluid.param_attr import ParamAttr
from paddle.fluid.regularizer import L2Decay

from paddle.fluid.dygraph.nn import Conv2D, BatchNorm
from paddle.fluid.dygraph.base import to_variable

class ConvBNLayer(fluid.dygraph.Layer):
    def __init__(self,
                 name_scope,
                 ch_out,
                 filter_size=3,
                 stride=1,
                 groups=1,
                 padding=0,
                 act="leaky",
                 is_test=True):
        super(ConvBNLayer, self).__init__(name_scope)

        self.conv = Conv2D(
            self.full_name(),
            num_filters=ch_out,
            filter_size=filter_size,
            stride=stride,
            padding=padding,
            groups=groups,
            param_attr=ParamAttr(
                initializer=fluid.initializer.Normal(0., 0.02)),
            bias_attr=False,
            act=None)

        self.batch_norm = BatchNorm(
            self.full_name(),
            num_channels=ch_out,
            is_test=is_test,
            param_attr=ParamAttr(
                initializer=fluid.initializer.Normal(0., 0.02),
                regularizer=L2Decay(0.)),
            bias_attr=ParamAttr(
                initializer=fluid.initializer.Constant(0.0),
                regularizer=L2Decay(0.)))

        self.act = act

    def forward(self, inputs):
        out = self.conv(inputs)
        out = self.batch_norm(out)
        if self.act == 'leaky':
            out = fluid.layers.leaky_relu(x=out, alpha=0.1)
        return out

class DownSample(fluid.dygraph.Layer):
    def __init__(self,
                 name_scope,
                 ch_out,
                 filter_size=3,
                 stride=2,
                 padding=1,
                 is_test=True):

        super(DownSample, self).__init__(name_scope)

        self.conv_bn_layer = ConvBNLayer(
            self.full_name(),
            ch_out=ch_out,
            filter_size=filter_size,
            stride=stride,
            padding=padding,
            is_test=is_test)
        self.ch_out = ch_out
    def forward(self, inputs):
        out = self.conv_bn_layer(inputs)
        return out

class BasicBlock(fluid.dygraph.Layer):
    def __init__(self, name_scope, ch_out, is_test=True):
        super(BasicBlock, self).__init__(name_scope)

        self.conv1 = ConvBNLayer(
            self.full_name(),
            ch_out=ch_out,
            filter_size=1,
            stride=1,
            padding=0,
            is_test=is_test
            )
        self.conv2 = ConvBNLayer(
            self.full_name(),
            ch_out=ch_out*2,
            filter_size=3,
            stride=1,
            padding=1,
            is_test=is_test
            )
    def forward(self, inputs):
        conv1 = self.conv1(inputs)
        conv2 = self.conv2(conv1)
        out = fluid.layers.elementwise_add(x=inputs, y=conv2, act=None)
        return out

class LayerWarp(fluid.dygraph.Layer):
    def __init__(self, name_scope, ch_out, count, is_test=True):
        super(LayerWarp,self).__init__(name_scope)

        self.basicblock0 = BasicBlock(self.full_name(),
            ch_out,
            is_test=is_test)
        self.res_out_list = []
        for i in range(1,count):
            res_out = self.add_sublayer("basic_block_%d" % (i),
                BasicBlock(self.full_name(),
                    ch_out,
                    is_test=is_test))
            self.res_out_list.append(res_out)

    def forward(self,inputs):
        y = self.basicblock0(inputs)
        for basic_block_i in self.res_out_list:
            y = basic_block_i(y)
        return y


DarkNet_cfg = {53: ([1, 2, 8, 8, 4])}


class DarkNet53_conv_body(fluid.dygraph.Layer):
    def __init__(self,
                 name_scope,
                 is_test=True):
        super(DarkNet53_conv_body, self).__init__(name_scope)
        self.stages = DarkNet_cfg[53]
        self.stages = self.stages[0:5]

        self.conv0 = ConvBNLayer(
            self.full_name(),
            ch_out=32,
            filter_size=3,
            stride=1,
            padding=1,
            is_test=is_test)

        self.downsample0 = DownSample(
            self.full_name(),
            ch_out=32 * 2,
            is_test=is_test)

        self.darknet53_conv_block_list = []
        self.downsample_list = []

        for i, stage in enumerate(self.stages):
            conv_block = self.add_sublayer(
                "stage_%d" % (i),
                LayerWarp(self.full_name(),
                32*(2**i),
                stage,
                is_test=is_test))
            self.darknet53_conv_block_list.append(conv_block)
        for i in range(len(self.stages) - 1):
            downsample = self.add_sublayer(
                "stage_%d_downsample" % i,
                DownSample(self.full_name(),
                    ch_out = 32*(2**(i+2)),
                    is_test=is_test))
            self.downsample_list.append(downsample)

    def forward(self,inputs):
        out = self.conv0(inputs)
        #print("conv1:",out.numpy())
        out = self.downsample0(out)
        #print("dy:",out.numpy())
        blocks = []
        for i, conv_block_i in enumerate(self.darknet53_conv_block_list):
            out = conv_block_i(out)
            blocks.append(out)
            if i < len(self.stages) - 1:
                out = self.downsample_list[i](out)
        return blocks[-1:-4:-1]

if __name__ == "__main__":
    import numpy as np
    import darknet
    import unittest

    import paddle.fluid as fluid
    from test_imperative_base import new_program_scope

    class TestDygraphGAN(unittest.TestCase):
        def test(self):
            startup = fluid.Program()
            startup.random_seed = 10
            main = fluid.Program()
            main.random_seed = 10
            scope = fluid.core.Scope()

            with fluid.scope_guard(scope):
                fluid.default_startup_program().random_seed = 10
                fluid.default_main_program().random_seed = 10

                xyz = fluid.layers.data(name='xyz', shape=[3, 256, 256], dtype='float32')
                out = darknet.add_DarkNet53_conv_body(xyz)
                place = fluid.CPUPlace()
                exe = fluid.Executor(place)
                exe.run(fluid.default_startup_program())
                #for param in fluid.default_startup_program().global_block().all_parameters():
                #    print (param.name)

                np.random.seed(1333)
                xyz_np = np.random.random((1, 3, 256, 256)).astype('float32')
                xyz_feats_np = np.random.random((1, 1)).astype('float32')
                # print("xyz: ", xyz_np.shape, xyz_np)
                # print("xyz_feats: ", xyz_feats_np.shape, xyz_feats_np)
                param_list = ["yolo_input.bn.scale"]
                ret = exe.run(fetch_list=param_list+[out[0]], feed={'xyz': xyz_np})
                pram = ret[0]
                sta_out = ret[-1]

            with fluid.dygraph.guard():
                fluid.default_startup_program().random_seed = 10
                fluid.default_main_program().random_seed = 10
                dy_param_init_value = {}
                model = DarkNet53_conv_body("darknet53")
                for param in model.parameters():
                    #print(param.name)
                    dy_param_init_value[param.name] = param.numpy()

                np.random.seed(1333)

                #data_np = np.random.random((1, 3, 256, 256)).astype('float32')

                data = to_variable(xyz_np)

                out = model(data)
                for i, block in enumerate(out):
                    print i
                    print (np.sum(block.numpy()))

                #print("st out: ", np.sum(np.abs(sta_out)))
                #print("dy out:", np.sum(np.abs(out[0].numpy())))

            self.assertTrue(np.allclose(sta_out, out[0].numpy()))


    unittest.main()

