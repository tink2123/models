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

from __future__ import division
from __future__ import print_function

import paddle.fluid as fluid
from paddle.fluid.param_attr import ParamAttr
from paddle.fluid.initializer import Constant
from paddle.fluid.initializer import Normal
from paddle.fluid.regularizer import L2Decay

from config import cfg

from paddle.fluid.dygraph.nn import Conv2D, BatchNorm
from darknet_dy import DarkNet53_conv_body
from darknet_dy import ConvBNLayer

from paddle.fluid.dygraph.base import to_variable

class YoloDetectionBlock(fluid.dygraph.Layer):
    def __init__(self,name_scope,channel,is_test=True):
        super(YoloDetectionBlock, self).__init__(name_scope)

        assert channel % 2 == 0, \
            "channel {} cannot be divided by 2".format(channel)

        self.conv0 = ConvBNLayer(
            self.full_name(),
            ch_out=channel,
            filter_size=1,
            stride=1,
            padding=0,
            is_test=is_test
            )
        self.conv1 = ConvBNLayer(
            self.full_name(),
            ch_out=channel*2,
            filter_size=3,
            stride=1,
            padding=1,
            is_test=is_test
            )
        self.conv2 = ConvBNLayer(
            self.full_name(),
            ch_out=channel,
            filter_size=1,
            stride=1,
            padding=0,
            is_test=is_test
            )
        self.conv3 = ConvBNLayer(
            self.full_name(),
            ch_out=channel*2,
            filter_size=3,
            stride=1,
            padding=1,
            is_test=is_test
            )
        self.route = ConvBNLayer(
            self.full_name(),
            ch_out=channel,
            filter_size=1,
            stride=1,
            padding=0,
            is_test=is_test
            )
        self.tip = ConvBNLayer(
            self.full_name(),
            ch_out=channel*2,
            filter_size=3,
            stride=1,
            padding=1,
            is_test=is_test
            )
    def forward(self, inputs):
        out = self.conv0(inputs)
        out = self.conv1(out)
        out = self.conv2(out)
        out = self.conv3(out)
        route = self.route(out)
        tip = self.tip(route)
        return route, tip


class Upsample(fluid.dygraph.Layer):
    def __init__(self,name_scope,scale=2):
        super(Upsample,self).__init__(name_scope)
        self.scale = scale

    def forward(self, inputs):
        # get dynamic upsample output shape
        shape_nchw = fluid.layers.shape(inputs)
        shape_hw = fluid.layers.slice(shape_nchw, axes=[0], starts=[2], ends=[4])
        shape_hw.stop_gradient = True
        in_shape = fluid.layers.cast(shape_hw, dtype='int32')
        out_shape = in_shape * self.scale
        out_shape.stop_gradient = True

        # reisze by actual_shape
        out = fluid.layers.resize_nearest(
            input=inputs, scale=self.scale, actual_shape=out_shape)
        return out

class Yolov3(fluid.dygraph.Layer):
    def __init__(self,name_scope,is_train=True, use_random=False):
        super(Yolov3,self).__init__(name_scope)

        self.is_train = is_train
        self.use_random = use_random

        self.block = DarkNet53_conv_body(self.full_name(),
                                         is_test = not self.is_train)
        self.block_outputs = []
        self.yolo_blocks = []
        self.route_blocks_2 = []
        for i in range(3):
            yolo_block = self.add_sublayer(
                "yolo_detecton_block_%d" % (i),
                YoloDetectionBlock(self.full_name(),
                                   channel = 512//(2**i),
                                   is_test = not self.is_train))
            self.yolo_blocks.append(yolo_block)

            num_filters = len(cfg.anchor_masks[i]) * (cfg.class_num + 5)

            block_out = self.add_sublayer(
                "block_out_%d" % (i),
                Conv2D(self.full_name(),
                       num_filters=num_filters,
                       filter_size=1,
                       stride=1,
                       padding=0,
                       act=None,
                       param_attr=ParamAttr(
                           initializer=fluid.initializer.Normal(0., 0.02)),
                       bias_attr=ParamAttr(
                           initializer=fluid.initializer.Constant(0.0),
                           regularizer=L2Decay(0.))))
            self.block_outputs.append(block_out)
            if i < 2:
                route = self.add_sublayer("route2_%d"%i,
                                          ConvBNLayer(self.full_name(),
                                                      ch_out=256//(2**i),
                                                      filter_size=1,
                                                      stride=1,
                                                      padding=0,
                                                      is_test=(not self.is_train)))
                self.route_blocks_2.append(route)
            self.upsample = Upsample(self.full_name())

    def forward(self, inputs, gtbox, gtlabel, gtscore ):
        self.outputs = []
        self.boxes = []
        self.scores = []
        self.losses = []
        self.downsample = 32
        blocks = self.block(inputs)
        for i, block in enumerate(blocks):
            if i > 0:
                block = fluid.layers.concat(input=[route, block], axis=1)
            route, tip = self.yolo_blocks[i](block)
            block_out = self.block_outputs[i](tip)
            self.outputs.append(block_out)

            if i < 2:
                route = self.route_blocks_2[i](route)
                route = self.upsample(route)

        self.gtbox = gtbox
        self.gtlabel = gtlabel
        self.gtscore = gtscore

        # cal loss
        for i,out in enumerate(self.outputs):
            anchor_mask = cfg.anchor_masks[i]
            if self.is_train:
                loss = fluid.layers.yolov3_loss(
                    x=out,
                    gt_box=self.gtbox,
                    gt_label=self.gtlabel,
                    gt_score=self.gtscore,
                    anchors=cfg.anchors,
                    anchor_mask=anchor_mask,
                    class_num=cfg.class_num,
                    ignore_thresh=cfg.ignore_thresh,
                    downsample_ratio=self.downsample,
                    use_label_smooth=False)
                self.losses.append(fluid.layers.reduce_mean(loss))

            else:
                mask_anchors = []
                for m in anchor_mask:
                    mask_anchors.append(cfg.anchors[2 * m])
                    mask_anchors.append(cfg.anchors[2 * m + 1])
                boxes, scores = fluid.layers.yolo_box(
                    x=out,
                    img_size=self.im_shape,
                    anchors=mask_anchors,
                    class_num=cfg.class_num,
                    conf_thresh=cfg.valid_thresh,
                    downsample_ratio=self.downsample,
                    name="yolo_box" + str(i))
                self.boxes.append(boxes)
                self.scores.append(
                    fluid.layers.transpose(
                        scores, perm=[0, 2, 1]))
            self.downsample //= 2



        if not self.is_train:
        # get pred
            yolo_boxes = fluid.layers.concat(self.boxes, axis=1)
            yolo_scores = fluid.layers.concat(self.scores, axis=2)

            pred = fluid.layers.multiclass_nms(
                bboxes=yolo_boxes,
                scores=yolo_scores,
                score_threshold=cfg.valid_thresh,
                nms_top_k=cfg.nms_topk,
                keep_top_k=cfg.nms_posk,
                nms_threshold=cfg.nms_thresh,
                background_label=-1)
            return pred
        else:
            return sum(self.losses)




if __name__ == "__main__":
    import numpy as np
    import unittest
    import functools
    import yolov3
    import unittest
    from config import cfg

    import paddle.fluid as fluid
    from test_imperative_base import new_program_scope


    class TestDygraphGAN(unittest.TestCase):
        def test(self):
            startup = fluid.Program()
            startup.random_seed = 10
            main = fluid.Program()
            main.random_seed = 10
            scope = fluid.core.Scope()

            np.random.seed(1333)
            xyz_np = np.random.random((1, 3, 256, 256)).astype('float32')
            gtbox_np = np.array([[[0.5, 0.6, 0.3, 0.8], [0.4, 0.9, 0.2, 0.7]]]).astype('float32')
            gtlabel_np = np.array([[1, 2]]).astype('int32')
            gtscore_np = np.array([[0.4, 0.5]]).astype('float32')


            with fluid.dygraph.guard(fluid.CPUPlace()):
                fluid.default_startup_program().random_seed = 10
                fluid.default_main_program().random_seed = 10

                model = Yolov3("yolov3")
                opt = fluid.optimizer.SGD(learning_rate=0.001)
                #for param in model.parameters():
                    #print(param.name)
                    #dy_param_init_value[param.name] = param.numpy()

                data = to_variable(xyz_np)
                gtbox = to_variable(gtbox_np)

                gtlabel = to_variable(gtlabel_np)
                gtlabel.stop_gradient = True

                gtscore = to_variable(gtscore_np)
                gtscore.stop_gradient = True

                dy_param_gradient = []
                for i in range(5):
                    loss = model(data, gtbox, gtlabel, gtscore)
                    #loss = out["loss"]
                    loss.backward()

                    dy_gradient=[]
                    dy_param_name = []
                    for param in model.parameters():
                        #print(param.name)
                        #if param.name == "yolov3/Yolov3_0/DarkNet53_conv_body_0/ConvBNLayer_0/Conv2D_0.w_0":
                        if param.name.endswith("w_1") or param.name.endswith("w_2"):
                            pass
                        else:
                            dy_gradient.append(param.numpy())
                            dy_param_name.append(param.name)
                    print("dy loss:", loss.numpy())

                    opt.minimize(loss)

                    model.clear_gradients()

            with fluid.scope_guard(scope):
                fluid.default_startup_program().random_seed = 10
                fluid.default_main_program().random_seed = 10

                model = yolov3.YOLOv3()
                model.build_model()
                # out is correct
                out = model.outputs
                loss = model.loss()

                opt.minimize(loss)

                place = fluid.CPUPlace()
                exe = fluid.Executor(place)
                exe.run(fluid.default_startup_program())
                st_param = []
                for param in fluid.default_main_program().global_block().all_parameters():
                    st_param.append(param.name)
                print("len st param:", len(st_param))

                fetch_param = []
                for grad in st_param:
                    if grad.endswith("var") or grad.endswith("mean"):
                        pass
                    else:
                        #fetch_param.append(str(grad)+"@GRAD")
                        fetch_param.append(str(grad))


                for i in range(5):
                    ret = exe.run(fetch_list= [loss.name], feed={'xyz': xyz_np,
                                                                                      'gt_box': gtbox_np,
                                                                                      'gt_label': gtlabel_np,
                                                                                      'gt_score': gtscore_np})

                    loss_data = ret[-1]

                    print("loss data:", loss_data)

            print("st loss:",loss_data)


    unittest.main()

