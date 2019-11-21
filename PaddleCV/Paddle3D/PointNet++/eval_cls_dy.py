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

import os
import sys
import time
import shutil
import argparse
import ast
import logging
import numpy as np
import paddle.fluid as fluid
from paddle.fluid.dygraph.base import to_variable
import paddle.fluid.layers.learning_rate_scheduler as lr_scheduler

from models import *
from data.modelnet40_reader import ModelNet40ClsReader
from utils import check_gpu, parse_outputs, Stat 
from data.data_utils import *


logging.root.handlers = []
FORMAT = '%(asctime)s-%(levelname)s: %(message)s'
logging.basicConfig(level=logging.INFO, format=FORMAT, stream=sys.stdout)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser("PointNet++ classification train script")
    parser.add_argument(
        '--model',
        type=str,
        default='MSG',
        help='SSG or MSG model to train, default MSG')
    parser.add_argument(
        '--use_gpu',
        type=ast.literal_eval,
        default=True,
        help='default use gpu.')
    parser.add_argument(
        '--use_data_parallel',
        type=ast.literal_eval,
        default=False,
        help='default training in single GPU.')
    parser.add_argument(
        '--batch_size',
        type=int,
        default=1,
        help='training batch size, default 1')
    parser.add_argument(
        '--num_points',
        type=int,
        default=2048,
        help='number of points in a sample, default: 2048')
    parser.add_argument(
        '--num_classes',
        type=int,
        default=40,
        help='number of classes in dataset, default: 40')
    parser.add_argument(
        '--lr',
        type=float,
        default=0.01,
        help='initial learning rate, default 0.01')
    parser.add_argument(
        '--lr_decay',
        type=float,
        default=0.7,
        help='learning rate decay gamma, default 0.5')
    parser.add_argument(
        '--bn_momentum',
        type=float,
        default=0.99,
        help='initial batch norm momentum, default 0.99')
    parser.add_argument(
        '--bn_decay',
        type=float,
        default=0.5,
        help='batch norm momentum decay gamma, default 0.5')
    parser.add_argument(
        '--decay_steps',
        type=int,
        default=12500,
        help='learning rate and batch norm momentum decay steps, default 12500')
    parser.add_argument(
        '--weight_decay',
        type=float,
        default=1e-5,
        help='L2 regularization weight decay coeff, default 1e-5.')
    parser.add_argument(
        '--epoch',
        type=int,
        default=201,
        help='epoch number. default 201.')
    parser.add_argument(
        '--data_dir',
        type=str,
        default='dataset/ModelNet40/modelnet40_ply_hdf5_2048',
        help='dataset directory')
    parser.add_argument(
        '--save_dir',
        type=str,
        default='checkpoints_cls',
        help='directory name to save train snapshoot')
    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='path to resume training based on previous checkpoints. '
        'None for not resuming any checkpoints.')
    parser.add_argument(
        '--log_interval',
        type=int,
        default=1,
        help='mini-batch interval for logging.')
    args = parser.parse_args()
    return args

def test():
    args = parse_args()
    with fluid.dygraph.guard():
        model = PointNet2ClsMSG_dy("pointnet2_cls_msg", num_classes=args.num_classes, num_points=args.num_points)
        restore,_ = fluid.load_dygraph(args.save_dir)
        model.set_dict(restore)
        model.eval()
        # lr = exponential_with_clip(args.lr, args.decay_steps, args.lr_decay, 1e-5)
        lr = fluid.layers.exponential_decay(
                learning_rate=args.lr,
                decay_steps=args.decay_steps,
                decay_rate=args.lr_decay,
                staircase=True)
        #lr = fluid.layers.clip(lr, 1e-5, args.lr)

        # get reader
        trans_list = [
            PointcloudScale(),
            PointcloudRotate(),
            PointcloudRotatePerturbation(),
            PointcloudTranslate(),
            PointcloudJitter(),
            PointcloudRandomInputDropout(),
        ]
        modelnet_reader = ModelNet40ClsReader(args.data_dir, mode='test', transforms=None)
        test_reader = modelnet_reader.get_reader(args.batch_size, args.num_points)
        global_step = 0
        bn_momentum = args.bn_momentum

        total_loss = 0.
        total_acc1 = 0.
        total_time = 0.
        total_sample = 0
        cur_time = time.time()
        for batch_id, data in enumerate(test_reader()):
            xyz_data = np.array([x[0].reshape(args.num_points, 3) for x in data]).astype('float32')
            label_data = np.array([x[1].reshape(1) for x in data]).astype('int64')
    
            xyz = to_variable(xyz_data)
            label = to_variable(label_data)
            label._stop_gradient = True
    
            out, loss, acc1 = model(xyz,label)
            period = time.time() - cur_time
            cur_time = time.time()
    
            total_loss += loss.numpy()[0]
            total_acc1 += acc1.numpy()[0]
            total_time += period
            total_sample += 1
    
            if batch_id % args.log_interval == 0:
                logger.info("[TEST] batch {}, loss: {:.3f}, acc(top-1): {:.3f}, time: {:.2f}".format(batch_id, total_loss / total_sample, total_acc1 / total_sample, total_time / total_sample))
        logger.info("[TEST] finish")



if __name__ == "__main__":
    test()

