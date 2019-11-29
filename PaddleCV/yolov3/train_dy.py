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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os


def set_paddle_flags(flags):
    for key, value in flags.items():
        if os.environ.get(key, None) is None:
            os.environ[key] = str(value)


set_paddle_flags({
    'FLAGS_eager_delete_tensor_gb': 0,  # enable gc
    'FLAGS_memory_fraction_of_eager_deletion': 1,
    'FLAGS_fraction_of_gpu_memory_to_use': 0.98
})

import sys
import numpy as np
import random
import time
import shutil
import subprocess
from utility import (parse_args, print_arguments,
                     SmoothedValue, check_gpu)

import paddle
import paddle.fluid as fluid
import reader
from models.yolov3_dy import Yolov3
from config import cfg
import dist_utils
from paddle.fluid.dygraph.base import to_variable


num_trainers = int(os.environ.get('PADDLE_TRAINERS_NUM', 1))

def get_device_num():
    # NOTE(zcd): for multi-processe training, each process use one GPU card.
    if num_trainers > 1:
        return 1
    return fluid.core.get_cuda_device_count()


def train():

    # check if set use_gpu=True in paddlepaddle cpu version
    check_gpu(cfg.use_gpu)

    devices_num = get_device_num()
    print("Found {} CUDA devices.".format(devices_num))

    if cfg.debug or args.enable_ce:
        fluid.default_startup_program().random_seed = 1000
        fluid.default_main_program().random_seed = 1000
        random.seed(0)
        np.random.seed(0)

    if not os.path.exists(cfg.model_save_dir):
        os.makedirs(cfg.model_save_dir)

    gpu_id = int(os.environ.get('FLAGS_selected_gpus', 0))
    place = fluid.CUDAPlace(gpu_id) if cfg.use_gpu else fluid.CPUPlace()

    with fluid.dygraph.guard(place):

        model = Yolov3("yolov3",is_train=True)

        learning_rate = cfg.learning_rate
        boundaries = cfg.lr_steps
        gamma = cfg.lr_gamma
        step_num = len(cfg.lr_steps)
        values = [learning_rate * (gamma ** i) for i in range(step_num + 1)]

        lr = fluid.layers.piecewise_decay(
            boundaries=boundaries,
            values=values
        )
        optimizer = fluid.optimizer.Momentum(
            learning_rate=lr,
            regularization=fluid.regularizer.L2Decay(cfg.weight_decay),
            momentum=cfg.momentum)


        start_time = time.time()
        snapshot_loss = 0
        snapshot_time = 0
        total_sample = 0

        input_size = cfg.input_size
        shuffle = True
        shuffle_seed = None
        total_iter = cfg.max_iter - cfg.start_iter
        mixup_iter = total_iter - cfg.no_mixup_iter

        random_sizes = [cfg.input_size]


        train_reader = reader.train(
            input_size,
            batch_size=cfg.batch_size,
            shuffle=shuffle,
            shuffle_seed=shuffle_seed,
            total_iter=total_iter * devices_num,
            mixup_iter=mixup_iter * devices_num,
            random_sizes=random_sizes,
            use_multiprocess_reader=cfg.use_multiprocess_reader,
            num_workers=cfg.worker_num)

        for iter_id in range(cfg.start_iter, cfg.max_iter):
            prev_start_time = start_time
            start_time = time.time()

            # todo:add reader
            """
            img = next(reader_data)
            gt_box = next()
            gt_label = next()
            gt_score = next()
            """
            data = next(train_reader())

            #print(len(data[0]))
            #print(data)
            #img, gt_box, gt_label, gt_score = data[0], data[1], data[2], data[3]
            #img = np.array([data[0].reshape(3,608,608)]).astype('float32')
            #img = to_variable(img)
            #gt_box = np.array([data[1]]).astype('float32')
            #gt_box = to_variable(gt_box)
            #gt_label = np.array([data[2]]).astype('int32')
            #gt_label = to_variable(gt_label)
            #gt_score = np.array([data[3]]).astype('float32')
            #gt_score = to_variable(gt_score)

            img = np.array([x[0] for x in data]).astype('float32')
            img = to_variable(img)
            
            gt_box = np.array([x[1] for x in data]).astype('float32')
            gt_box = to_variable(gt_box)
            
            gt_label = np.array([x[2] for x in data]).astype('int32')
            gt_label = to_variable(gt_label)

            gt_score = np.array([x[3] for x in data]).astype('float32')
            gt_score = to_variable(gt_score)


            loss = model(img, gt_box, gt_label, gt_score)
            snapshot_loss += loss.numpy()
            snapshot_time += start_time - prev_start_time
            total_sample += 1

            print("Iter {:d}, loss {:.6f}".format(
                iter_id,
                float(snapshot_loss/total_sample)))

            loss.backward()

            optimizer.minimize(loss)
            model.clear_gradients()

            if iter_id > 1 and iter_id % cfg.snapshot_iter == 0:
                fluid.save_dygraph(model.state_dict(),args.model_save_dir+"/yolove_{}".format(iter_id))




if __name__ == '__main__':
    args = parse_args()
    print_arguments(args)
    train()
