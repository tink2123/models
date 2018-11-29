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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import sys
import numpy as np
import random
import time
import shutil
from utility import parse_args, print_arguments, SmoothedValue

import paddle
import paddle.fluid as fluid
import reader
import models
from config.config import cfg


def train():

    if cfg.debug:
        fluid.default_startup_program().random_seed = 1000
        fluid.default_main_program().random_seed = 1000
        random.seed(0)
        np.random.seed(0)

    # devices = os.getenv("CUDA_VISIBLE_DEVICES") or ""
    # devices_num = len(devices.split(","))
    # total_batch_size = devices_num * cfg.TRAIN.im_per_batch

    model = models.YOLOv3(cfg.model_cfg_path)
    model.build_model()
    loss = model.loss()
    loss.persistable = True

    hyperparams = model.get_hyperparams()
    learning_rate = float(hyperparams['learning_rate'])
    optimizer = fluid.optimizer.Adam(learning_rate=learning_rate)
    optimizer.minimize(loss)

    fluid.memory_optimize(fluid.default_main_program())

    place = fluid.CUDAPlace(0) if cfg.use_gpu else fluid.CPUPlace()
    exe = fluid.Executor(place)
    exe.run(fluid.default_startup_program())
    fluid.io.save_persistables(exe, "./test")

    if cfg.pretrained_model:
        def if_exist(var):
            return os.path.exists(os.path.join(cfg.pretrained_model, var.name))
        fluid.io.load_vars(exe, cfg.pretrained_model, predicate=if_exist)

    if cfg.parallel:
        train_exe = fluid.ParallelExecutor(
            use_cuda=bool(cfg.use_gpu), loss_name=loss.name)

    # if cfg.use_pyreader:
    #     train_reader = reader.train(
    #         batch_size=cfg.TRAIN.im_per_batch,
    #         total_batch_size=total_batch_size,
    #         padding_total=cfg.TRAIN.padding_minibatch,
    #         shuffle=True)
    #     py_reader = model.py_reader
    #     py_reader.decorate_paddle_reader(train_reader)
    input_size = model.get_input_size()
    train_reader = reader.train(input_size, batch_size=cfg.TRAIN.im_per_batch, shuffle=True)
    feeder = fluid.DataFeeder(place=place, feed_list=model.feeds())

    def save_model(postfix):
        model_path = os.path.join(cfg.model_save_dir, postfix)
        if os.path.isdir(model_path):
            shutil.rmtree(model_path)
        fluid.io.save_persistables(exe, model_path)

    fetch_list = [loss]

    def train_loop_pyreader():
        py_reader.start()
        smoothed_loss = SmoothedValue(cfg.log_window)
        try:
            start_time = time.time()
            prev_start_time = start_time
            every_pass_loss = []
            for iter_id in range(cfg.max_iter):
                prev_start_time = start_time
                start_time = time.time()
                losses = train_exe.run(fetch_list=[v.name for v in fetch_list])
                every_pass_loss.append(np.mean(np.array(losses[0])))
                smoothed_loss.add_value(np.mean(np.array(losses[0])))
                lr = np.array(fluid.global_scope().find_var('learning_rate')
                              .get_tensor())
                print("Iter {:d}, lr {:.6f}, loss {:.6f}, time {:.5f}".format(
                    iter_id, lr[0],
                    smoothed_loss.get_median_value(
                    ), start_time - prev_start_time))
                sys.stdout.flush()
                if (iter_id + 1) % cfg.TRAIN.snapshot_iter == 0:
                    save_model("model_iter{}".format(iter_id))
        except fluid.core.EOFException:
            py_reader.reset()
        return np.mean(every_pass_loss)

    def train_loop():
        start_time = time.time()
        prev_start_time = start_time
        start = start_time
        every_pass_loss = []
        smoothed_loss = SmoothedValue(cfg.log_window)
        for iter_id, data in enumerate(train_reader()):
            prev_start_time = start_time
            start_time = time.time()
            losses = train_exe.run(fetch_list=[v.name for v in fetch_list],
                                   feed=feeder.feed(data))
            # loss_v = np.mean(np.array(losses[0]))
            every_pass_loss.append(losses[0])
            smoothed_loss.add_value(losses[0])
            lr = np.array(fluid.global_scope().find_var('learning_rate')
                          .get_tensor())
            print("Iter {:d}, lr {:.6f}, loss {:.6f}, time {:.5f}".format(
                iter_id, lr[0], 
                smoothed_loss.get_median_value(), start_time - prev_start_time))
            sys.stdout.flush()
            if (iter_id + 1) % cfg.TRAIN.snapshot_iter == 0:
                save_model("model_iter{}".format(iter_id))
            if (iter_id + 1) == cfg.max_iter:
                break
        return np.mean(every_pass_loss)

    if cfg.use_pyreader:
        train_loop_pyreader()
    else:
        train_loop()
    save_model('model_final')


if __name__ == '__main__':
    args = parse_args()
    print_arguments(args)
    train()
