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
"""
Contains common utility functions.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import sys
import distutils.util
import numpy as np
import six
from collections import deque
from paddle.fluid import core
import argparse
import functools
from config.config import *


def print_arguments(args):
    """Print argparse's arguments.

    Usage:

    .. code-block:: python

        parser = argparse.ArgumentParser()
        parser.add_argument("name", default="Jonh", type=str, help="User name.")
        args = parser.parse_args()
        print_arguments(args)

    :param args: Input argparse.Namespace for printing.
    :type args: argparse.Namespace
    """
    print("-----------  Configuration Arguments -----------")
    for arg, value in sorted(six.iteritems(vars(args))):
        print("%s: %s" % (arg, value))
    print("------------------------------------------------")


def add_arguments(argname, type, default, help, argparser, **kwargs):
    """Add argparse's argument.

    Usage:

    .. code-block:: python

        parser = argparse.ArgumentParser()
        add_argument("name", str, "Jonh", "User name.", parser)
        args = parser.parse_args()
    """
    type = distutils.util.strtobool if type == bool else type
    argparser.add_argument(
        "--" + argname,
        default=default,
        type=type,
        help=help + ' Default: %(default)s.',
        **kwargs)


class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size):
        self.deque = deque(maxlen=window_size)

    def add_value(self, value):
        self.deque.append(value)

    def get_median_value(self):
        return np.median(self.deque)


def parse_args():
    """return all args
    """
    parser = argparse.ArgumentParser(description=__doc__)
    add_arg = functools.partial(add_arguments, argparser=parser)
    # yapf: disable
    # ENV
    add_arg('parallel',         bool,   True,       "Whether use parallel.")
    add_arg('use_gpu',          bool,   True,      "Whether use GPU.")
    # add_arg('data_cfg_path',    str,    'dataset/coco.data', "dataset split file config path")
    add_arg('name_path',        str,    'dataset/coco.names', "label name file path")
    add_arg('model_cfg_path',   str,    'config/yolov3.cfg', "YOLO model config file path.")
    add_arg('model_save_dir',   str,    'checkpoints',     "The path to save model.")
    add_arg('pretrain_base',    str,    'weights/pretrain', "The init model weights path.")
    add_arg('pretrained_model', str,    'weights/mxnet', "The pretrained model path.")
    add_arg('dataset',          str,   'coco2017',  "coco2014, coco2017.")
    add_arg('class_num',        int,   80,          "Class number.")
    add_arg('data_dir',         str,   'dataset/coco',        "The data root path.")
    add_arg('use_pyreader',     bool,   False,           "Use pyreader.")
    add_arg('use_profile',         bool,   False,       "Whether use profiler.")
    #SOLVER
    add_arg('learning_rate',    float,  0.001,     "Learning rate.")
    add_arg('max_iter',         int,    10000000,   "Iter number.")
    add_arg('log_window',       int,    20,        "Log smooth window, set 1 for debug, set 20 for train.")
    # TRAIN TEST INFER
    add_arg('im_per_batch',       int,   1,        "Minibatch size.")
    add_arg('valid_thresh',    float, 0.01,    "Valid confidence score for NMS.")
    add_arg('nms_thresh',    float, 0.45,    "NMS threshold.")
    add_arg('nms_topk',    int, 400,    "The number of boxes to perform NMS.")
    add_arg('nms_posk',    int, 100,    "The number of boxes of NMS output.")
    add_arg('conf_thresh',    float, 0.5,    "confidence, score threshold for NMS.")
    add_arg('snapshot_stride',  int,    2000,    "save model every snapshot stride.")
    add_arg('debug',            bool,   False,   "Debug mode")
    # SINGLE EVAL AND DRAW
    add_arg('image_path',       str,   'image',  "The image path used to inference and visualize.")
    add_arg('image_name',        str,    'dog.jpg',       "The single image used to inference and visualize.")
    # yapf: enable
    args = parser.parse_args()
    file_name = sys.argv[0]
    if 'train' in file_name or 'profile' in file_name:
        merge_cfg_from_args(args, 'train')
    else:
        merge_cfg_from_args(args, 'test')
    return args
