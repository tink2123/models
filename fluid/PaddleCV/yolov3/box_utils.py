# Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved
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
from __future__ import unicode_literals

import numpy as np
from PIL import Image, ImageDraw


def sigmoid(x):
    """Perform sigmoid to input numpy array"""
    return 1.0 / (1.0 + np.exp(-1.0 * x))

def get_yolo_detection(preds, anchors, class_num, img_width, img_height):
    """Get yolo box, confidence score, class label from Darknet53 output"""
    preds_n = np.array(preds)
    n, c, h, w = preds_n.shape
    anchor_num = len(anchors) // 2
    preds_n = preds_n.reshape([n, anchor_num, class_num + 5, h, w]) \
                     .transpose((0, 1, 3, 4, 2))
    preds_n[:, :, :, :, :2] = sigmoid(preds_n[:, :, :, :, :2])
    preds_n[:, :, :, :, 4:] = sigmoid(preds_n[:, :, :, :, 4:])

    pred_boxes = preds_n[:, :, :, :, :4]
    pred_confs = preds_n[:, :, :, :, 4]
    # pred_classes = preds_n[:, :, :, :, 5:]
    pred_classes = preds_n[:, :, :, :, 5:] * np.expand_dims(pred_confs, axis=4)

    grid_x = np.tile(np.arange(w).reshape((1, w)), (h, 1))
    grid_y = np.tile(np.arange(h).reshape((h, 1)), (1, w))
    anchors = [(anchors[i], anchors[i+1]) for i in range(0, len(anchors), 2)]
    anchors_s = np.array([(an_w * w / img_width, an_h * h / img_height) for an_w, an_h in anchors])
    anchor_w = anchors_s[:, 0:1].reshape((1, anchor_num, 1, 1))
    anchor_h = anchors_s[:, 1:2].reshape((1, anchor_num, 1, 1))

    pred_boxes[:, :, :, :, 0] += grid_x
    pred_boxes[:, :, :, :, 1] += grid_y
    pred_boxes[:, :, :, :, 2] = np.exp(pred_boxes[:, :, :, :, 2]) * anchor_w
    pred_boxes[:, :, :, :, 3] = np.exp(pred_boxes[:, :, :, :, 3]) * anchor_h

    return (
            pred_boxes.reshape((n, -1, 4)) * img_width / w, 
            pred_confs.reshape((n, -1)), 
            pred_classes.reshape((n, -1, class_num))
            )

def get_all_yolo_pred(outputs, yolo_anchors, yolo_classes, input_shape):    
    all_pred_boxes = []
    all_pred_confs = []
    all_pred_labels = []
    for output, anchors, classes in zip(outputs, yolo_anchors, yolo_classes):
        pred_boxes, pred_confs, pred_labels = get_yolo_detection(output, anchors, classes, input_shape[0], input_shape[1])
        preds = np.concatenate([pred_boxes, np.expand_dims(pred_confs, 2)], axis=2)
        # f = open("output{}.txt".format(index), 'w')
        # f.write(str(preds.shape) + "\n")
        # f.write(str(preds))
        # f.close()
        # index += 1
        all_pred_boxes.append(pred_boxes)
        all_pred_confs.append(pred_confs)
        all_pred_labels.append(pred_labels)
    pred_boxes = np.concatenate(all_pred_boxes, axis=1)
    pred_confs = np.concatenate(all_pred_confs, axis=1)
    pred_labels = np.concatenate(all_pred_labels, axis=1)

    return (pred_boxes, pred_confs, pred_labels)

def coco_anno_box_to_center_relative(box, img_width, img_height):
    """
    Convert COCO annotations box with format [x1, y1, w, h] to 
    center mode [center_x, center_y, w, h] and divide image width
    and height to get relative value in range[0, 1]
    """
    assert len(box) == 4, "box should be a len(4) list or tuple"
    x, y, w, h = box

    x1 = max(x, 0)
    x2 = min(x + w, img_width - 1)
    y1 = max(y, 0)
    y2 = min(y + h, img_height - 1)

    x = (x1 + x2) // 2 // (img_width - 1)
    y = (y1 + y2) // 2 // (img_height - 1)
    w = (x2 - x1) // (img_width - 1)
    h = (y2 - y1) // (img_height - 1)

    return np.array([x, y, w, h])

def clip_relative_box_in_image(x, y, w, h):
    """Clip relative box coordinates x, y, w, h to [0, 1]"""
    x1 = max(x - w / 2, 0.)
    x2 = min(x + w / 2, 1.)
    y1 = min(y - h / 2, 0.)
    y2 = max(y + h / 2, 1.)
    x = (x1 + x2) / 2
    y = (y1 + y2) / 2
    w = x2 - x1
    h = y2 - y1

def box_xywh_to_xyxy(box):
    shape = box.shape
    assert shape[-1] == 4, "Box shape[-1] should be 4."

    box = box.reshape((-1, 4))
    box[:, 0], box[:, 2] = box[:, 0] - box[:, 2] / 2, box[:, 0] + box[:, 2] / 2
    box[:, 1], box[:, 3] = box[:, 1] - box[:, 3] / 2, box[:, 1] + box[:, 3] / 2
    box = box.reshape(shape)
    return box

def box_iou_xywh(box1, box2):
    assert box1.shape[-1] == 4, "Box1 shape[-1] should be 4."
    assert box2.shape[-1] == 4, "Box2 shape[-1] should be 4."

    b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
    b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
    b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
    b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2

    inter_x1 = np.maximum(b1_x1, b2_x1)
    inter_x2 = np.minimum(b1_x2, b2_x2)
    inter_y1 = np.maximum(b1_y1, b2_y1)
    inter_y2 = np.maximum(b1_y2, b2_y2)
    inter_w = inter_x2 - inter_x1 + 1
    inter_h = inter_y2 - inter_y1 + 1
    inter_w[inter_w < 0] == 0
    inter_h[inter_h < 0] == 0

    inter_area = inter_w * inter_h
    b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)
    
    return inter_area / (b1_area + b2_area - inter_area)

def box_iou_xyxy(box1, box2):
    assert box1.shape[-1] == 4, "Box1 shape[-1] should be 4."
    assert box2.shape[-1] == 4, "Box2 shape[-1] should be 4."

    b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
    b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

    inter_x1 = np.maximum(b1_x1, b2_x1)
    inter_x2 = np.minimum(b1_x2, b2_x2)
    inter_y1 = np.maximum(b1_y1, b2_y1)
    inter_y2 = np.maximum(b1_y2, b2_y2)
    inter_w = inter_x2 - inter_x1 + 1
    inter_h = inter_y2 - inter_y1 + 1
    inter_w[inter_w < 0] == 0
    inter_h[inter_h < 0] == 0

    inter_area = inter_w * inter_h
    b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)
    
    return inter_area / (b1_area + b2_area - inter_area)

def rescale_box_in_input_image(boxes, im_shape, input_size):
    """Scale (x1, x2, y1, y2) box of yolo output to input image"""
    h, w = im_shape
    max_dim = max(h , w)
    boxes = boxes * max_dim / input_size
    dim_diff = np.abs(h - w)
    pad = dim_diff // 2
    if h <= w:
        boxes[:, 1] -= pad
        boxes[:, 3] -= pad
    else:
        boxes[:, 0] -= pad
        boxes[:, 2] -= pad
    boxes[boxes<0] == 0
    return boxes

def calc_nms_box(pred_boxes, pred_confs, pred_labels, im_shape, input_size, valid_thresh=0.8, nms_thresh=0.4, nms_topk=400, nms_posk=100):
    """
    Removes detections which confidence score under conf_thresh and perform 
    Non-Maximun Suppression to filtered boxes
    """
    _, box_num, class_num = pred_labels.shape
    pred_boxes = box_xywh_to_xyxy(pred_boxes)
    output_boxes = np.empty((0, 4))
    output_confs = np.empty(0)
    output_labels = np.empty((0))
    for i, (boxes, confs, classes) in enumerate(zip(pred_boxes, pred_confs, pred_labels)):
        conf_mask = confs > valid_thresh
        if conf_mask.sum() == 0:
            continue
        boxes = boxes[conf_mask]
        classes = classes[conf_mask]
        confs = confs[conf_mask]

        conf_sort_index = np.argsort(confs)[::-1]
        boxes = boxes[conf_sort_index][:nms_topk]
        classes = classes[conf_sort_index][:nms_topk]
        confs = confs[conf_sort_index][:nms_topk]
        cls_score = np.max(classes, axis=1)
        cls_pred = np.argmax(classes, axis=1)

        for c in np.unique(cls_pred):
            c_mask = cls_pred == c
            c_confs = confs[c_mask]
            c_boxes = boxes[c_mask]
            c_scores = cls_score[c_mask]
            c_score_index = np.argsort(c_scores)
            c_boxes_s = c_boxes[c_score_index[::-1]]
            c_confs_s = c_confs[c_score_index[::-1]]

            detect_boxes = []
            detect_confs = []
            detect_labels = []
            while c_boxes_s.shape[0]:
                detect_boxes.append(c_boxes_s[0])
                detect_confs.append(c_confs_s[0])
                detect_labels.append(c)
                if c_boxes_s.shape[0] == 1:
                    break
                iou = box_iou_xyxy(detect_boxes[-1].reshape((1, 4)), c_boxes_s[1:])
                c_boxes_s = c_boxes_s[1:][iou < nms_thresh]
                c_confs_s = c_confs_s[1:][iou < nms_thresh]

            output_boxes = np.append(output_boxes, detect_boxes, axis=0)
            output_confs = np.append(output_confs, detect_confs)
            output_labels = np.append(output_labels, detect_labels)
    
    output_boxes = output_boxes[:nms_posk]
    output_confs = output_confs[:nms_posk]
    output_labels = output_labels[:nms_posk]

    output_boxes = rescale_box_in_input_image(output_boxes, im_shape, input_size)
    return (output_boxes, output_confs, output_labels)

def draw_boxes_on_image(image_path, boxes, confs, labels, label_names, conf_thresh=0.5):
    image = Image.open(image_path)
    draw = ImageDraw.Draw(image)

    image_name = image_path.split('/')[-1]
    print("Image {} detect: ".format(image_name))
    for box, conf, label in zip(boxes, confs, labels):
        if conf < conf_thresh:
            continue
        x1, y1, x2, y2 = box[0], box[1], box[2], box[3]
        draw.rectangle((x1, y1, x2, y2), outline='red')
        if image.mode == 'RGB':
            draw.text((x1, y1), "{} {:.4f}".format(label_names[int(label)], conf), (255, 255, 0))
        print("\t {:15s} at {:25} score: {:.5f}".format(label_names[int(label)], map(int, list(box)), conf))
    image.save("./output/" + image_name)

