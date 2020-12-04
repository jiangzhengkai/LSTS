# --------------------------------------------------------
# Online Memory Networks
# Copyright (c) 2016 by Contributors
# Copyright (c) 2017 Microsoft
# Licensed under The Apache-2.0 License [see LICENSE for details]
# Modified by zhengkai jiang
# --------------------------------------------------------
import _init_paths
import matplotlib
matplotlib.use('Agg')

import cv2
import time
import argparse
import logging
import pprint
import os
import glob
import sys
from config.config import config, update_config
from utils.image import resize, transform
import numpy as np
# get config
os.environ['PYTHONUNBUFFERED'] = '1'
os.environ['MXNET_CUDNN_AUTOTUNE_DEFAULT'] = '0'
os.environ['MXNET_ENABLE_GPU_P2P'] = '0'
cur_path = os.path.abspath(os.path.dirname(__file__))
update_config(cur_path + '/../experiments/impression_rfcn/cfgs/impression_demo.yaml')

sys.path.insert(0, os.path.join(cur_path, '../external/mxnet', config.MXNET_VERSION))

import mxnet as mx
from core.tester import im_detect_impression_online, Predictor
from symbols import *
from utils.load_model import load_param
from utils.show_boxes import show_boxes, draw_boxes
from utils.tictoc import tic, toc
from nms.nms import py_nms_wrapper, cpu_nms_wrapper, gpu_nms_wrapper

def parse_args():
    parser = argparse.ArgumentParser(description='Show Online Memory Networks demo')
    args = parser.parse_args()
    return args

args = parse_args()

def main():
    # get symbol
    pprint.pprint(config)
    config.symbol = 'impression_network_dynamic_offset_sparse'
    model = '/../local_run_output/impression_dynamic_offset-lr-10000-times-neighbor-4-dense-4'
    first_sym_instance = eval(config.symbol + '.' + config.symbol)()
    key_sym_instance = eval(config.symbol + '.' + config.symbol)()
    cur_sym_instance = eval(config.symbol + '.' + config.symbol)()

    first_sym = first_sym_instance.get_first_test_symbol_impression(config)
    key_sym = key_sym_instance.get_key_test_symbol_impression(config)
    cur_sym = cur_sym_instance.get_cur_test_symbol_impression(config)

    # set up class names
    num_classes = 31
    classes = ['airplane', 'antelope', 'bear', 'bicycle',
               'bird', 'bus', 'car', 'cattle',
               'dog', 'domestic_cat', 'elephant', 'fox',
               'giant_panda', 'hamster', 'horse', 'lion',
               'lizard', 'monkey', 'motorcycle', 'rabbit',
               'red_panda', 'sheep', 'snake', 'squirrel',
               'tiger', 'train', 'turtle', 'watercraft',
               'whale', 'zebra']

    # load demo data
    image_names = glob.glob(cur_path + '/../demo/ILSVRC2015_val_00011005/*.JPEG')
    output_dir = cur_path + '/../demo/motion-prior-output-00011005/'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    key_frame_interval = 10
    image_names.sort()
    data = []
    for idx, im_name in enumerate(image_names):
        assert os.path.exists(im_name), ('%s does not exist'.format(im_name))
        im = cv2.imread(im_name, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
        target_size = config.SCALES[0][0]
        max_size = config.SCALES[0][1]
        im, im_scale = resize(im, target_size, max_size, stride=config.network.IMAGE_STRIDE)
        im_tensor = transform(im, config.network.PIXEL_MEANS)
        im_info = np.array([[im_tensor.shape[2], im_tensor.shape[3], im_scale]], dtype=np.float32)
        if idx % key_frame_interval == 0:
            if idx == 0:
                data_oldkey = im_tensor.copy()
                data_newkey = im_tensor.copy()
                data_cur = im_tensor.copy()
            else:
                data_oldkey = data_newkey.copy()
                data_newkey = im_tensor
        else:
            data_cur = im_tensor
        shape = im_tensor.shape
        infer_height = int(np.ceil(shape[2] / 16.0))
        infer_width = int(np.ceil(shape[3] / 16.0))
        data.append({'data_oldkey': data_oldkey, 'data_newkey': data_newkey, 'data_cur': data_cur, 'im_info': im_info,
                     'impression': np.zeros((1, config.network.DFF_FEAT_DIM, infer_height, infer_width)),
                     'key_feat_task': np.zeros((1, config.network.DFF_FEAT_DIM, infer_height, infer_width))})

    # get predictor
    data_names = ['data_oldkey', 'data_cur', 'data_newkey', 'im_info', 'impression', 'key_feat_task']
    label_names = []
    data = [[mx.nd.array(data[i][name]) for name in data_names] for i in xrange(len(data))]
    max_data_shape = [[('data_oldkey', (1, 3, max([v[0] for v in config.SCALES]), max([v[1] for v in config.SCALES]))),
                       ('data_newkey', (1, 3, max([v[0] for v in config.SCALES]), max([v[1] for v in config.SCALES]))),
                       ('data_cur', (1, 3, max([v[0] for v in config.SCALES]), max([v[1] for v in config.SCALES]))),
                       ('impression', (1, 1024, 38, 63)),
                       ('key_feat_task', (1, 1024, 38, 63))]]
    provide_data = [[(k, v.shape) for k, v in zip(data_names, data[i])] for i in xrange(len(data))]
    provide_label = [None for i in xrange(len(data))]
    arg_params, aux_params = load_param(cur_path + model, 4, process=True)
    first_predictor = Predictor(first_sym, data_names, label_names,
                              context=[mx.gpu(0)], max_data_shapes=max_data_shape,
                              provide_data=provide_data, provide_label=provide_label,
                              arg_params=arg_params, aux_params=aux_params)
    key_predictor = Predictor(key_sym, data_names, label_names,
                              context=[mx.gpu(0)], max_data_shapes=max_data_shape,
                              provide_data=provide_data, provide_label=provide_label,
                              arg_params=arg_params, aux_params=aux_params)
    cur_predictor = Predictor(cur_sym, data_names, label_names,
                              context=[mx.gpu(0)], max_data_shapes=max_data_shape,
                              provide_data=provide_data, provide_label=provide_label,
                              arg_params=arg_params, aux_params=aux_params)
    nms = gpu_nms_wrapper(config.TEST.NMS, 0)
    # warm up
    for j in xrange(2):
        data_batch = mx.io.DataBatch(data=[data[j]], label=[], pad=0, index=0,
                                     provide_data=[[(k, v.shape) for k, v in zip(data_names, data[j])]],
                                     provide_label=[None])
        scales = [data_batch.data[i][3].asnumpy()[0, 2] for i in xrange(len(data_batch.data))]
        if j % key_frame_interval == 0:  # keyframe
            if j == 0:  # first frame
                scores, boxes, data_dict, conv_feat, _, _, _ = im_detect_impression_online(first_predictor, data_batch, data_names, scales, config)
                feat_task = conv_feat
                impression = conv_feat
            else:  # keyframe
                data_batch.data[0][-2] = impression
                data_batch.provide_data[0][-2] = ('impression', impression.shape)
                scores, boxes, data_dict, conv_feat, impression, feat_task = im_detect_impression_online(key_predictor, data_batch, data_names, scales, config)
        else:  # current frame
            data_batch.data[0][-1] = feat_task
            data_batch.provide_data[0][-1] = ('key_feat_task', feat_task.shape)
            scores, boxes, data_dict, _, _, _, _ = im_detect_impression_online(cur_predictor, data_batch, data_names, scales, config)
    print "warmup done"
    # test
    time = 0
    count = 0
    for idx, im_name in enumerate(image_names):
        data_batch = mx.io.DataBatch(data=[data[idx]], label=[], pad=0, index=idx,
                                     provide_data=[[(k, v.shape) for k, v in zip(data_names, data[idx])]],
                                     provide_label=[None])
        scales = [data_batch.data[i][3].asnumpy()[0, 2] for i in xrange(len(data_batch.data))]
        tic()
        print(idx)
        if idx % key_frame_interval == 0:  # keyframe
            if idx == 0:  # first frame
                scores, boxes, data_dict, conv_feat, _, _, _ = im_detect_impression_online(first_predictor, data_batch, data_names, scales, config)
                feat_task = conv_feat
                impression = conv_feat
                feat_task_numpy = feat_task.asnumpy()
                np.save("features/impression_%s.npy" % (idx), feat_task_numpy)
            else:  # keyframe
                data_batch.data[0][-2] = impression
                data_batch.provide_data[0][-2] = ('impression', impression.shape)

                scores, boxes, data_dict, conv_feat, impression, feat_task, _ = im_detect_impression_online(key_predictor, data_batch, data_names, scales, config)
                feat_task_key_numpy = feat_task.asnumpy()
                np.save("features/impression_%s.npy" % (idx), feat_task_key_numpy)
        else:  # current frame
            data_batch.data[0][-1] = feat_task
            data_batch.provide_data[0][-1] = ('key_feat_task', feat_task.shape)
            scores, boxes, data_dict, _, _, _, feat_task_cur = im_detect_impression_online(cur_predictor, data_batch, data_names, scales, config)
            if idx >= 1:
                feat_task_cur_numpy = feat_task_cur.asnumpy()
                np.save("features/impression_%s.npy"%(idx),feat_task_cur_numpy)
                #import pdb;pdb.set_trace()
        time += toc()
        count += 1
        print 'testing {} {:.4f}s'.format(im_name, time/count)
        boxes = boxes[0].astype('f')
        scores = scores[0].astype('f')
        dets_nms = []
        for j in range(1, scores.shape[1]):
            cls_scores = scores[:, j, np.newaxis]
            cls_boxes = boxes[:, 4:8] if config.CLASS_AGNOSTIC else boxes[:, j * 4:(j + 1) * 4]
            cls_dets = np.hstack((cls_boxes, cls_scores))
            keep = nms(cls_dets)
            cls_dets = cls_dets[keep, :]
            cls_dets = cls_dets[cls_dets[:, -1] > 0.7, :]
            dets_nms.append(cls_dets)
        # visualize
        im = cv2.imread(im_name)
        #im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        # show_boxes(im, dets_nms, classes, 1)
        out_im = draw_boxes(im, dets_nms, classes, 1)
        _, filename = os.path.split(im_name)
        cv2.imwrite(output_dir + filename,out_im)
    print 'done'

if __name__ == '__main__':
    main()
