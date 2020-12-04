# --------------------------------------------------------------
# Online Memory Networks
# Copyright (c) 2017 by Contributors
# Copyright (c) 2018 Microsoft
# Licensed under The Apache-2.0 License [see LICENSE for details]
# Modified by Zhengkai Jiang
# ---------------------------------------------------------------

import numpy as np
import mxnet as mx
from mxnet.executor_manager import _split_input_slice

from config.config import config
from utils.image import tensor_vstack
from rpn.rpn import get_rpn_testbatch, get_rpn_pair_batch, get_rpn_triple_batch_online, get_rpn_triple_batch_fromrec_online, get_rpn_triple_batch_offline, get_rpn_triple_batch_fromrec_offline,  get_online_impression_testbatch, get_offline_impression_testbatch, assign_anchor
from rcnn import get_rcnn_testbatch, get_rcnn_batch
class TestLoader_Impression_Online(mx.io.DataIter):
    def __init__(self, roidb, config, batch_size=1, shuffle=False,
                 has_rpn=False, from_rec = False):
        super(TestLoader_Impression_Online, self).__init__()
        # save parameters as properties
        self.cfg = config
        self.roidb = roidb
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.has_rpn = has_rpn

        # infer properties from roidb
        self.size = np.sum([x['frame_seg_len'] for x in self.roidb])
        self.index = np.arange(self.size)
        self.from_rec = from_rec
        if self.from_rec:
            self.rec = mx.recordio.MXIndexedRecordIO(config.dataset.rec_idx, config.dataset.rec_data, 'r')
            with open(config.dataset.video_index_list, 'r') as f:
                video_index_dict = {}
                for line in f.readlines():
                    video_index_dict[line.split(' ')[0]] = int(line.split(' ')[1])
            self.video_index_dict = video_index_dict
        else:
            self.video_index_dict = None
            self.rec = None
        # decide data and label names (only for training)
        self.data_name = ['data_oldkey', 'data_cur', 'data_newkey', 'im_info', 'impression', 'key_feat_task']
        self.label_name = None

        #
        self.cur_roidb_index = 0
        self.cur_frameid = 0
        self.key_frameid = 0
        self.cur_seg_len = 0
        self.key_frame_flag = -1

        # status variable for synchronization between get_data and get_label
        self.cur = 0
        self.data = None
        self.label = []
        self.im_info = None

        # get first batch to fill in provide_data and provide_label
        self.reset()
        self.get_batch_impression_online()

    @property
    def provide_data(self):
        return [[(k, v.shape) for k, v in zip(self.data_name, idata)] for idata in self.data]

    @property
    def provide_label(self):
        return [None for _ in range(len(self.data))]

    @property
    def provide_data_single(self):
        return [(k, v.shape) for k, v in zip(self.data_name, self.data[0])]

    @property
    def provide_label_single(self):
        return None

    def reset(self):
        self.cur = 0
        if self.shuffle:
            np.random.shuffle(self.index)

    def iter_next(self):
        return self.cur < self.size

    def next(self):
        if self.iter_next():
            self.get_batch_impression_online()
            self.cur += self.batch_size
            self.cur_frameid += 1
            if self.cur_frameid == self.cur_seg_len:
                self.cur_roidb_index += 1
                self.cur_frameid = 0
                self.key_frameid = 0
            elif self.cur_frameid - self.key_frameid == self.cfg.TEST.KEY_FRAME_INTERVAL:
                self.key_frameid = self.cur_frameid
            return self.im_info, self.key_frame_flag, mx.io.DataBatch(data=self.data, label=self.label,
                                   pad=self.getpad(), index=self.getindex(),
                                   provide_data=self.provide_data, provide_label=self.provide_label)
        else:
            raise StopIteration

    def getindex(self):
        return self.cur / self.batch_size

    def getpad(self):
        if self.cur + self.batch_size > self.size:
            return self.cur + self.batch_size - self.size
        else:
            return 0

    def get_batch_impression_online(self):
        cur_roidb = self.roidb[self.cur_roidb_index].copy()
        cur_roidb['image'] = cur_roidb['pattern'] % self.cur_frameid
        cur_roidb['frame_id'] = self.cur_frameid
        self.cur_seg_len = cur_roidb['frame_seg_len']
        data, label, im_info = get_online_impression_testbatch([cur_roidb], self.cfg, video_index_dict=self.video_index_dict, rec=self.rec)
        if self.cur_frameid == self.key_frameid:
            if self.cur_frameid == 0:
                self.data_oldkey = data[0]['data'].copy()
                self.data_newkey = data[0]['data'].copy()
                self.data_cur = data[0]['data'].copy()
                self.key_frame_flag = 0
            else:
                self.data_oldkey = self.data_newkey.copy()
                self.data_newkey = data[0]['data'].copy()
                self.key_frame_flag = 1
        else:
            self.data_cur = data[0]['data'].copy()
            self.key_frame_flag = 2
        # get shape of the new video
        shape = self.data_newkey.shape
        infer_height = int(np.ceil(shape[2]/16.0))
        infer_width = int(np.ceil(shape[3]/16.0))

        extend_data = [{'data_oldkey': self.data_oldkey,
                        'data_newkey': self.data_newkey,
                        'data_cur': self.data_cur,
                        'im_info': data[0]['im_info'],
			            'impression': np.zeros((1,self.cfg.network.DFF_FEAT_DIM,infer_height,infer_width)),
			            'key_feat_task': np.zeros((1,self.cfg.network.DFF_FEAT_DIM,infer_height,infer_width))}]
        self.data = [[mx.nd.array(extend_data[i][name]) for name in self.data_name] for i in xrange(len(data))]
        self.im_info = im_info


class TestLoader_Impression_Offline(mx.io.DataIter):
    def __init__(self, roidb, config, batch_size=1, shuffle=False,
                 has_rpn=False, from_rec = False):
        super(TestLoader_Impression_Offline, self).__init__()
        # save parameters as properties
        self.cfg = config
        self.roidb = roidb
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.has_rpn = has_rpn

        # infer properties from roidb
        self.size = np.sum([x['frame_seg_len'] for x in self.roidb])
        self.index = np.arange(self.size)

        self.from_rec = from_rec
        if self.from_rec:
            self.rec = mx.recordio.MXIndexedRecordIO(config.dataset.rec_idx, config.dataset.rec_data, 'r')
            with open(config.dataset.video_index_list, 'r') as f:
                video_index_dict = {}
                for line in f.readlines():
                    video_index_dict[line.split(' ')[0]] = int(line.split(' ')[1])
            self.video_index_dict = video_index_dict
        else:
            self.video_index_dict = None
            self.rec = None
        # decide data and label names (only for training)
        self.data_name = ['data_oldkey', 'data_cur', 'data_newkey', 'im_info', 'impression', 'key_feat_task']
        self.label_name = None

        #
        self.cur_roidb_index = 0
        self.cur_frameid = 0
        self.key_frameid = 0
        self.cur_seg_len = 0
        self.cur_seg_index = 0
        self.key_frame_flag = -1
        self.key_frame_list = []
        self.key_frame_changed = True

        # status variable for synchronization between get_data and get_label
        self.cur = 0
        self.data = None
        self.label = []
        self.im_info = None

        # get first batch to fill in provide_data and provide_label
        self.reset()
        self.get_batch_impression_offline()

    @property
    def provide_data(self):
        return [[(k, v.shape) for k, v in zip(self.data_name, idata)] for idata in self.data]

    @property
    def provide_label(self):
        return [None for _ in range(len(self.data))]

    @property
    def provide_data_single(self):
        return [(k, v.shape) for k, v in zip(self.data_name, self.data[0])]

    @property
    def provide_label_single(self):
        return None

    def reset(self):
        self.cur = 0
        if self.shuffle:
            np.random.shuffle(self.index)

    def iter_next(self):
        return self.cur < self.size

    def next(self):
        if self.iter_next():
            self.get_batch_impression_offline()
            self.cur += self.batch_size
            self.cur_frameid += 1
            if self.cur_frameid == self.cur_seg_len:
                self.cur_roidb_index += 1
                self.cur_frameid = 0
                self.cur_seg_index = 0
                self.key_frame_changed = True
            elif self.cur_frameid - self.key_frame_list[self.cur_seg_index] > 1/2.* self.cfg.TEST.KEY_FRAME_INTERVAL:
                self.cur_seg_index += 1
                self.key_frame_changed = True
            else:
                self.key_frame_changed = False
            return self.im_info, self.key_frame_flag, mx.io.DataBatch(data=self.data, label=self.label,
                                   pad=self.getpad(), index=self.getindex(),
                                   provide_data=self.provide_data, provide_label=self.provide_label)
        else:
            raise StopIteration

    def getindex(self):
        return self.cur / self.batch_size

    def getpad(self):
        if self.cur + self.batch_size > self.size:
            return self.cur + self.batch_size - self.size
        else:
            return 0

    def get_batch_impression_offline(self):
        cur_roidb = self.roidb[self.cur_roidb_index].copy()
        cur_roidb['image'] = cur_roidb['pattern'] % self.cur_frameid
        self.cur_seg_len = cur_roidb['frame_seg_len']
        num_segment = int(np.ceil(float(self.cur_seg_len) / self.cfg.TEST.KEY_FRAME_INTERVAL))
        self.key_frame_list = [self.cfg.TEST.KEY_FRAME_INTERVAL / 2 - 1 + i*self.cfg.TEST.KEY_FRAME_INTERVAL
			                    if self.cfg.TEST.KEY_FRAME_INTERVAL / 2 -1 + i*self.cfg.TEST.KEY_FRAME_INTERVAL < self.cur_seg_len
			                    else self.cur_seg_len -1 for i in range(num_segment)]
        cur_roidb['data_cur'] = cur_roidb['pattern'] % self.cur_frameid
        cur_roidb['data_newkey'] = cur_roidb['pattern'] % self.key_frame_list[self.cur_seg_index]
        cur_roidb['key_frame_changed'] = self.key_frame_changed
        cur_roidb['from_rec'] = self.from_rec
        cur_roidb['frame_id'] = self.cur_frameid
        cur_roidb['newkey_id'] = self.key_frame_list[self.cur_seg_index]
        data, label, im_info = get_offline_impression_testbatch([cur_roidb], self.cfg, video_index_dict=self.video_index_dict, rec=self.rec)
        if self.cur_frameid == 0: # first frame
            # print 'current video is ', cur_roidb['pattern']
            self.data_oldkey = data[0]['data_newkey'].copy()
            self.data_newkey = data[0]['data_newkey'].copy()
            self.data_cur = data[0]['data_cur'].copy()
        elif self.cur_seg_index > 0 and self.key_frame_changed: # update newkey
            self.data_oldkey = self.data_newkey.copy()
            self.data_newkey = data[0]['data_newkey'].copy()
            self.data_cur = data[0]['data_cur'].copy()
        else:	# current key
            self.data_cur = data[0]['data_cur'].copy()
        if self.cur_frameid == 0: # first frame of the video
            self.key_frame_flag = 0
        elif self.cur_frameid == self.key_frame_list[self.cur_seg_index]:
            self.key_frame_flag = 1 # keyframe of the video
        elif self.key_frame_changed and self.cur_frameid != 0:
            self.key_frame_flag = 2 # first frame of the new segment
        else:
            self.key_frame_flag = 3 # current frame
        # get shape of the new video
        shape = self.data_newkey.shape
        infer_height = int(np.ceil(shape[2]/16.0))
        infer_width = int(np.ceil(shape[3]/16.0))
        extend_data = [{'data_oldkey': self.data_oldkey,
                        'data_newkey': self.data_newkey,
                        'data_cur': self.data_cur,
                        'im_info': data[0]['im_info'],
			            'impression': np.zeros((1,self.cfg.network.DFF_FEAT_DIM,infer_height,infer_width)),
			            'key_feat_task': np.zeros((1,self.cfg.network.DFF_FEAT_DIM,infer_height,infer_width))}]
        self.data = [[mx.nd.array(extend_data[i][name]) for name in self.data_name] for i in xrange(len(data))]
        self.im_info = im_info

class AnchorLoaderOnline(mx.io.DataIter):
    def __init__(self, feat_sym, roidb, cfg, video_index_dict=None, rec_file=None, from_rec=False, batch_size=1, shuffle=False, ctx=None, work_load_list=None,
                 feat_stride=16, anchor_scales=(8, 16, 32), anchor_ratios=(0.5, 1, 2), allowed_border=0,
                 aspect_grouping=False, normalize_target=False, bbox_mean=(0.0, 0.0, 0.0, 0.0),
                 bbox_std=(0.1, 0.1, 0.4, 0.4)):
        """
        This Iter will provide roi data to Fast R-CNN network
        :param feat_sym: to infer shape of assign_output
        :param roidb: must be preprocessed
        :param video_index_dict: a dict of video corresponding to index in record file
        :param rec_file: record file
        :param batch_size: must divide BATCH_SIZE(128)
        :param shuffle: bool
        :param ctx: list of contexts
        :param work_load_list: list of work load
        :param aspect_grouping: group images with similar aspects
        :param normalize_target: normalize rpn target
        :param bbox_mean: anchor target mean
        :param bbox_std: anchor target std
        :return: AnchorLoader
        """
        super(AnchorLoaderOnline, self).__init__()

        # save parameters as properties
        self.feat_sym = feat_sym
        self.roidb = roidb
        self.rec = rec_file
        self.video_index_dict = video_index_dict
        self.from_rec = from_rec
        self.cfg = cfg
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.ctx = ctx
        if self.ctx is None:
            self.ctx = [mx.cpu()]
        self.work_load_list = work_load_list
        self.feat_stride = feat_stride
        self.anchor_scales = anchor_scales
        self.anchor_ratios = anchor_ratios
        self.allowed_border = allowed_border
        self.aspect_grouping = aspect_grouping
        self.normalize_target = normalize_target
        self.bbox_mean = bbox_mean
        self.bbox_std = bbox_std

        # infer properties from roidb
        self.size = len(roidb)
        self.index = np.arange(self.size)

        # decide data and label names
        if config.TRAIN.END2END:
            self.data_name = ['data_oldkey', 'data_cur', 'data_newkey', 'eq_flag_key2key', 'eq_flag_key2cur', 'im_info', 'gt_boxes']
        else:
            self.data_name = ['data']
        self.label_name = ['label', 'bbox_target', 'bbox_weight']

        # status variable for synchronization between get_data and get_label
        self.cur = 0
        self.batch = None
        self.data = None
        self.label = None

        # get first batch to fill in provide_data and provide_label
        self.reset()
        self.get_batch_individual()

    @property
    def provide_data(self):
        return [[(k, v.shape) for k, v in zip(self.data_name, self.data[i])] for i in xrange(len(self.data))]

    @property
    def provide_label(self):
        return [[(k, v.shape) for k, v in zip(self.label_name, self.label[i])] for i in xrange(len(self.data))]

    @property
    def provide_data_single(self):
        return [(k, v.shape) for k, v in zip(self.data_name, self.data[0])]

    @property
    def provide_label_single(self):
        return [(k, v.shape) for k, v in zip(self.label_name, self.label[0])]

    def reset(self):
        self.cur = 0
        if self.shuffle:
            if self.aspect_grouping:        #ensure every batch of data with similar aspect
                widths = np.array([r['width'] for r in self.roidb])
                heights = np.array([r['height'] for r in self.roidb])
                horz = (widths >= heights)
                vert = np.logical_not(horz)
                horz_inds = np.where(horz)[0]
                vert_inds = np.where(vert)[0]
                inds = np.hstack((np.random.permutation(horz_inds), np.random.permutation(vert_inds)))
                extra = inds.shape[0] % self.batch_size
                inds_ = np.reshape(inds[:-extra], (-1, self.batch_size))
                row_perm = np.random.permutation(np.arange(inds_.shape[0]))
                inds[:-extra] = np.reshape(inds_[row_perm, :], (-1,))
                self.index = inds
            else:
                np.random.shuffle(self.index)

    def iter_next(self):
        return self.cur + self.batch_size <= self.size

    def next(self):
        if self.iter_next():
            self.get_batch_individual()
            self.cur += self.batch_size
            return mx.io.DataBatch(data=self.data, label=self.label,
                                   pad=self.getpad(), index=self.getindex(),
                                   provide_data=self.provide_data, provide_label=self.provide_label)
        else:
            raise StopIteration

    def getindex(self):
        return self.cur / self.batch_size

    def getpad(self):
        if self.cur + self.batch_size > self.size:
            return self.cur + self.batch_size - self.size
        else:
            return 0

    def infer_shape(self, max_data_shape=None, max_label_shape=None):
        """ Return maximum data and label shape for single gpu """
        if max_data_shape is None:
            max_data_shape = []
        if max_label_shape is None:
            max_label_shape = []
        max_shapes = dict(max_data_shape + max_label_shape)
        input_batch_size = max_shapes['data_cur'][0]
        im_info = [[max_shapes['data_cur'][2], max_shapes['data_cur'][3], 1.0]]
        _, feat_shape, _ = self.feat_sym.infer_shape(**max_shapes)
        label = assign_anchor(feat_shape[0], np.zeros((0, 5)), im_info, self.cfg,
                              self.feat_stride, self.anchor_scales, self.anchor_ratios, self.allowed_border,
                              self.normalize_target, self.bbox_mean, self.bbox_std)
        label = [label[k] for k in self.label_name]
        label_shape = [(k, tuple([input_batch_size] + list(v.shape[1:]))) for k, v in zip(self.label_name, label)]
        return max_data_shape, label_shape

    def get_batch(self):
        # slice roidb
        cur_from = self.cur
        cur_to = min(cur_from + self.batch_size, self.size)
        roidb = [self.roidb[self.index[i]] for i in range(cur_from, cur_to)]

        # decide multi device slice
        work_load_list = self.work_load_list
        ctx = self.ctx
        if work_load_list is None:
            work_load_list = [1] * len(ctx)
        assert isinstance(work_load_list, list) and len(work_load_list) == len(ctx), \
            "Invalid settings for work load. "
        slices = _split_input_slice(self.batch_size, work_load_list)

        # get testing data for multigpu
        data_list = []
        label_list = []
        for islice in slices:
            iroidb = [roidb[i] for i in range(islice.start, islice.stop)]
            data, label = get_rpn_pair_batch(iroidb, self.cfg)
            data_list.append(data)
            label_list.append(label)

        # pad data first and then assign anchor (read label)
        data_tensor = tensor_vstack([batch['data'] for batch in data_list])
        for data, data_pad in zip(data_list, data_tensor):
            data['data'] = data_pad[np.newaxis, :]

        new_label_list = []
        for data, label in zip(data_list, label_list):
            # infer label shape
            data_shape = {k: v.shape for k, v in data.items()}
            del data_shape['im_info']
            _, feat_shape, _ = self.feat_sym.infer_shape(**data_shape)
            feat_shape = [int(i) for i in feat_shape[0]]

            # add gt_boxes to data for e2e
            data['gt_boxes'] = label['gt_boxes'][np.newaxis, :, :]

            # assign anchor for label
            label = assign_anchor(feat_shape, label['gt_boxes'], data['im_info'], self.cfg,
                                  self.feat_stride, self.anchor_scales,
                                  self.anchor_ratios, self.allowed_border,
                                  self.normalize_target, self.bbox_mean, self.bbox_std)
            new_label_list.append(label)

        all_data = dict()
        for key in self.data_name:
            all_data[key] = tensor_vstack([batch[key] for batch in data_list])

        all_label = dict()
        for key in self.label_name:
            pad = -1 if key == 'label' else 0
            all_label[key] = tensor_vstack([batch[key] for batch in new_label_list], pad=pad)

        self.data = [mx.nd.array(all_data[key]) for key in self.data_name]
        self.label = [mx.nd.array(all_label[key]) for key in self.label_name]

    def get_batch_individual(self):
        cur_from = self.cur
        cur_to = min(cur_from + self.batch_size, self.size)
        roidb = [self.roidb[self.index[i]] for i in range(cur_from, cur_to)]
        # decide multi device slice
        work_load_list = self.work_load_list
        ctx = self.ctx
        if work_load_list is None:
            work_load_list = [1] * len(ctx)
        assert isinstance(work_load_list, list) and len(work_load_list) == len(ctx), \
            "Invalid settings for work load. "
        slices = _split_input_slice(self.batch_size, work_load_list)
        rst = []
        for idx, islice in enumerate(slices):
            iroidb = [roidb[i] for i in range(islice.start, islice.stop)]
            if self.from_rec:
                rst.append(self.parfetch_online_memory_fromrec(iroidb))
            else:
                rst.append(self.parfetch_online_memory(iroidb))
        all_data = [_['data'] for _ in rst]
        all_label = [_['label'] for _ in rst]
        self.data = [[mx.nd.array(data[key]) for key in self.data_name] for data in all_data]
        self.label = [[mx.nd.array(label[key]) for key in self.label_name] for label in all_label]

    def parfetch(self, iroidb):
        # get testing data for multigpu
        data, label = get_rpn_pair_batch(iroidb, self.cfg)
        data_shape = {k: v.shape for k, v in data.items()}
        del data_shape['im_info']
        _, feat_shape, _ = self.feat_sym.infer_shape(**data_shape)
        feat_shape = [int(i) for i in feat_shape[0]]

        # add gt_boxes to data for e2e
        data['gt_boxes'] = label['gt_boxes'][np.newaxis, :, :]

        # assign anchor for label
        label = assign_anchor(feat_shape, label['gt_boxes'], data['im_info'], self.cfg,
                              self.feat_stride, self.anchor_scales,
                              self.anchor_ratios, self.allowed_border,
                              self.normalize_target, self.bbox_mean, self.bbox_std)
        return {'data': data, 'label': label}

    def parfetch_online_memory(self, iroidb):
        # get testing data for multigpu
        data, label = get_rpn_triple_batch(iroidb, self.cfg)
        data_shape = {k: v.shape for k, v in data.items()}
        del data_shape['im_info']
        _, feat_shape, _ = self.feat_sym.infer_shape(**data_shape)
        feat_shape = [int(i) for i in feat_shape[0]]

        # add gt_boxes to data for e2e
        data['gt_boxes'] = label['gt_boxes'][np.newaxis, :, :]

        # assign anchor for label
        label = assign_anchor(feat_shape, label['gt_boxes'], data['im_info'], self.cfg,
                              self.feat_stride, self.anchor_scales,
                              self.anchor_ratios, self.allowed_border,
                              self.normalize_target, self.bbox_mean, self.bbox_std)
        return {'data': data, 'label': label}

    def parfetch_online_memory_fromrec(self, iroidb):
        # get testing data for multigpu
        data, label = get_rpn_triple_batch_fromrec(iroidb, self.cfg, self.video_index_dict, self.rec)
        data_shape = {k: v.shape for k, v in data.items()}
        del data_shape['im_info']
        _, feat_shape, _ = self.feat_sym.infer_shape(**data_shape)
        feat_shape = [int(i) for i in feat_shape[0]]

        # add gt_boxes to data for e2e
        data['gt_boxes'] = label['gt_boxes'][np.newaxis, :, :]

        # assign anchor for label
        label = assign_anchor(feat_shape, label['gt_boxes'], data['im_info'], self.cfg,
                              self.feat_stride, self.anchor_scales,
                              self.anchor_ratios, self.allowed_border,
                              self.normalize_target, self.bbox_mean, self.bbox_std)
        return {'data': data, 'label': label}

class AnchorLoaderOffline(mx.io.DataIter):
    def __init__(self, feat_sym, roidb, cfg, video_index_dict=None, rec_file=None, from_rec=False, batch_size=1, shuffle=False, ctx=None, work_load_list=None,
                 feat_stride=16, anchor_scales=(8, 16, 32), anchor_ratios=(0.5, 1, 2), allowed_border=0,
                 aspect_grouping=False, normalize_target=False, bbox_mean=(0.0, 0.0, 0.0, 0.0),
                 bbox_std=(0.1, 0.1, 0.4, 0.4)):
        """
        This Iter will provide roi data to Fast R-CNN network
        :param feat_sym: to infer shape of assign_output
        :param roidb: must be preprocessed
        :param video_index_dict: a dict of video corresponding to index in record file
        :param rec_file: record file
        :param batch_size: must divide BATCH_SIZE(128)
        :param shuffle: bool
        :param ctx: list of contexts
        :param work_load_list: list of work load
        :param aspect_grouping: group images with similar aspects
        :param normalize_target: normalize rpn target
        :param bbox_mean: anchor target mean
        :param bbox_std: anchor target std
        :return: AnchorLoader
        """
        super(AnchorLoaderOffline, self).__init__()

        # save parameters as properties
        self.feat_sym = feat_sym
        self.roidb = roidb
        self.rec = rec_file
        self.video_index_dict = video_index_dict
        self.from_rec = from_rec
        self.cfg = cfg
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.ctx = ctx
        if self.ctx is None:
            self.ctx = [mx.cpu()]
        self.work_load_list = work_load_list
        self.feat_stride = feat_stride
        self.anchor_scales = anchor_scales
        self.anchor_ratios = anchor_ratios
        self.allowed_border = allowed_border
        self.aspect_grouping = aspect_grouping
        self.normalize_target = normalize_target
        self.bbox_mean = bbox_mean
        self.bbox_std = bbox_std

        # infer properties from roidb
        self.size = len(roidb)
        self.index = np.arange(self.size)

        # decide data and label names
        if config.TRAIN.END2END:
            self.data_name = ['data_oldkey', 'data_cur', 'data_newkey', 'eq_flag_key2key', 'eq_flag_key2cur', 'im_info', 'gt_boxes']
        else:
            self.data_name = ['data']
        self.label_name = ['label', 'bbox_target', 'bbox_weight']

        # status variable for synchronization between get_data and get_label
        self.cur = 0
        self.batch = None
        self.data = None
        self.label = None

        # get first batch to fill in provide_data and provide_label
        self.reset()
        self.get_batch_individual()

    @property
    def provide_data(self):
        return [[(k, v.shape) for k, v in zip(self.data_name, self.data[i])] for i in xrange(len(self.data))]

    @property
    def provide_label(self):
        return [[(k, v.shape) for k, v in zip(self.label_name, self.label[i])] for i in xrange(len(self.data))]

    @property
    def provide_data_single(self):
        return [(k, v.shape) for k, v in zip(self.data_name, self.data[0])]

    @property
    def provide_label_single(self):
        return [(k, v.shape) for k, v in zip(self.label_name, self.label[0])]

    def reset(self):
        self.cur = 0
        if self.shuffle:
            if self.aspect_grouping:        #ensure every batch of data with similar aspect
                widths = np.array([r['width'] for r in self.roidb])
                heights = np.array([r['height'] for r in self.roidb])
                horz = (widths >= heights)
                vert = np.logical_not(horz)
                horz_inds = np.where(horz)[0]
                vert_inds = np.where(vert)[0]
                inds = np.hstack((np.random.permutation(horz_inds), np.random.permutation(vert_inds)))
                extra = inds.shape[0] % self.batch_size
                inds_ = np.reshape(inds[:-extra], (-1, self.batch_size))
                row_perm = np.random.permutation(np.arange(inds_.shape[0]))
                inds[:-extra] = np.reshape(inds_[row_perm, :], (-1,))
                self.index = inds
            else:
                np.random.shuffle(self.index)

    def iter_next(self):
        return self.cur + self.batch_size <= self.size

    def next(self):
        if self.iter_next():
            self.get_batch_individual()
            self.cur += self.batch_size
            return mx.io.DataBatch(data=self.data, label=self.label,
                                   pad=self.getpad(), index=self.getindex(),
                                   provide_data=self.provide_data, provide_label=self.provide_label)
        else:
            raise StopIteration

    def getindex(self):
        return self.cur / self.batch_size

    def getpad(self):
        if self.cur + self.batch_size > self.size:
            return self.cur + self.batch_size - self.size
        else:
            return 0

    def infer_shape(self, max_data_shape=None, max_label_shape=None):
        """ Return maximum data and label shape for single gpu """
        if max_data_shape is None:
            max_data_shape = []
        if max_label_shape is None:
            max_label_shape = []
        max_shapes = dict(max_data_shape + max_label_shape)
        input_batch_size = max_shapes['data_cur'][0]
        im_info = [[max_shapes['data_cur'][2], max_shapes['data_cur'][3], 1.0]]
        _, feat_shape, _ = self.feat_sym.infer_shape(**max_shapes)
        label = assign_anchor(feat_shape[0], np.zeros((0, 5)), im_info, self.cfg,
                              self.feat_stride, self.anchor_scales, self.anchor_ratios, self.allowed_border,
                              self.normalize_target, self.bbox_mean, self.bbox_std)
        label = [label[k] for k in self.label_name]
        label_shape = [(k, tuple([input_batch_size] + list(v.shape[1:]))) for k, v in zip(self.label_name, label)]
        return max_data_shape, label_shape

    def get_batch(self):
        # slice roidb
        cur_from = self.cur
        cur_to = min(cur_from + self.batch_size, self.size)
        roidb = [self.roidb[self.index[i]] for i in range(cur_from, cur_to)]

        # decide multi device slice
        work_load_list = self.work_load_list
        ctx = self.ctx
        if work_load_list is None:
            work_load_list = [1] * len(ctx)
        assert isinstance(work_load_list, list) and len(work_load_list) == len(ctx), \
            "Invalid settings for work load. "
        slices = _split_input_slice(self.batch_size, work_load_list)

        # get testing data for multigpu
        data_list = []
        label_list = []
        for islice in slices:
            iroidb = [roidb[i] for i in range(islice.start, islice.stop)]
            data, label = get_rpn_pair_batch(iroidb, self.cfg)
            data_list.append(data)
            label_list.append(label)

        # pad data first and then assign anchor (read label)
        data_tensor = tensor_vstack([batch['data'] for batch in data_list])
        for data, data_pad in zip(data_list, data_tensor):
            data['data'] = data_pad[np.newaxis, :]

        new_label_list = []
        for data, label in zip(data_list, label_list):
            # infer label shape
            data_shape = {k: v.shape for k, v in data.items()}
            del data_shape['im_info']
            _, feat_shape, _ = self.feat_sym.infer_shape(**data_shape)
            feat_shape = [int(i) for i in feat_shape[0]]

            # add gt_boxes to data for e2e
            data['gt_boxes'] = label['gt_boxes'][np.newaxis, :, :]

            # assign anchor for label
            label = assign_anchor(feat_shape, label['gt_boxes'], data['im_info'], self.cfg,
                                  self.feat_stride, self.anchor_scales,
                                  self.anchor_ratios, self.allowed_border,
                                  self.normalize_target, self.bbox_mean, self.bbox_std)
            new_label_list.append(label)

        all_data = dict()
        for key in self.data_name:
            all_data[key] = tensor_vstack([batch[key] for batch in data_list])

        all_label = dict()
        for key in self.label_name:
            pad = -1 if key == 'label' else 0
            all_label[key] = tensor_vstack([batch[key] for batch in new_label_list], pad=pad)

        self.data = [mx.nd.array(all_data[key]) for key in self.data_name]
        self.label = [mx.nd.array(all_label[key]) for key in self.label_name]

    def get_batch_individual(self):
        cur_from = self.cur
        cur_to = min(cur_from + self.batch_size, self.size)
        roidb = [self.roidb[self.index[i]] for i in range(cur_from, cur_to)]
        # decide multi device slice
        work_load_list = self.work_load_list
        ctx = self.ctx
        if work_load_list is None:
            work_load_list = [1] * len(ctx)
        assert isinstance(work_load_list, list) and len(work_load_list) == len(ctx), \
            "Invalid settings for work load. "
        slices = _split_input_slice(self.batch_size, work_load_list)
        rst = []
        for idx, islice in enumerate(slices):
            iroidb = [roidb[i] for i in range(islice.start, islice.stop)]
            if self.from_rec:
                rst.append(self.parfetch_offline_memory_fromrec(iroidb))
            else:
                rst.append(self.parfetch_offline_memory(iroidb))
        all_data = [_['data'] for _ in rst]
        all_label = [_['label'] for _ in rst]
        self.data = [[mx.nd.array(data[key]) for key in self.data_name] for data in all_data]
        self.label = [[mx.nd.array(label[key]) for key in self.label_name] for label in all_label]

    def parfetch(self, iroidb):
        # get testing data for multigpu
        data, label = get_rpn_pair_batch(iroidb, self.cfg)
        data_shape = {k: v.shape for k, v in data.items()}
        del data_shape['im_info']
        _, feat_shape, _ = self.feat_sym.infer_shape(**data_shape)
        feat_shape = [int(i) for i in feat_shape[0]]

        # add gt_boxes to data for e2e
        data['gt_boxes'] = label['gt_boxes'][np.newaxis, :, :]

        # assign anchor for label
        label = assign_anchor(feat_shape, label['gt_boxes'], data['im_info'], self.cfg,
                              self.feat_stride, self.anchor_scales,
                              self.anchor_ratios, self.allowed_border,
                              self.normalize_target, self.bbox_mean, self.bbox_std)
        return {'data': data, 'label': label}

    def parfetch_offline_memory(self, iroidb):
        # get testing data for multigpu
        data, label = get_rpn_triple_batch_offline(iroidb, self.cfg)
        data_shape = {k: v.shape for k, v in data.items()}
        del data_shape['im_info']
        _, feat_shape, _ = self.feat_sym.infer_shape(**data_shape)
        feat_shape = [int(i) for i in feat_shape[0]]

        # add gt_boxes to data for e2e
        data['gt_boxes'] = label['gt_boxes'][np.newaxis, :, :]

        # assign anchor for label
        label = assign_anchor(feat_shape, label['gt_boxes'], data['im_info'], self.cfg,
                              self.feat_stride, self.anchor_scales,
                              self.anchor_ratios, self.allowed_border,
                              self.normalize_target, self.bbox_mean, self.bbox_std)
        return {'data': data, 'label': label}

    def parfetch_offline_memory_fromrec(self, iroidb):
        # get testing data for multigpu
        data, label = get_rpn_triple_batch_fromrec_offline(iroidb, self.cfg, self.video_index_dict, self.rec)
        data_shape = {k: v.shape for k, v in data.items()}
        del data_shape['im_info']
        _, feat_shape, _ = self.feat_sym.infer_shape(**data_shape)
        feat_shape = [int(i) for i in feat_shape[0]]

        # add gt_boxes to data for e2e
        data['gt_boxes'] = label['gt_boxes'][np.newaxis, :, :]

        # assign anchor for label
        label = assign_anchor(feat_shape, label['gt_boxes'], data['im_info'], self.cfg,
                              self.feat_stride, self.anchor_scales,
                              self.anchor_ratios, self.allowed_border,
                              self.normalize_target, self.bbox_mean, self.bbox_std)
        return {'data': data, 'label': label}
