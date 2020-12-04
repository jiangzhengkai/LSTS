import numpy as np
import os
import cv2
import random
from PIL import Image
from bbox.bbox_transform import clip_boxes
import mxnet as mx

# TODO: This two functions should be merged with individual data loader
def get_image(roidb, config):
    """
    preprocess image and return processed roidb
    :param roidb: a list of roidb
    :return: list of img as in mxnet format
    roidb add new item['im_info']
    0 --- x (width, second dim of im)
    |
    y (height, first dim of im)
    """
    num_images = len(roidb)
    processed_ims = []
    processed_roidb = []
    for i in range(num_images):
        roi_rec = roidb[i]
        assert os.path.exists(roi_rec['image']), '%s does not exist'.format(roi_rec['image'])
        im = cv2.imread(roi_rec['image'], cv2.IMREAD_COLOR|cv2.IMREAD_IGNORE_ORIENTATION)
        if roidb[i]['flipped']:
            im = im[:, ::-1, :]
        new_rec = roi_rec.copy()
        scale_ind = random.randrange(len(config.SCALES))
        target_size = config.SCALES[scale_ind][0]
        max_size = config.SCALES[scale_ind][1]
        im, im_scale = resize(im, target_size, max_size, stride=config.network.IMAGE_STRIDE)
        im_tensor = transform(im, config.network.PIXEL_MEANS)
        processed_ims.append(im_tensor)
        im_info = [im_tensor.shape[2], im_tensor.shape[3], im_scale]
        new_rec['boxes'] = clip_boxes(np.round(roi_rec['boxes'].copy() * im_scale), im_info[:2])
        new_rec['im_info'] = im_info
        processed_roidb.append(new_rec)
    return processed_ims, processed_roidb


def get_image_fromrec(roidb, config, video_index_dict, rec, is_train=False):
    """
    preprocess image and return processed roidb
    :param roidb: a list of roidb
    :return: list of img as in mxnet format
    roidb add new item['im_info']
    0 --- x (width, second dim of im)
    |
    y (height, first dim of im)
    """
    num_images = len(roidb)
    processed_ims = []
    processed_roidb = []
    for i in range(num_images):
        roi_rec = roidb[i]
        if roi_rec.has_key('pattern'):
            video_start_index = video_index_dict[os.path.dirname(roi_rec['image'].split('VID/')[-1])]
            cur_frame_id = int(os.path.splitext(roi_rec['image'].split('/')[-1])[0])
            im_index = video_start_index + cur_frame_id
            cur_s = rec.read_idx(im_index)
            _, im_cur_str = mx.recordio.unpack(cur_s)
            im = mx.image.imdecode(im_cur_str).asnumpy()
            im = im[:,:,::-1]
        elif not roi_rec.has_key('pattern') and is_train:
            im_index = video_index_dict[ roi_rec['image'].split('DET/')[-1][:-5] ]
            cur_s = rec.read_idx(im_index)
            _, im_cur_str = mx.recordio.unpack(cur_s)
            im = mx.image.imdecode(im_cur_str).asnumpy()
            im = im[:,:,::-1]                      #transform RGB format into BGR for resize input!!!!
        else:
            # /data/ILSVRC2015/Data/VID/val/ILSVRC2015_val_00000000/000000.JPEG
            video_start_index = video_index_dict[os.path.dirname(roi_rec['image'].split('VID/')[-1])]
            cur_frame_id = int(os.path.splitext(roi_rec['image'].split('/')[-1])[0])
            im_index = video_start_index + cur_frame_id
            cur_s = rec.read_idx(im_index)
            _, im_cur_str = mx.recordio.unpack(cur_s)
            im = mx.image.imdecode(im_cur_str).asnumpy()
            im = im[:,:,::-1]

        if roidb[i]['flipped']:
            im = im[:, ::-1, :]
        new_rec = roi_rec.copy()
        scale_ind = random.randrange(len(config.SCALES))
        target_size = config.SCALES[scale_ind][0]
        max_size = config.SCALES[scale_ind][1]
        im, im_scale = resize(im, target_size, max_size, stride=config.network.IMAGE_STRIDE)
        im_tensor = transform(im, config.network.PIXEL_MEANS)
        processed_ims.append(im_tensor)
        im_info = [im_tensor.shape[2], im_tensor.shape[3], im_scale]
        new_rec['boxes'] = clip_boxes(np.round(roi_rec['boxes'].copy() * im_scale), im_info[:2])
        new_rec['im_info'] = im_info
        processed_roidb.append(new_rec)
    return processed_ims, processed_roidb

def get_image_online_memory_fromrec(roidb, config, video_index_dict, rec):
    """
    preprocess image and return processed roidb
    :param roidb: a list of roidb
    :return: list of img as in mxnet format
    roidb add new item['im_info']
    0 --- x (width, second dim of im)
    |
    y (height, first dim of im)
    """
    num_images = len(roidb)
    processed_ims = []
    processed_roidb = []
    for i in range(num_images):
        roi_rec = roidb[i]
        if roi_rec.has_key('pattern'):
            video_start_index = video_index_dict[roi_rec['pattern'].split('VID/')[-1][:-10]]
            im_index = video_start_index + roi_rec['frame_id']
            cur_s = rec.read_idx(im_index)
            _, im_cur_str = mx.recordio.unpack(cur_s)

            im = mx.image.imdecode(im_cur_str).asnumpy()
            im = im[:, :, ::-1]  # transform RGB format into BGR for resize input!!!!
        else:
            im_index = video_index_dict[roi_rec['image'].split('DET/')[-1][:-5]]
            cur_s = rec.read_idx(im_index)
            _, im_cur_str = mx.recordio.unpack(cur_s)

            im = mx.image.imdecode(im_cur_str).asnumpy()
            im = im[:, :, ::-1]
        if roidb[i]['flipped']:
            im = im[:, ::-1, :]
        new_rec = roi_rec.copy()
        scale_ind = random.randrange(len(config.SCALES))
        target_size = config.SCALES[scale_ind][0]
        max_size = config.SCALES[scale_ind][1]
        im, im_scale = resize(im, target_size, max_size, stride=config.network.IMAGE_STRIDE)
        im_tensor = transform(im, config.network.PIXEL_MEANS)
        processed_ims.append(im_tensor)
        im_info = [im_tensor.shape[2], im_tensor.shape[3], im_scale]
        new_rec['boxes'] = clip_boxes(np.round(roi_rec['boxes'].copy() * im_scale), im_info[:2])
        new_rec['im_info'] = im_info
        processed_roidb.append(new_rec)
    return processed_ims, processed_roidb


def get_image_memory_withkey(roidb, config, video_index_dict, rec):
    """
    preprocess image and return processed roidb
    :param roidb: a list of roidb
    :return: list of img as in mxnet format
    roidb add new item['im_info']
    0 --- x (width, second dim of im)
    |
    y (height, first dim of im)
    """
    num_images = len(roidb)
    processed_ims_cur = []
    processed_ims_newkey = []
    processed_roidb = []
    for i in range(num_images):
        roi_rec = roidb[i]
        if not roi_rec['from_rec']:
            assert os.path.exists(roi_rec['data_cur']), '%s does not exist'.format(roi_rec['data_cur'])
            assert os.path.exists(roi_rec['data_newkey']), '%s does not exist'.format(roi_rec['data_newkey'])
            im_cur = cv2.imread(roi_rec['data_cur'], cv2.IMREAD_COLOR|cv2.IMREAD_IGNORE_ORIENTATION)
            im_newkey = cv2.imread(roi_rec['data_newkey'], cv2.IMREAD_COLOR|cv2.IMREAD_IGNORE_ORIENTATION)
        else:
            video_start_index = video_index_dict[roi_rec['pattern'].split('VID/')[-1][:-10] ]
            cur_index = video_start_index + roi_rec['frame_id']
            newkey_index = video_start_index + roi_rec['newkey_id']
            cur_s = rec.read_idx(cur_index)
            _, cur_im_str = mx.recordio.unpack(cur_s)
            im_cur = mx.image.imdecode(cur_im_str).asnumpy()
            im_cur = im_cur[:,:,::-1]
            newkey_s = rec.read_idx(newkey_index)
            _, newkey_im_str = mx.recordio.unpack(newkey_s)
            im_newkey = mx.image.imdecode(newkey_im_str).asnumpy()
            im_newkey = im_newkey[:,:,::-1]

        if roidb[i]['flipped']:
            im_cur = im_cur[:, ::-1, :]
            im_newkey = im_newkey[:, ::-1, :]
        new_rec = roi_rec.copy()
        scale_ind = random.randrange(len(config.SCALES))
        target_size = config.SCALES[scale_ind][0]
        max_size = config.SCALES[scale_ind][1]
        im_cur, im_scale = resize(im_cur, target_size, max_size, stride=config.network.IMAGE_STRIDE)
        im_tensor_cur = transform(im_cur, config.network.PIXEL_MEANS)
        im_newkey, im_scale = resize(im_newkey, target_size, max_size, stride=config.network.IMAGE_STRIDE)
        im_tensor_newkey = transform(im_newkey, config.network.PIXEL_MEANS)
        processed_ims_cur.append(im_tensor_cur)
        processed_ims_newkey.append(im_tensor_newkey)
        im_info = [im_tensor_cur.shape[2], im_tensor_cur.shape[3], im_scale]
        new_rec['boxes'] = clip_boxes(np.round(roi_rec['boxes'].copy() * im_scale), im_info[:2])
        new_rec['im_info'] = im_info
        processed_roidb.append(new_rec)
    return processed_ims_cur, processed_ims_newkey, processed_roidb


def get_image_memory_nokey(roidb, config, video_index_dict, rec):
    """
    preprocess image and return processed roidb
    :param roidb: a list of roidb
    :return: list of img as in mxnet format
    roidb add new item['im_info']
    0 --- x (width, second dim of im)
    |
    y (height, first dim of im)
    """
    num_images = len(roidb)
    processed_ims_cur = []
    processed_roidb = []
    for i in range(num_images):
        roi_rec = roidb[i]
        if not roi_rec['from_rec']:
            assert os.path.exists(roi_rec['data_cur']), '%s does not exist'.format(roi_rec['data_cur'])
            assert os.path.exists(roi_rec['data_newkey']), '%s does not exist'.format(roi_rec['data_newkey'])
            im_cur = cv2.imread(roi_rec['data_cur'], cv2.IMREAD_COLOR|cv2.IMREAD_IGNORE_ORIENTATION)
        else:

            video_start_index = video_index_dict[roi_rec['pattern'].split('VID/')[-1][:-10] ]
            cur_index = video_start_index + roi_rec['frame_id']

            cur_s = rec.read_idx(cur_index)
            _, cur_im_str = mx.recordio.unpack(cur_s)
            im_cur = mx.image.imdecode(cur_im_str).asnumpy()
            im_cur = im_cur[:,:,::-1]

        if roidb[i]['flipped']:
            im_cur = im_cur[:, ::-1, :]
        new_rec = roi_rec.copy()
        scale_ind = random.randrange(len(config.SCALES))
        target_size = config.SCALES[scale_ind][0]
        max_size = config.SCALES[scale_ind][1]
        im_cur, im_scale = resize(im_cur, target_size, max_size, stride=config.network.IMAGE_STRIDE)
        im_tensor_cur = transform(im_cur, config.network.PIXEL_MEANS)
        processed_ims_cur.append(im_tensor_cur)
        im_info = [im_tensor_cur.shape[2], im_tensor_cur.shape[3], im_scale]
        new_rec['boxes'] = clip_boxes(np.round(roi_rec['boxes'].copy() * im_scale), im_info[:2])
        new_rec['im_info'] = im_info
        processed_roidb.append(new_rec)
    return processed_ims_cur,  processed_roidb


def get_pair_image(roidb, config):
    """
    preprocess image and return processed roidb
    :param roidb: a list of roidb
    :return: list of img as in mxnet format
    roidb add new item['im_info']
    0 --- x (width, second dim of im)
    |
    y (height, first dim of im)
    """
    num_images = len(roidb)
    processed_ims = []
    processed_ref_ims = []
    processed_eq_flags = []
    processed_roidb = []
    for i in range(num_images):
        roi_rec = roidb[i]

        eq_flag = 0 # 0 for unequal, 1 for equal
        assert os.path.exists(roi_rec['image']), '%s does not exist'.format(roi_rec['image'])
        im = cv2.imread(roi_rec['image'], cv2.IMREAD_COLOR|cv2.IMREAD_IGNORE_ORIENTATION)

        if roi_rec.has_key('pattern'):
            ref_id = min(max(roi_rec['frame_seg_id'] + np.random.randint(config.TRAIN.MIN_OFFSET, config.TRAIN.MAX_OFFSET+1), 0),roi_rec['frame_seg_len']-1)
            ref_image = roi_rec['pattern'] % ref_id
            assert os.path.exists(ref_image), '%s does not exist'.format(ref_image)
            ref_im = cv2.imread(ref_image, cv2.IMREAD_COLOR|cv2.IMREAD_IGNORE_ORIENTATION)
            if ref_id == roi_rec['frame_seg_id']:
                eq_flag = 1
        else:
            ref_im = im.copy()
            eq_flag = 1

        if roidb[i]['flipped']:
            im = im[:, ::-1, :]
            ref_im = ref_im[:, ::-1, :]

        new_rec = roi_rec.copy()
        scale_ind = random.randrange(len(config.SCALES))
        target_size = config.SCALES[scale_ind][0]
        max_size = config.SCALES[scale_ind][1]

        im, im_scale = resize(im, target_size, max_size, stride=config.network.IMAGE_STRIDE)
        ref_im, im_scale = resize(ref_im, target_size, max_size, stride=config.network.IMAGE_STRIDE)
        im_tensor = transform(im, config.network.PIXEL_MEANS)
        ref_im_tensor = transform(ref_im, config.network.PIXEL_MEANS)
        processed_ims.append(im_tensor)
        processed_ref_ims.append(ref_im_tensor)
        processed_eq_flags.append(eq_flag)
        im_info = [im_tensor.shape[2], im_tensor.shape[3], im_scale]
        new_rec['boxes'] = roi_rec['boxes'].copy() * im_scale
        new_rec['im_info'] = im_info
        processed_roidb.append(new_rec)
    return processed_ims, processed_ref_ims, processed_eq_flags, processed_roidb

def get_pair_image_fromrec(roidb, config, video_index_dict, rec):
    """
    preprocess image and return processed roidb
    :param roidb: a list of roidb
    :return: list of img as in mxnet format
    roidb add new item['im_info']
    0 --- x (width, second dim of im)
    |
    y (height, first dim of im)
    """
    num_images = len(roidb)
    processed_ims = []
    processed_ref_ims = []
    processed_eq_flags = []
    processed_roidb = []
    for i in range(num_images):
        roi_rec = roidb[i]
        if roi_rec.has_key('pattern'):
            video_start_index = video_index_dict[roi_rec['pattern'].split('VID/')[-1][:-10]]
            im_index = video_start_index + int(roi_rec['frame_seg_id'])
            ref_index = min(max(im_index + np.random.randint(config.TRAIN.MIN_OFFSET, config.TRAIN.MAX_OFFSET+1), video_start_index),video_start_index + roi_rec['frame_seg_len'])

            im_s = rec.read_idx(im_index)
            _, im_str = mx.recordio.unpack(im_s)
            im = mx.image.imdecode(im_str).asnumpy()
            im = im[:,:,::-1]

            ref_s = rec.read_idx(ref_index)
            _, ref_str = mx.recordio.unpack(ref_s)
            ref_im = mx.image.imdecode(ref_str).asnumpy()
            ref_im = ref_im[:,:,::-1]
            eq_flag = 0
        else:
            im_index = video_index_dict[ roi_rec['image'].split('DET/')[-1][:-5] ]
            cur_s = rec.read_idx(im_index)
            _, im_str = mx.recordio.unpack(cur_s)
            im = mx.image.imdecode(im_str).asnumpy()
            im = im[:,:,::-1]
            ref_im = im.copy()
            eq_flag = 1

        if roidb[i]['flipped']:
            im = im[:, ::-1, :]
            ref_im = ref_im[:, ::-1, :]

        new_rec = roi_rec.copy()
        scale_ind = random.randrange(len(config.SCALES))
        target_size = config.SCALES[scale_ind][0]
        max_size = config.SCALES[scale_ind][1]

        im, im_scale = resize(im, target_size, max_size, stride=config.network.IMAGE_STRIDE)
        ref_im, im_scale = resize(ref_im, target_size, max_size, stride=config.network.IMAGE_STRIDE)
        im_tensor = transform(im, config.network.PIXEL_MEANS)
        ref_im_tensor = transform(ref_im, config.network.PIXEL_MEANS)
        processed_ims.append(im_tensor)
        processed_ref_ims.append(ref_im_tensor)
        processed_eq_flags.append(eq_flag)
        im_info = [im_tensor.shape[2], im_tensor.shape[3], im_scale]
        new_rec['boxes'] = roi_rec['boxes'].copy() * im_scale
        new_rec['im_info'] = im_info
        processed_roidb.append(new_rec)
    return processed_ims, processed_ref_ims, processed_eq_flags, processed_roidb


def get_triple_image_offline(roidbs, cfg):
    assert len(roidbs)==1, 'Single batch only'
    num_samples = len(roidbs)
    processed_oldkey_images = []
    processed_newkey_images = []
    processed_cur_images = []
    #processed_im_info = []
    processed_eq_flag_key2key = []
    processed_eq_flag_key2cur = []
    processed_roidbs = []

    for i in range(num_samples):
        roidb = roidbs[0]
        eq_flag_key2cur = 0
        eq_flag_key2key = 0
        assert os.path.exists(roidb['image']), '%s does not exist'.format(roidb['image'])
        im_cur = cv2.imread(roidb['image'], cv2.IMREAD_COLOR|cv2.IMREAD_IGNORE_ORIENTATION)

        if roidb.has_key('pattern'):
            oldkey_id = max(roidb['frame_seg_id'] + np.random.randint(int(-1*cfg.TRAIN.segment_length), int(-0.5*cfg.TRAIN.segment_length)), 0)
            newkey_id = min(max(roidb['frame_seg_id']+np.random.randint(int(-0.5*cfg.TRAIN.segment_length), int(0.5*cfg.TRAIN.segment_length)), 0), roidb['frame_seg_len']-1)
            oldkey_imname = roidb['pattern'] % oldkey_id
            newkey_imname = roidb['pattern'] % newkey_id
            assert os.path.exists(roidb['pattern'] % oldkey_id), '%s does not exist'.format(oldkey_imname)
            assert os.path.exists(roidb['pattern'] % newkey_id), '%s does not exist'.format(
                newkey_imname)
            oldkey_im = cv2.imread(roidb['pattern'] % oldkey_id, cv2.IMREAD_COLOR|cv2.IMREAD_IGNORE_ORIENTATION)
            newkey_im = cv2.imread(roidb['pattern'] % newkey_id, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
            if oldkey_id == newkey_id:
                eq_flag_key2key = 1
            if newkey_id == roidb['frame_seg_id']:
                eq_flag_key2cur = 1
        else:
            oldkey_im = im_cur.copy()
            newkey_im = im_cur.copy()
            eq_flag_key2cur = 1
            eq_flag_key2key = 1

        if roidb['flipped']:
            oldkey_im = oldkey_im[:, ::-1, :]
            newkey_im = newkey_im[:, ::-1, :]
            im_cur = im_cur[:, ::-1, :]

        new_rec = roidb.copy()
        scale_ind = random.randrange(len(cfg.SCALES))
        target_size = cfg.SCALES[scale_ind][0]
        max_size = cfg.SCALES[scale_ind][1]
        oldkey_im, im_scale = resize(oldkey_im, target_size, max_size, stride=cfg.network.IMAGE_STRIDE)
        newkey_im, im_scale = resize(newkey_im, target_size, max_size, stride=cfg.network.IMAGE_STRIDE)
        im_cur, im_scale = resize(im_cur, target_size, max_size, stride=cfg.network.IMAGE_STRIDE)
        oldkey_im_tensor = transform(oldkey_im, cfg.network.PIXEL_MEANS)
        newkey_im_tensor = transform(newkey_im, cfg.network.PIXEL_MEANS)
        im_cur_tensor = transform(im_cur, cfg.network.PIXEL_MEANS)
        im_info = [im_cur_tensor.shape[2], im_cur_tensor.shape[3], im_scale]
        processed_cur_images.append(im_cur_tensor)
        processed_newkey_images.append(newkey_im_tensor)
        processed_oldkey_images.append(oldkey_im_tensor)
        processed_eq_flag_key2cur.append(eq_flag_key2cur)
        processed_eq_flag_key2key.append(eq_flag_key2key)
        new_rec['boxes'] = roidb['boxes'].copy() * im_scale
        new_rec['im_info'] = im_info
        processed_roidbs.append(new_rec)
    return processed_oldkey_images, processed_cur_images, processed_newkey_images, processed_eq_flag_key2key, processed_eq_flag_key2cur, processed_roidbs

def get_triple_image_fromrec_offline(roidbs, cfg, video_index_dict, rec):
    assert len(roidbs)==1, 'Single batch only'
    num_samples = len(roidbs)
    processed_oldkey_images = []
    processed_newkey_images = []
    processed_cur_images = []
    #processed_im_info = []
    processed_eq_flag_key2key = []
    processed_eq_flag_key2cur = []
    processed_roidbs = []
    for i in range(num_samples):
        roidb = roidbs[0]
        # 0 for unequal 1 for equal
        eq_flag_key2cur = 0
        eq_flag_key2key = 0
        #print 'path',roidb['image']
        #assert os.path.exists(roidb['image']), '%s does not exist'.format(roidb['image'])
        #im_cur = cv2.imread(roidb['image'], cv2.IMREAD_COLOR|cv2.IMREAD_IGNORE_ORIENTATION)
        if roidb.has_key('pattern'):
            video_start_index = video_index_dict[roidb['pattern'].split('VID/')[-1][:-10]]
            im_index = video_start_index + int(roidb['frame_seg_id'])      #roidb['pattern'] look likes './data/ILSVRC2015/Data/VID/train/ILSVRC2015_VID_train_0002/ILSVRC2015_train_00533002/%06d.JPEG'
            oldkey_index = max(im_index + np.random.randint(int(-1*cfg.TRAIN.segment_length), int(-0.5*cfg.TRAIN.segment_length)), video_start_index)
            newkey_index = min(max(im_index + np.random.randint(int(-0.5*cfg.TRAIN.segment_length), int(0.5*cfg.TRAIN.segment_length)), video_start_index), video_start_index + roidb['frame_seg_len']-1)
            #print video_start_index, im_index, oldkey_index, newkey_index
            im_cur_name = roidb['pattern'] % (im_index-video_start_index)
            oldkey_imname = roidb['pattern'] % (im_index-video_start_index)
            newkey_imname = roidb['pattern'] % (im_index-video_start_index)

            oldkey_s = rec.read_idx(oldkey_index)
            _, oldkey_im_str = mx.recordio.unpack(oldkey_s)
            oldkey_im = mx.image.imdecode(oldkey_im_str).asnumpy()
            oldkey_im = oldkey_im[:,:,::-1]

            newkey_s = rec.read_idx(newkey_index)
            _, newkey_im_str = mx.recordio.unpack(newkey_s)
            newkey_im = mx.image.imdecode(newkey_im_str).asnumpy()
            newkey_im = newkey_im[:,:,::-1]


            cur_s = rec.read_idx(im_index)
            _, im_cur_str = mx.recordio.unpack(cur_s)
            im_cur = mx.image.imdecode(im_cur_str).asnumpy()
            im_cur = im_cur[:,:,::-1]

            if oldkey_index == newkey_index:
                eq_flag_key2key = 1
            if newkey_index == im_index:
                eq_flag_key2cur = 1
        else:
            im_index = video_index_dict[roidb['image'].split('DET/')[-1][:-5]]
            cur_s = rec.read_idx(im_index)
            _, im_cur_str = mx.recordio.unpack(cur_s)
            im_cur = mx.image.imdecode(im_cur_str).asnumpy()
            im_cur = im_cur[:,:,::-1]
            oldkey_im = im_cur.copy()
            newkey_im = im_cur.copy()
            eq_flag_key2cur = 1
            eq_flag_key2key = 1

        if roidb['flipped']:
            oldkey_im = oldkey_im[:, ::-1, :]
            newkey_im = newkey_im[:, ::-1, :]
            im_cur = im_cur[:, ::-1, :]

        new_rec = roidb.copy()
        scale_ind = random.randrange(len(cfg.SCALES))
        target_size = cfg.SCALES[scale_ind][0]
        max_size = cfg.SCALES[scale_ind][1]
        oldkey_im, im_scale = resize(oldkey_im, target_size, max_size, stride=cfg.network.IMAGE_STRIDE)
        newkey_im, im_scale = resize(newkey_im, target_size, max_size, stride=cfg.network.IMAGE_STRIDE)
        im_cur, im_scale = resize(im_cur, target_size, max_size, stride=cfg.network.IMAGE_STRIDE)
        oldkey_im_tensor = transform(oldkey_im, cfg.network.PIXEL_MEANS)
        newkey_im_tensor = transform(newkey_im, cfg.network.PIXEL_MEANS)
        im_cur_tensor = transform(im_cur, cfg.network.PIXEL_MEANS)
        im_info = [im_cur_tensor.shape[2], im_cur_tensor.shape[3], im_scale]
        processed_cur_images.append(im_cur_tensor)
        processed_newkey_images.append(newkey_im_tensor)
        processed_oldkey_images.append(oldkey_im_tensor)
        processed_eq_flag_key2cur.append(eq_flag_key2cur)
        processed_eq_flag_key2key.append(eq_flag_key2key)
        new_rec['boxes'] = roidb['boxes'].copy() * im_scale
        new_rec['im_info'] = im_info

        processed_roidbs.append(new_rec)
    return processed_oldkey_images, processed_cur_images, processed_newkey_images, processed_eq_flag_key2key, processed_eq_flag_key2cur, processed_roidbs

def get_triple_image_online(roidbs, cfg):
    assert len(roidbs)==1, 'Single batch only'
    num_samples = len(roidbs)
    processed_oldkey_images = []
    processed_newkey_images = []
    processed_cur_images = []
    #processed_im_info = []
    processed_eq_flag_key2key = []
    processed_eq_flag_key2cur = []
    processed_roidbs = []

    for i in range(num_samples):
        roidb = roidbs[0]
        eq_flag_key2cur = 0
        eq_flag_key2key = 0
        assert os.path.exists(roidb['image']), '%s does not exist'.format(roidb['image'])
        im_cur = cv2.imread(roidb['image'], cv2.IMREAD_COLOR|cv2.IMREAD_IGNORE_ORIENTATION)

        if roidb.has_key('pattern'):
            newkey_id = min(max(roidb['frame_seg_id']+np.random.randint(int(-1.0*cfg.TRAIN.segment_length), int(0.0*cfg.TRAIN.segment_length)), 0), roidb['frame_seg_len']-1)
            oldkey_id = max(roidb['frame_seg_id'] + np.random.randint(int(-1.5 * cfg.TRAIN.segment_length), int(-1.0 * cfg.TRAIN.segment_length)), 0)
            oldkey_imname = roidb['pattern'] % oldkey_id
            newkey_imname = roidb['pattern'] % newkey_id
            assert os.path.exists(roidb['pattern'] % oldkey_id), '%s does not exist'.format(oldkey_imname)
            assert os.path.exists(roidb['pattern'] % newkey_id), '%s does not exist'.format(
                newkey_imname)
            oldkey_im = cv2.imread(roidb['pattern'] % oldkey_id, cv2.IMREAD_COLOR|cv2.IMREAD_IGNORE_ORIENTATION)
            newkey_im = cv2.imread(roidb['pattern'] % newkey_id, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
            if oldkey_id == newkey_id:
                eq_flag_key2key = 1
            if newkey_id == roidb['frame_seg_id']:
                eq_flag_key2cur = 1
        else:
            oldkey_im = im_cur.copy()
            newkey_im = im_cur.copy()
            eq_flag_key2cur = 1
            eq_flag_key2key = 1

        if roidb['flipped']:
            oldkey_im = oldkey_im[:, ::-1, :]
            newkey_im = newkey_im[:, ::-1, :]
            im_cur = im_cur[:, ::-1, :]

        new_rec = roidb.copy()
        scale_ind = random.randrange(len(cfg.SCALES))
        target_size = cfg.SCALES[scale_ind][0]
        max_size = cfg.SCALES[scale_ind][1]
        oldkey_im, im_scale = resize(oldkey_im, target_size, max_size, stride=cfg.network.IMAGE_STRIDE)
        newkey_im, im_scale = resize(newkey_im, target_size, max_size, stride=cfg.network.IMAGE_STRIDE)
        im_cur, im_scale = resize(im_cur, target_size, max_size, stride=cfg.network.IMAGE_STRIDE)
        oldkey_im_tensor = transform(oldkey_im, cfg.network.PIXEL_MEANS)
        newkey_im_tensor = transform(newkey_im, cfg.network.PIXEL_MEANS)
        im_cur_tensor = transform(im_cur, cfg.network.PIXEL_MEANS)
        im_info = [im_cur_tensor.shape[2], im_cur_tensor.shape[3], im_scale]
        processed_cur_images.append(im_cur_tensor)
        processed_newkey_images.append(newkey_im_tensor)
        processed_oldkey_images.append(oldkey_im_tensor)
        processed_eq_flag_key2cur.append(eq_flag_key2cur)
        processed_eq_flag_key2key.append(eq_flag_key2key)
        new_rec['boxes'] = roidb['boxes'].copy() * im_scale
        new_rec['im_info'] = im_info
        processed_roidbs.append(new_rec)
    return processed_oldkey_images, processed_cur_images, processed_newkey_images, processed_eq_flag_key2key, processed_eq_flag_key2cur, processed_roidbs

def get_triple_image_fromrec_online(roidbs, cfg, video_index_dict, rec):
    assert len(roidbs)==1, 'Single batch only'
    num_samples = len(roidbs)
    processed_oldkey_images = []
    processed_newkey_images = []
    processed_cur_images = []
    #processed_im_info = []
    processed_eq_flag_key2key = []
    processed_eq_flag_key2cur = []
    processed_roidbs = []
    for i in range(num_samples):
        roidb = roidbs[0]
        # 0 for unequal 1 for equal
        eq_flag_key2cur = 0
        eq_flag_key2key = 0
        if roidb.has_key('pattern'):
            video_start_index = video_index_dict[roidb['pattern'].split('VID/')[-1][:-10]]
            im_index = video_start_index + int(roidb['frame_seg_id'])      #roidb['pattern'] look likes './data/ILSVRC2015/Data/VID/train/ILSVRC2015_VID_train_0002/ILSVRC2015_train_00533002/%06d.JPEG'
            newkey_index = min(max(im_index + np.random.randint(int(-1.0*cfg.TRAIN.segment_length), int(0.0*cfg.TRAIN.segment_length)), video_start_index), video_start_index + roidb['frame_seg_len']-1)
            oldkey_index = max(im_index + np.random.randint(int(-1.5 * cfg.TRAIN.segment_length), int(-1.0 * cfg.TRAIN.segment_length)), video_start_index)
            #print video_start_index, im_index, oldkey_index, newkey_index
            im_cur_name = roidb['pattern'] % (im_index-video_start_index)
            oldkey_imname = roidb['pattern'] % (im_index-video_start_index)
            newkey_imname = roidb['pattern'] % (im_index-video_start_index)
         
            oldkey_s = rec.read_idx(oldkey_index)
            _, oldkey_im_str = mx.recordio.unpack(oldkey_s)
            oldkey_im = mx.image.imdecode(oldkey_im_str).asnumpy()
            oldkey_im = oldkey_im[:,:,::-1]

            newkey_s = rec.read_idx(newkey_index)
            _, newkey_im_str = mx.recordio.unpack(newkey_s)
            newkey_im = mx.image.imdecode(newkey_im_str).asnumpy()
            newkey_im = newkey_im[:,:,::-1]


            cur_s = rec.read_idx(im_index)
            _, im_cur_str = mx.recordio.unpack(cur_s)
            im_cur = mx.image.imdecode(im_cur_str).asnumpy()
            im_cur = im_cur[:,:,::-1]

            if oldkey_index == newkey_index:
                eq_flag_key2key = 1
            if newkey_index == im_index:
                eq_flag_key2cur = 1
        else:
            im_index = video_index_dict[roidb['image'].split('DET/')[-1][:-5]]
            cur_s = rec.read_idx(im_index)
            _, im_cur_str = mx.recordio.unpack(cur_s)
            im_cur = mx.image.imdecode(im_cur_str).asnumpy()
            im_cur = im_cur[:,:,::-1]
            #print 'p41'
            oldkey_im = im_cur.copy()
            newkey_im = im_cur.copy()
            eq_flag_key2cur = 1
            eq_flag_key2key = 1

        if roidb['flipped']:
            oldkey_im = oldkey_im[:, ::-1, :]
            newkey_im = newkey_im[:, ::-1, :]
            im_cur = im_cur[:, ::-1, :]

        new_rec = roidb.copy()
        scale_ind = random.randrange(len(cfg.SCALES))
        target_size = cfg.SCALES[scale_ind][0]
        max_size = cfg.SCALES[scale_ind][1]
        oldkey_im, im_scale = resize(oldkey_im, target_size, max_size, stride=cfg.network.IMAGE_STRIDE)
        newkey_im, im_scale = resize(newkey_im, target_size, max_size, stride=cfg.network.IMAGE_STRIDE)
        im_cur, im_scale = resize(im_cur, target_size, max_size, stride=cfg.network.IMAGE_STRIDE)
        oldkey_im_tensor = transform(oldkey_im, cfg.network.PIXEL_MEANS)
        newkey_im_tensor = transform(newkey_im, cfg.network.PIXEL_MEANS)
        im_cur_tensor = transform(im_cur, cfg.network.PIXEL_MEANS)
        im_info = [im_cur_tensor.shape[2], im_cur_tensor.shape[3], im_scale]
        processed_cur_images.append(im_cur_tensor)
        processed_newkey_images.append(newkey_im_tensor)
        processed_oldkey_images.append(oldkey_im_tensor)
        processed_eq_flag_key2cur.append(eq_flag_key2cur)
        processed_eq_flag_key2key.append(eq_flag_key2key)
        new_rec['boxes'] = roidb['boxes'].copy() * im_scale
        new_rec['im_info'] = im_info

        processed_roidbs.append(new_rec)
    return processed_oldkey_images, processed_cur_images, processed_newkey_images, processed_eq_flag_key2key, processed_eq_flag_key2cur, processed_roidbs


def get_segmentation_image_fromrec(segdb, config, img_rec, label_rec, video_index_dict, label_index_dict, is_train=True):
    """
    propocess image and return segdb
    :param segdb: a list of segdb
    :return: list of img as mxnet format
    """
    num_images = len(segdb)
    assert num_images > 0, 'No images'
    processed_cur_images = []
    processed_segdb = []
    processed_seg_cls_gt = []
    for i in range(num_images):
        seg_rec = segdb[i]

        image_name = seg_rec['image_name']
        img_cur_id = video_index_dict[image_name]
        cur_s = img_rec.read_idx(img_cur_id)

        im_cur = mx.image.imdecode(cur_s).asnumpy()
        im_cur = im_cur[:,:,::-1]

        if seg_rec['flipped']  and is_train:
            im_cur = im_cur[:, ::-1, :]

        new_rec = seg_rec.copy()

        scale_ind = random.randrange(len(config.SCALES))
        target_size = config.SCALES[scale_ind][0]
        max_size = config.SCALES[scale_ind][1]

        im_cur, im_scale = resize(im_cur, target_size, max_size, stride=config.network.IMAGE_STRIDE)
        im_cur_tensor = transform(im_cur, config.network.PIXEL_MEANS)
        im_info = [im_cur_tensor.shape[2], im_cur_tensor.shape[3], im_scale]
        new_rec['im_info'] = im_info

        seg_cls_gt_rec_id = label_index_dict[seg_rec['image_name']]
        seg_cls_gt_s = label_rec.read_idx(seg_cls_gt_rec_id)

        seg_cls_gt = np.squeeze(mx.image.imdecode(seg_cls_gt_s, flag=0).asnumpy())

        row, col = np.where(seg_cls_gt==-1)
        seg_cls_gt[row, col] = 255
        if seg_rec['flipped']:
            seg_cls_gt = seg_cls_gt[:,::-1]
        assert seg_cls_gt.ndim == 2, 'seg gt dim %d' % seg_cls_gt.ndim

        seg_cls_gt, seg_cls_gt_scale = resize(
            seg_cls_gt, target_size, max_size, stride=config.network.IMAGE_STRIDE, interpolation=cv2.INTER_NEAREST)
        seg_cls_gt_tensor = transform_seg_gt(seg_cls_gt)

        processed_cur_images.append(im_cur_tensor)
        processed_segdb.append(new_rec)
        processed_seg_cls_gt.append(seg_cls_gt_tensor)

    return processed_cur_images, processed_seg_cls_gt, processed_segdb

def get_segmentation_image_triplet_fromrec(segdb, config, img_rec, label_rec, video_index_dict, label_index_dict, is_train=True):
    """
        propocess image and return segdb
        :param segdb: a list of segdb
        :return: list of img as mxnet format
        """
    num_images = len(segdb)
    assert num_images > 0, 'No images'
    processed_newkey_images = []
    processed_oldkey_images = []
    processed_cur_images = []
    processed_segdb = []
    processed_seg_cls_gt = []
    processed_eq_flags_key2key = []
    processed_eq_flags_key2cur = []

    for i in range(num_images):
        seg_rec = segdb[i]
        eq_flag_key2key = 0
        eq_flag_key2cur = 0
        frame_name = seg_rec['frame_name']
        video_20frame_index = video_index_dict[frame_name]
        im_cur_id = video_20frame_index
        cur_s = img_rec.read_idx(im_cur_id)
        im_cur = mx.image.imdecode(cur_s).asnumpy()
        im_cur = im_cur[:, :, ::-1]

        if is_train:
            im_oldkey_id = im_cur_id + np.random.randint(-1.*config.TRAIN.segment_length, 0.0*config.TRAIN.segment_length)
            im_newkey_id = im_cur_id + np.random.randint(-1.*config.TRAIN.segment_length, config.TRAIN.segment_length)

        else:
            im_oldkey_id = im_cur_id + np.random.randint(-1.*config.TRAIN.segment_length, -0.5*config.TRAIN.segment_length)
            im_newkey_id = im_cur_id + np.random.randint(-.5*config.TRAIN.segment_length, 0.5*config.TRAIN.segment_length)

        if im_oldkey_id == im_newkey_id:
            eq_flag_key2key = 1
        if im_newkey_id == im_cur_id:
            eq_flag_key2cur = 1

        oldkey_s = img_rec.read_idx(im_oldkey_id)
        im_oldkey = mx.image.imdecode(oldkey_s).asnumpy()
        im_oldkey = im_oldkey[:, :, ::-1]

        newkey_s = img_rec.read_idx(im_newkey_id)
        im_newkey = mx.image.imdecode(newkey_s).asnumpy()
        im_newkey = im_newkey[:, :, ::-1]

        if seg_rec['flipped'] and is_train:
            im_cur = im_cur[:, ::-1, :]
            im_oldkey = im_oldkey[:, ::-1, :]
            im_newkey = im_newkey[:, ::-1, :]

        new_rec = seg_rec.copy()

        scale_ind = random.randrange(len(config.SCALES))
        target_size = config.SCALES[scale_ind][0]
        max_size = config.SCALES[scale_ind][1]

        im_cur, im_scale = resize(im_cur, target_size, max_size, stride=config.network.IMAGE_STRIDE)
        im_oldkey, im_scale = resize(im_oldkey, target_size, max_size, stride=config.network.IMAGE_STRIDE)
        im_newkey, im_scale = resize(im_newkey, target_size, max_size, stride=config.network.IMAGE_STRIDE)

        im_cur_tensor = transform(im_cur, config.network.PIXEL_MEANS)
        im_oldkey_tensor = transform(im_oldkey, config.network.PIXEL_MEANS)
        im_newkey_tensor = transform(im_newkey, config.network.PIXEL_MEANS)

        im_info = [im_cur_tensor.shape[2], im_cur_tensor.shape[3], im_scale]
        new_rec['im_info'] = im_info

        # seg_cls_gt = np.array(Image.open(seg_rec['seg_cls_path']))
        seg_cls_gt_rec_id = label_index_dict[seg_rec['frame_name']]
        seg_cls_gt_s = label_rec.read_idx(seg_cls_gt_rec_id)

        seg_cls_gt = np.squeeze(mx.image.imdecode(seg_cls_gt_s, flag=0).asnumpy())

        row, col = np.where(seg_cls_gt == -1)
        seg_cls_gt[row, col] = 255
        if seg_rec['flipped']:
            seg_cls_gt = seg_cls_gt[:, ::-1]
        assert seg_cls_gt.ndim == 2, 'seg gt dim %d' % seg_cls_gt.ndim

        seg_cls_gt, seg_cls_gt_scale = resize(seg_cls_gt, target_size, max_size, stride=config.network.IMAGE_STRIDE, interpolation=cv2.INTER_NEAREST)
        seg_cls_gt_tensor = transform_seg_gt(seg_cls_gt)

        processed_newkey_images.append(im_newkey_tensor)
        processed_oldkey_images.append(im_oldkey_tensor)
        processed_cur_images.append(im_cur_tensor)
        processed_eq_flags_key2cur.append(eq_flag_key2cur)
        processed_eq_flags_key2key.append(eq_flag_key2key)

        processed_segdb.append(new_rec)
        processed_seg_cls_gt.append(seg_cls_gt_tensor)

    return processed_cur_images, processed_oldkey_images, processed_newkey_images, processed_eq_flags_key2cur, processed_eq_flags_key2key, processed_seg_cls_gt, processed_segdb


def get_segmentation_image_sequence_fromrec(segdb, config, img_rec, label_rec, video_index_dict, label_index_dict):
    """
        propocess image and return segdb
        :param segdb: a list of segdb
        :return: list of img as mxnet format
        """
    num_images = len(segdb)
    assert num_images > 0, 'No images'
    processed_newkey_images = []
    processed_oldkey_images = []
    processed_cur_images = []
    processed_segdb = []

    for i in range(num_images):
        seg_rec = segdb[i]
        eq_flag_key2key = 0
        eq_flag_key2cur = 0
        video_frame_20_index = video_index_dict[seg_rec['video_frame_20_name']]
        im_cur_id = seg_rec['rec_idx']
        min_idx = video_frame_20_index - 19
        max_idx = video_frame_20_index + 10

        cur_s = img_rec.read_idx(im_cur_id)
        im_cur = mx.image.imdecode(cur_s).asnumpy()
        im_cur = im_cur[:, :, ::-1]

        if seg_rec['frame_flag'] == 0:
            im_newkey_id = im_cur_id
            im_oldkey_id = im_cur_id
        elif seg_rec['frame_flag'] % config.TRAIN.segment_length == 0:
            im_oldkey_id = im_cur_id - config.TRAIN.segment_length
            im_newkey_id = im_cur_id
        else:
            im_oldkey_id = im_cur_id - seg_rec['frame_flag'] % config.TRAIN.segment_length
            im_newkey_id = im_oldkey_id + config.TRAIN.segment_length

        im_oldkey_id = max(im_oldkey_id, min_idx)
        im_newkey_id = min(im_newkey_id, max_idx)

        if im_oldkey_id == im_newkey_id:
            eq_flag_key2key = 1
        if im_newkey_id == im_cur_id:
            eq_flag_key2cur = 1

        oldkey_s = img_rec.read_idx(im_oldkey_id)
        im_oldkey = mx.image.imdecode(oldkey_s).asnumpy()
        im_oldkey = im_oldkey[:, :, ::-1]

        newkey_s = img_rec.read_idx(im_newkey_id)
        im_newkey = mx.image.imdecode(newkey_s).asnumpy()
        im_newkey = im_newkey[:, :, ::-1]

        if seg_rec['flipped'] and is_train:
            im_cur = im_cur[:, ::-1, :]
            im_oldkey = im_oldkey[:, ::-1, :]
            im_newkey = im_newkey[:, ::-1, :]

        new_rec = seg_rec.copy()

        scale_ind = random.randrange(len(config.SCALES))
        target_size = config.SCALES[scale_ind][0]
        max_size = config.SCALES[scale_ind][1]

        im_cur, im_scale = resize(im_cur, target_size, max_size, stride=config.network.IMAGE_STRIDE)
        im_oldkey, im_scale = resize(im_oldkey, target_size, max_size, stride=config.network.IMAGE_STRIDE)
        im_newkey, im_scale = resize(im_newkey, target_size, max_size, stride=config.network.IMAGE_STRIDE)

        im_cur_tensor = transform(im_cur, config.network.PIXEL_MEANS)
        im_oldkey_tensor = transform(im_oldkey, config.network.PIXEL_MEANS)
        im_newkey_tensor = transform(im_newkey, config.network.PIXEL_MEANS)

        im_info = [im_cur_tensor.shape[2], im_cur_tensor.shape[3], im_scale]
        new_rec['im_info'] = im_info

        processed_newkey_images.append(im_newkey_tensor)
        processed_oldkey_images.append(im_oldkey_tensor)
        processed_cur_images.append(im_cur_tensor)

        processed_segdb.append(new_rec)


    return processed_cur_images, processed_oldkey_images, processed_newkey_images, processed_segdb



def resize(im, target_size, max_size, stride=0, interpolation = cv2.INTER_LINEAR):
    """
    only resize input image to target size and return scale
    :param im: BGR image input by opencv
    :param target_size: one dimensional size (the short side)
    :param max_size: one dimensional max size (the long side)
    :param stride: if given, pad the image to designated stride
    :param interpolation: if given, using given interpolation method to resize image
    :return:
    """
    im_shape = im.shape
    im_size_min = np.min(im_shape[0:2])
    im_size_max = np.max(im_shape[0:2])
    im_scale = float(target_size) / float(im_size_min)
    # prevent bigger axis from being more than max_size:
    if np.round(im_scale * im_size_max) > max_size:
        im_scale = float(max_size) / float(im_size_max)
    im = cv2.resize(im, None, None, fx=im_scale, fy=im_scale, interpolation=interpolation)

    if stride == 0:
        return im, im_scale
    else:
        # pad to product of stride
        im_height = int(np.ceil(im.shape[0] / float(stride)) * stride)
        im_width = int(np.ceil(im.shape[1] / float(stride)) * stride)
        im_channel = im.shape[2]
        padded_im = np.zeros((im_height, im_width, im_channel))
        padded_im[:im.shape[0], :im.shape[1], :] = im
        return padded_im, im_scale

def transform(im, pixel_means):
    """
    transform into mxnet tensor
    substract pixel size and transform to correct format
    :param im: [height, width, channel] in BGR
    :param pixel_means: [B, G, R pixel means]
    :return: [batch, channel, height, width]
    """
    im_tensor = np.zeros((1, 3, im.shape[0], im.shape[1]))
    for i in range(3):
        im_tensor[0, i, :, :] = im[:, :, 2 - i] - pixel_means[2 - i]
    return im_tensor

def transform_seg_gt(gt):
    """
    transform segmentation gt image into mxnet tensor
    :param gt: [height, width, channel = 1]
    :return: [batch, channel = 1, height, width]
    """
    gt_tensor = np.zeros((1, 1, gt.shape[0], gt.shape[1]))
    gt_tensor[0, 0, :, :] = gt[:, :]

    return gt_tensor

def transform_inverse(im_tensor, pixel_means):
    """
    transform from mxnet im_tensor to ordinary RGB image
    im_tensor is limited to one image
    :param im_tensor: [batch, channel, height, width]
    :param pixel_means: [B, G, R pixel means]
    :return: im [height, width, channel(RGB)]
    """
    assert im_tensor.shape[0] == 1
    im_tensor = im_tensor.copy()
    # put channel back
    channel_swap = (0, 2, 3, 1)
    im_tensor = im_tensor.transpose(channel_swap)
    im = im_tensor[0]
    assert im.shape[2] == 3
    im += pixel_means[[2, 1, 0]]
    im = im.astype(np.uint8)
    return im

def tensor_vstack(tensor_list, pad=0):
    """
    vertically stack tensors
    :param tensor_list: list of tensor to be stacked vertically
    :param pad: label to pad with
    :return: tensor with max shape
    """
    ndim = len(tensor_list[0].shape)
    dtype = tensor_list[0].dtype
    islice = tensor_list[0].shape[0]
    dimensions = []
    first_dim = sum([tensor.shape[0] for tensor in tensor_list])
    dimensions.append(first_dim)
    for dim in range(1, ndim):
        dimensions.append(max([tensor.shape[dim] for tensor in tensor_list]))
    if pad == 0:
        all_tensor = np.zeros(tuple(dimensions), dtype=dtype)
    elif pad == 1:
        all_tensor = np.ones(tuple(dimensions), dtype=dtype)
    else:
        all_tensor = np.full(tuple(dimensions), pad, dtype=dtype)
    if ndim == 1:
        for ind, tensor in enumerate(tensor_list):
            all_tensor[ind*islice:(ind+1)*islice] = tensor
    elif ndim == 2:
        for ind, tensor in enumerate(tensor_list):
            all_tensor[ind*islice:(ind+1)*islice, :tensor.shape[1]] = tensor
    elif ndim == 3:
        for ind, tensor in enumerate(tensor_list):
            all_tensor[ind*islice:(ind+1)*islice, :tensor.shape[1], :tensor.shape[2]] = tensor
    elif ndim == 4:
        for ind, tensor in enumerate(tensor_list):
            all_tensor[ind*islice:(ind+1)*islice, :tensor.shape[1], :tensor.shape[2], :tensor.shape[3]] = tensor
    else:
        raise Exception('Sorry, unimplemented.')
    return all_tensor
