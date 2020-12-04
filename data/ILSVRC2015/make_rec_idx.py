import os
import mxnet as mx
import argparse
import xml.etree.ElementTree as ET
import numpy as np

def img2rec(det_imagenet_path, out_rec_idx_path, root_path, rec_idx_path, rec_data_path):
    """
    :param vid_imageset_path: list of vid image sets: [vid_train, vid_val]
    :param det_imageset_path: list of det image sets:
    :param out_rec_idx_list: a file record the start index of video in rec file
    :param rec_idx_path: rec index file path
    :param rec_data_path: rec data file path
    """
    vid_imageset_path = ['ImageSets/VID_train_15frames.txt', 'ImageSets/VID_val_videos.txt']
    video_index_dict = {}
    rec = mx.recordio.MXIndexedRecordIO(rec_idx_path, rec_data_path, 'w')
    rec_index_list = open(out_rec_idx_path, 'w')
    idx = 0
    for imageset_path in vid_imageset_path:
        print('imageset_path', imageset_path)
        for line in open(imageset_path, 'r').readlines():
            key = line.split(' ')[0]
            if not video_index_dict.has_key(key):
                video_index_dict[key] = idx
                print('recording %d video' % idx)
                images_path = sorted(os.listdir(os.path.join(root_path, 'Data/VID', key)))
                num_images = len(images_path)
                rec_index_list.writelines(key + ' ' + str(idx) + ' ' + str(num_images) + '\n')
                for image_path in images_path:
                    annotation_path = os.path.join(root_path, 'Annotations/VID', key, image_path.split('.')[0] + '.xml')
                    image_path = os.path.join(root_path, 'Data/VID', key, image_path)
                    image_data = open(image_path, 'rb').read()
                    gt_boxes = load_annotations(annotation_path, num_classes=31)
                    header = mx.recordio.IRHeader(flag=0, label=gt_boxes, id=idx, id2=0)
                    s = mx.recordio.pack(header, image_data)
                    rec.write_idx(idx, s)
                    idx += 1
    
    print('det_imageset_path', det_imageset_path)
    for line in open(det_imageset_path, 'r').readlines():
        key = line.split(' ')[0]
        if not video_index_dict.has_key(key):
            video_index_dict[key] = idx
            print('recording %d video' % idx)
            rec_index_list.writelines(key + ' ' + str(idx) + ' ' + str(1) + '\n')
            annotation_path = os.path.join(root_path, 'Annotations/DET', key + '.xml')
            image_path = os.path.join(root_path, 'Data/DET', key + '.JPEG')
            image_data = open(image_path, 'rb').read()
            gt_boxes = load_annotations(annotation_path, num_classes=31)
            header = mx.recordio.IRHeader(flag=0, label=gt_boxes, idx=idx, id2=0)
            s = mx.recordio.pack(header, image_data)
            rec.write_idx(idx, s)
            idx += 1
    rec.close()
    rec_index_list.close()
    print('rec done')
                    
def load_annotations(annotation_path, num_classes=31):
    roi_rec = {}
    classes_map =  ['__background__',  # always index 0
                    'n02691156', 'n02419796', 'n02131653', 'n02834778',
                    'n01503061', 'n02924116', 'n02958343', 'n02402425',
                    'n02084071', 'n02121808', 'n02503517', 'n02118333',
                    'n02510455', 'n02342885', 'n02374451', 'n02129165',
                    'n01674464', 'n02484322', 'n03790512', 'n02324045',
                    'n02509815', 'n02411705', 'n01726692', 'n02355227',
                    'n02129604', 'n04468005', 'n01662784', 'n04530566',
                    'n02062744', 'n02391049']
    tree = ET.parse(annotation_path)
    size = tree.find('size')
    roi_rec['height'] = float(size.find('height').text)
    roi_rec['width'] = float(size.find('width').text)
    
    objs = tree.findall('object')
    num_objs = len(objs)

    boxes = np.zeros((num_objs, 4), dtype=np.uint16)
    gt_classes = np.zeros((num_objs), dtype=np.int32)

    valid_objs = np.zeros((num_objs), dtype=np.bool)
    class_to_index = dict(zip(classes_map, range(num_classes)))
    # Load object bounding boxes into a data frame.
    for ix, obj in enumerate(objs):
        bbox = obj.find('bndbox')
        # Make pixel indexes 0-based
        x1 = np.maximum(float(bbox.find('xmin').text), 0)
        y1 = np.maximum(float(bbox.find('ymin').text), 0)
        x2 = np.minimum(float(bbox.find('xmax').text), roi_rec['width'] - 1)
        y2 = np.minimum(float(bbox.find('ymax').text), roi_rec['height'] - 1)
        if not class_to_index.has_key(obj.find('name').text):
            continue
        valid_objs[ix] = True
        cls = class_to_index[obj.find('name').text.lower().strip()]
        boxes[ix, :] = [x1, y1, x2, y2]
        gt_classes[ix] = cls

    boxes = boxes[valid_objs, :]
    gt_classes = gt_classes[valid_objs]

    assert (boxes[:, 2] >= boxes[:, 0]).all()
    if gt_classes.size > 0:
        gt_inds = np.where(gt_classes != 0)[0]
        gt_boxes = np.empty((boxes.shape[0], 5), dtype=np.float32)
        gt_boxes[:,:4] = boxes[gt_inds, :]
        gt_boxes[:,4] = gt_classes[gt_inds]
    else:
        gt_boxes = np.empty((0,5), dtype=np.float32)

    return gt_boxes

    
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--det_imageset_path', help='path to det image set', type=str, default='./ImageSets/DET_train_30classes.txt')
    parser.add_argument('--out_rec_idx_path', help='path to output list that records index of video to rec', type=str, default='.VID_DET_rec_idx_list.txt')
    parser.add_argument('--root_path', help='root path of dataset', type=str, default='.')
    parser.add_argument('--rec_idx_path', help='path of record index file', type=str, default='.VID_DET_rec_idx.idx')
    parser.add_argument('--rec_data_path', help='path of record data file', type=str, default='.VID_DET_rec_data.rec')

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    img2rec(args.det_imageset_path, args.out_rec_idx_path, args.root_path, args.rec_idx_path, args.rec_data_path)

