"""

Test scripts

"""
import argparse
import json
import tqdm
import torch
import os
import numpy as np
from lib.det_ops.anchors import compute_anchor_whs, generate_anchors
from lib.bbox import bbox, box_transform
from lib.nms import nms
from utils.logger import load_checkpoints
from models import retina
from IPython import embed
from datasets import voc, coco, images
from cfgs import config as cfg
from torchvision.datasets import ImageFolder

import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)


def inference(model, dataset, anchor_wh, strides, result_file, config):

    model.eval()
    num_samples = len(dataset)
    pbar = tqdm.tqdm(range(num_samples))

    if os.path.exists(result_file):
        try:
            os.remove(result_file)
        except:
            print("Error while deleting current file and opening new file: ", result_file)

    results = []

    with torch.no_grad():
        for i, idx in enumerate(pbar):
            img, im_name, scale, im_size = dataset[idx]

            img = img.cuda()
            cls_pred, bbox_pred = model(img.unsqueeze(0))
            scores = cls_pred.sigmoid()
            # bbox [N, 4]
            bbox_pred = bbox_pred[0]
            # cls [N, C]
            scores = scores[0]

            h, w = img.shape[1], img.shape[2]

            anchors = generate_anchors(anchor_wh, input_size=np.array([h, w]),
                                       strides=strides)
            anchors = anchors.cuda()

            # transform to bboxes
            boxes = box_transform.bbox_transform_inv(anchors, bbox_pred.squeeze())
            boxes = boxes/scale

            filter_boxes_inds_x = boxes[:, 0] >= boxes[:, 2]
            filter_boxes_inds_y = boxes[:, 1] >= boxes[:, 3]
            filter_boxes_inds = (1 - filter_boxes_inds_x) * (1 - filter_boxes_inds_y)
            boxes = boxes[filter_boxes_inds]
            scores = scores[filter_boxes_inds]

            scores = scores.squeeze()

            result_boxes = []  # []
            # every class
            # 1. max detection score
            # 2. score thresh
            # 3. do nms
            # 4. top k
            max_labels = torch.argmax(scores, dim=1)

            for cls in range(config['num_classes']-1):

                # filter predictions through 'classification threshold'
                score = scores[:, cls]
                cls_inds = score > config['cls_thresh']
                # current class has the max score over all classes
                max_inds = max_labels == cls
                cls_inds = max_inds * cls_inds
                if cls_inds.sum() < 1:
                    continue
                # score [K]
                score = score[cls_inds]

                # _boxes [K, 4]
                _boxes = boxes[cls_inds]

                # NMS remove duplicate
                keep = nms(torch.cat([_boxes, score.unsqueeze(1)], 1), config['test_nms'])

                score = score[keep]
                _boxes = _boxes[keep]

                for i in range(_boxes.shape[0]):
                    result_boxes.append((cls, score[i].item(), _boxes[i].cpu().data.numpy().tolist()))

            # Keep Max Num Boxes
            if len(result_boxes) > config['test_max_boxes']:
                result_boxes = sorted(result_boxes, key=lambda x: x[1], reverse=True)
                result_boxes = result_boxes[:config['test_max_boxes']]
            pbar.set_description('im_det:{}/{}'.format(idx, num_samples))

            if len(result_boxes) == 0:
                continue

            for j in range(len(result_boxes)):
                cls, s, b, = result_boxes[j]
                if s > 0.75:
                    pred = dict()

                    pred['image_id'] = im_name
                    pred['score'] = s
                    pred['category_id'] = cls + 1
                    pred['bbox'] = b
                    results.append(pred)

    with open(result_file, 'a+') as f:
        s = json.dumps(results)
        f.write(s)
    print("JSON Finished!")
    print('File written to: ', result_file)


def load_img(path):
    npi = imread(path)
    return Image.fromarray(npi)


def validate(args, config):

    anchor_scales = config['anchor_sizes']
    anchor_apsect_ratios = config['aspect_ratios']
    num_anchors = len(anchor_scales) * len(anchor_apsect_ratios)

    model = retina.RetinaNet(config['num_classes']-1, num_anchors, config['basemodel_path']).cuda()

    model_path = args.model_path
    print('Model Path: ', model_path)
    output_file = args.output
    if args.dataset == 'VOC':
        dataset = voc.VOC2012(dataroot=config['data_dir'], imageset=args.imageset, config=config)
    elif args.dataset == 'COCO':
        dataset = coco.COCODetection(dataroot=config['data_dir'], imageset=args.imageset, config=config, training=False)
    elif args.dataset == 'images':
        dataset = images.ImageData(dataroot=args.path, config=config)
    else:
        raise NotImplemented()
    print('Dataset: ', len(dataset))
    state_dict, _, _, _ = load_checkpoints(model_path)
    model.load_state_dict(state_dict)

    anchor_whs = compute_anchor_whs(len(config['strides']), areas=config['anchor_areas'],
                                    aspect_ratios=anchor_apsect_ratios,
                                    sizes=anchor_scales)

    inference(model, dataset, anchor_whs, config['strides'], result_file=output_file, config=config)


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = '2'

    parser = argparse.ArgumentParser('Test RetinaNet')
    parser.add_argument('-o', '--output', type=str, default='./result.json', help='output file path')
    parser.add_argument('-m', '--model_path', type=str, help='saved model path')
    parser.add_argument('-i', '--imageset', type=str, default='val', help='imageset to predict')
    parser.add_argument('-p', '--path', type=str, default='',
                        help='path, where to find the image set')
    parser.add_argument('-e', '--experiment', type=str, default='coco_baseline',
                        help='experiment name, correspond to `config.py`')
    parser.add_argument('-ds', '--dataset', type=str, default='COCO', help='dataset')

    _args = parser.parse_args()
    config = cfg.config[_args.experiment]
    _args = parser.parse_args()
    validate(_args, config)
















