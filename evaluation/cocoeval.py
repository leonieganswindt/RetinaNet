"""

"""
import sys
import os
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
sys.path.append('../')
import pycocotools.coco as COCO
import pycocotools.cocoeval as COCOeval
from cfgs import config as cfg
import os


def coco_bbox_eval(result_file, annotation_file):

    ann_type = 'bbox'
    coco_gt = COCO.COCO(annotation_file)
    coco_dt = coco_gt.loadRes(result_file)
    print('COCO_dt: ', coco_dt)
    cocoevaler = COCOeval.COCOeval(coco_gt, coco_dt, ann_type)
    cocoevaler.evaluate()
    cocoevaler.accumulate()
    cocoevaler.summarize()


def coco_proposal_eval(result_file, annotation_file):

    ann_type = 'bbox'
    coco_gt = COCO.COCO(annotation_file)
    coco_dt = coco_gt.loadRes(result_file)
    cocoevaler = COCOeval.COCOeval(coco_gt, coco_dt, ann_type)
    cocoevaler.params.useCats = 0
    cocoevaler.evaluate()
    cocoevaler.accumulate()
    cocoevaler.summarize()


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = '2'

    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--result', type=str, default='./result.json', help='detection result file')
    parser.add_argument('-a', '--annotation', type=str, default='val', help='COCO groundtruth')
    parser.add_argument('-t', '--type', type=str, default='bbox', help='eval type: [bbox, seg, proposal]')
    _args = parser.parse_args()

    config = cfg.config['coco_baseline']

    filex = 'annotations/instances_' + _args.annotation + '.json'
    annotation_file = os.path.join(config['data_dir'], filex)

    print('Result File used: ', _args.result)

    if _args.type == 'bbox':
        coco_bbox_eval(_args.result, annotation_file)
    elif _args.type == 'proposal':
        coco_proposal_eval(_args.result, annotation_file)
    else:
        pass


