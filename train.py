"""

Training RetinaNet


"""
import os
import tqdm
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from models import retina
from datasets import coco, voc, minibatch
from torch.utils.data import DataLoader
from lib.det_ops.loss import SigmoidFocalLoss, SmoothL1Loss
from IPython import embed
from torch.nn.utils import clip_grad
import tensorboardX
from utils import logger
from cfgs import config as cfg
import tensorflow as tf
from torchvision.utils import make_grid

import matplotlib.pyplot as plt

import datetime

def initialize(config, args):

    logdir = config['logdir']
    if not os.path.exists(logdir):
        os.mkdir(logdir)
    if not os.path.exists(os.path.join(logdir, args.experiment)):
        os.mkdir(os.path.join(logdir, args.experiment))

    model_dump_dir = os.path.join(logdir, args.experiment, 'model_dump')
    #tb_dump = os.path.join(logdir, args.experiment, 'tb_dump')
    tb_dump = config['tb_dump_dir']

    if not os.path.exists(model_dump_dir):
        os.mkdir(model_dump_dir)

    if not os.path.exists(tb_dump):
        os.mkdir(tb_dump)

    #config['tb_dump_dir'] = tb_dump
    config['model_dump_dir'] = model_dump_dir


def learning_rate_decay(optimizer, step, config):
    base_lr = config['base_lr']
    lr = base_lr
    if step <= config['warmup']:
        lr = (lr - 1e-4)*step/config['warmup'] + 1e-4
    if step >= config['lr_decay'][0]:
        lr = base_lr * 0.1
    if step >= config['lr_decay'][1]:
        lr = base_lr * 0.01

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return lr


def train(args, config):
    anchor_scales = config['anchor_sizes']
    anchor_apsect_ratios = config['aspect_ratios']
    num_anchors = len(anchor_scales) * len(anchor_apsect_ratios)

    model = retina.RetinaNet(config['num_classes']-1, num_anchors, config['basemodel_path']).cuda()
    model = nn.DataParallel(model, device_ids=list(range(args.device)))

    if args.dataset == 'COCO':
        train_dataset = coco.COCODetection(dataroot=config['data_dir'], imageset='train', config=config, training=True)
    elif args.dataset == 'VOC':
        train_dataset = voc.VOC2012(dataroot=config['data_dir'], imageset='train', config=config)
    else:
        raise NotImplemented()

    collate_minibatch = minibatch.create_minibatch_func(config)

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size*args.device,
        shuffle=True,
        num_workers=config['workers'],
        collate_fn=collate_minibatch
    )

    writer = tensorboardX.SummaryWriter(config['tb_dump_dir'])
    # torch model

    optimizer = optim.SGD(lr=config['base_lr'], params=model.parameters(),
                          weight_decay=config['weight_decay'], momentum=0.9)

    cls_criterion = SigmoidFocalLoss().cuda()
    box_criterion = SmoothL1Loss().cuda()

    start_epoch = 0
    global_step = 0

    # Load state dict from saved model
    if len(args.continue_path) > 0:
        model_state, optimizer_state, epoch, step = logger.load_checkpoints(args.continue_path)
        model.module.load_state_dict(model_state)
        optimizer.load_state_dict(optimizer_state)
        global_step = step+1
        start_epoch = epoch + 1
    for p in model.module.modules():
        if p.__class__.__name__ == 'BatchNorm2d':
            p.eval()
    for epoch in range(start_epoch, config['epochs']):
        losses = []
        data_iter = iter(train_loader)
        pbar = tqdm.tqdm(range(len(train_loader)))
        for i in pbar:
            lr = learning_rate_decay(optimizer, global_step, config)
            img, labels, boxes, anchors = next(data_iter)

            #add image to tensorboard
            tbimg = make_grid(img)
            #writer.add_images('img', img, epoch)

            #writer.add_image_with_boxes('img with BBox', img[0], boxes[0], epoch)

            img = img.cuda()
            labels = labels.long().cuda()
            boxes = boxes.cuda()
            cls_outputs, bbox_outputs = model(img)

            cls_loss = cls_criterion(cls_outputs, labels)
            box_loss = box_criterion(bbox_outputs, boxes, labels)
            loss = cls_loss + box_loss

            optimizer.zero_grad()
            loss.backward()
            clip_grad.clip_grad_norm_(model.parameters(), 30)

            optimizer.step()
            writer.add_scalar('train/box_loss', box_loss.item(), global_step)
            writer.add_scalar('train/cls_loss', cls_loss.item(), global_step)
            global_step += 1
            pbar.set_description('e:{} i:{} loss:{:.3f} cls_loss:{:.3f} box_loss:{:.3f} lr:{}'.format(
                epoch, i + 1, loss.item(), cls_loss.item(), box_loss.item(), lr
            ))
            losses.append(loss.item())

            scores = cls_outputs
            max_score, max_labels = torch.max(scores[0], dim=1)


            """
            good_anchors = []
            for cls in range(config['num_classes'] - 1):

                # filter predictions through 'classification threshold'
                score = scores[:, cls]
                cls_inds = score > config['cls_thresh']
                # current class has the max score over all classes
                max_inds = max_labels == cls
                cls_inds = max_inds * cls_inds
                if cls_inds.sum() < 1:
                    continue

                # _boxes [K, 4]
                good_anchors.append(anchors[cls_inds])

            good_anchors = torch.cat(good_anchors, dim=0)
            """

            good_anchors = anchors[0][max_score > config['cls_thresh']]

            # learning rate decay

        if epoch % 25 == 0:
            writer.add_images('img', img, epoch)
            writer.add_image_with_boxes('img with gt BBox', img[0], anchors[0][labels[0]>0], epoch)
            # TODO: plot only good boxes
            writer.add_image_with_boxes('img with predicted BBox', img[0], good_anchors, epoch)

        print("e:{} loss: {}".format(epoch, np.mean(losses)))
        logger.save_checkpoints(model.module, optimizer, epoch, global_step,
                                path=os.path.join(config['model_dump_dir'],
                                                  'epoch-{}-iter-{}.pth'.format(epoch, global_step)))


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = '2'

    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--device', type=int, default=1, help='training with ? GPUs')
    parser.add_argument('-b', '--batch_size', type=int, default=4, help='training batch size per GPU')
    parser.add_argument('-c', '--continue_path', type=str, default='', help='continue model parameters')
    parser.add_argument('-e', '--experiment', type=str, default='coco_baseline',
                        help='experiment name, correspond to `config.py`')
    parser.add_argument('-ds', '--dataset', type=str, default='COCO', help='dataset')

    _args = parser.parse_args()
    config = cfg.config[_args.experiment]
    initialize(config, _args)
    train(_args, config)

    print('Finished at: ', datetime.datetime.now().time())
