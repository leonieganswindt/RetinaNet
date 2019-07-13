"""

"""
import os
import cv2
import torch
import random
import numpy as np
from PIL import Image, ImageDraw
import matplotlib
matplotlib.use('Agg')
import pycocotools.coco as COCO
from datasets.utils import normalize_image, get_im_scale
from torch.utils.data import Dataset
from torchvision.transforms import transforms
from skimage.external.tifffile import imread, imsave
from skimage import img_as_float

CLASSES = (
    'background', 'mating', 'single_cell', 'crowd')


class COCODetection(Dataset):
    """ COCO Detection Dataset

    dataroot [annotations, train, val]
    imageset [train, val]
    """
    def __init__(self, dataroot, config, imageset='train', training=True):
        assert imageset == 'train' or 'val'
        # train/ val
        self.imageset = imageset
        self.config = config
        self.training = training
        self.images_dir = os.path.join(dataroot, imageset)
        annotation_path = os.path.join(dataroot, 'annotations', 'instances_{}.json'.format(imageset))
        self.coco_helper = COCO.COCO(annotation_path)
        self.img_ids = list(self.coco_helper.imgs.keys())

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.ToPILImage(),
            transforms.Resize((1024, 1024)),
            transforms.ToTensor()
        ])
        catids = self.coco_helper.getCatIds()
        catids = [0] + catids
        self.catid2id = dict(zip(catids, range(len(catids))))
        self.id2catid = dict(zip(range(len(catids)), catids))

    def get_categories(self):
        ids = self.coco_helper.getCatIds()
        return ids

    def _get_category_map(self):
        cats = self.coco_helper.loadCats(self.coco_helper.getCatIds())
        cats = [(x['name'], x['id']) for x in cats]
        cat_id_map = dict(cats)
        return cat_id_map

    def load_annotation(self, img_id):
        annotation = self.coco_helper.loadAnns(self.coco_helper.getAnnIds(img_id))
        boxes = [(ann['bbox'], self.catid2id[ann['category_id']]) for ann in annotation]
        return boxes

    def get_img_path(self, img_id):
        info = self.coco_helper.imgs[img_id]
        filename = info['file_name']
        filepath = os.path.join(self.images_dir, filename)
        return filepath

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        im_id = self.img_ids[idx]
        img_path = self.get_img_path(im_id)

        img = imread(img_path)
        img = np.array(img).astype('float32')
        img /= np.max(img)

        if len(img.shape) is 2:
            img = np.expand_dims(img, axis=2)
            img = np.repeat(img, 3, axis=2)

        source_image = Image.fromarray(img[:,:,0] * 255)
        source_image = source_image.convert('RGB')
        imsave('./input_image', img[:, :, 1].astype(np.float32), imagej=True)

        if not self.training:
            img = np.array(img).astype('float32')
            h, w = img.shape[:2]
            resize_h, resize_w, scale = get_im_scale(h, w, target_size=self.config['test_image_size'][0],
                                                 max_size=self.config['test_max_image_size'])
            img = cv2.resize(img, (resize_w, resize_h))
            img = img.transpose(2, 0, 1)
            img = torch.Tensor(img)
            return img, im_id, scale, (h, w)

        annotations = self.load_annotation(im_id)

        boxes = np.array([x[0] for x in annotations], dtype='float32')
        if boxes.shape[0] == 0:
            boxes = np.array([[0, 0, 0, 0]], dtype='float32')
        # x1,y1,w,h -> x1,y1,x2,y2
        boxes[:, 2:] = boxes[:, 2:] + boxes[:, :2] - 1


        draw = ImageDraw.Draw(source_image)

        for i, bbox in enumerate(boxes):
            draw.rectangle((bbox[0], bbox[1], bbox[2], bbox[3]), outline='red')
        source_image.save('./input_image_boxes', 'JPEG')

        labels = torch.LongTensor([x[1] for x in annotations])

        # labels N
        # onehot_labels = torch.zeros((labels.shape[0], config.num_classes))
        # onehot_labels[range(labels.shape[0]), labels] = 1
        # B*N
        return img, labels, boxes
