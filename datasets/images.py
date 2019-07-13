
import glob
import os
import numpy as np
import cv2
import torch
from datasets.utils import normalize_image, get_im_scale
from torch.utils.data import Dataset
from torchvision.transforms import transforms
from skimage.external.tifffile import imread, imsave
from skimage import img_as_float


class ImageData(Dataset):
    """
    ImageData:

    dataroot [image folder]
    config [config file]
    """

    def get_ids(self, paths):
        tmp = glob.glob(paths)
        ids = []
        for t in tmp:
            ids.append(t.split('/')[-1].split(".t")[0])
        return ids

    def __init__(self, dataroot, config):
        self.config = config
        self.images_dir = dataroot

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.ToPILImage(),
            transforms.Resize((1024, 1024)),
            transforms.ToTensor()
        ])
        paths = os.path.join(dataroot, '*')
        self.img_ids = self.get_ids(paths)

    def __len__(self):
        return len(self.img_ids)

    def get_img_path(self, img_id):
        tmp = img_id + '.tif'
        filepath = os.path.join(self.images_dir, tmp)
        return filepath

    def __getitem__(self, idx):
        im_id = self.img_ids[idx]
        img_path = self.get_img_path(im_id)

        img = imread(img_path)
        img = np.array(img).astype('float32')
        img /= np.max(img)

        if len(img.shape) is 2:
            img = np.expand_dims(img, axis=2)
            img = np.repeat(img, 3, axis=2)

        img = np.array(img).astype('float32')
        h, w = img.shape[:2]
        resize_h, resize_w, scale = get_im_scale(h, w, target_size=self.config['test_image_size'][0],
                                                             max_size=self.config['test_max_image_size'])
        img = cv2.resize(img, (resize_w, resize_h))
        img = img.transpose(2, 0, 1)
        img = torch.Tensor(img)
        return img, im_id, scale, (h, w)
