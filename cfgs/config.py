"""
Config File
"""


config = {

    "voc_baseline": {
        # lr and general config
        'base_lr': 1e-2,
        "lr_decay": [60000, 80000],
        "warmup": 500,
        "workers": 8,
        "num_classes": 4,
        "weight_decay": 1e-4,
        "epochs": 200,

        "basemodel_path": '/home/thcheng/workspace/.torch/models/resnet50-19c8e357.pth',
        "data_dir": "/home/thcheng/workspace/VOCdevkit",

        # anchor config
        "positive_anchor_threshold": 0.5,
        "negative_anchor_threshold": 0.4,
        "anchor_sizes": [2 ** 0, 2 ** (1 / 3), 2 ** (2 / 3)],
        "aspect_ratios": [0.5, 1, 2],
        "anchor_areas": [32 ** 2, 64 ** 2, 128 ** 2, 256 ** 2, 512 ** 2],
        "strides": [8, 16, 32, 64, 128],
        "base_size": 8,

        # dataset
        "image_scales": [600],
        "max_image_size": 1000,

        # test config
        "test_image_size": [600],
        "test_max_image_size": 1000,
        "pre_nms_boxes": 1000,
        "test_nms": 0.5,
        "test_max_boxes": 300,
        "cls_thresh": 0.05,

        # log
        "logdir": "log",
        "tb_dump_dir": "",
        "model_dump_dir": "",
    },

    "coco_baseline": {
        # lr and general config
        'base_lr': 1e-2,
        "lr_decay": [60000, 80000],
        "warmup": 2000,
        "workers": 4,
        "num_classes": 4,
        "weight_decay": 1e-4,
        "epochs": 500,

        "basemodel_path": '/scratch/ganswindt/retinanet/resnet50-19c8e357.pth',
        "data_dir": '/scratch/ganswindt/retinanet/COCO/DIR/',

        # anchor config
        #"positive_anchor_threshold": 0.7,
        #"negative_anchor_threshold": 0.4,
        "positive_anchor_threshold": 0.8,
        "negative_anchor_threshold": 0.4,
        #"anchor_sizes": [2 ** 0, 2 ** (1 / 3), 2 ** (2 / 3)],
        #"aspect_ratios": [0.5, 1, 2],
        #"anchor_sizes": [2 ** 0, 2 ** (1 / 4), 2 ** (2 / 4), 2 ** (3 / 4)],
        "anchor_sizes": [2, 4, 8, 16, 32],
        "aspect_ratios": [0.1, 0.25, 0.5, 1, 2],
        #"aspect_ratios": [0.5, 1, 2, 3],
        "anchor_areas": [8 ** 2, 16 ** 2, 32 ** 2, 64 ** 2, 128 ** 2],
        "strides": [8, 16, 32, 64, 128],
        #"anchor_areas": [4 ** 2, 8 ** 2, 16 ** 2, 32 ** 2, 64 ** 2],
        #"strides": [4, 8, 16, 32, 64],
        #"anchor_areas": [16 ** 2, 32 ** 2, 64 ** 2, 128 ** 2, 512 ** 2],
        #"strides": [16, 32, 64, 128, 512],
        "base_size": 8,

        # dataset
        "image_scales": [800],
        "max_image_size": 800,

        # test config
        "test_image_size": [800],
        "test_max_image_size": 800,
        "pre_nms_boxes": 1000,
        "test_nms": 0.8,
        #"test_nms": 0.3,
        "test_max_boxes": 100,
        "cls_thresh": 0.7,

        # log
        "logdir": '/scratch/ganswindt/retinanet/log_2906',
        "tb_dump_dir": "/scratch/ganswindt/retinanet/log_2906/2906_1",
        "model_dump_dir": "",
    }

}
