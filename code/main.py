"""
@author:  muzishen
@contact: shenfei140721@126.com
"""

import os
import torch
from config import cfg
from datasets.make_dataloader import make_dataloader
# from loss import Loss
# from utils.lr_scheduler import LRScheduler
# from torch.backends import cudnn
# import torch.nn as nn
# import numpy as np
# import random
# from torch.cuda.amp import autocast as autocast, GradScaler
#
# cudnn.benchmark = True
os.environ['CUDA_VISIBLE_DEVICES'] = cfg.MODEL.DEVICE_ID

from tools.train import train
from tools.test import test
from filter_data import filter_data


if __name__ == '__main__':

    if cfg.INPUT.FILTER:
        filter_data(cfg.INPUT.LABEL_PATH, cfg.INPUT.FILTER_DATA_PATH, cfg.INPUT.FILTER_DATA_NUM)

    train_loader, val_loader, num_query, num_classes, query_name, gallery_name = make_dataloader(cfg)
    if cfg.MODEL.MODE == 'train':
######################################### resume model ###################################
        train(train_loader, num_classes)

        with torch.no_grad():
            test(val_loader, num_query, query_name, gallery_name, num_classes)

    if cfg.MODEL.MODE == 'evaluate':
        with torch.no_grad():
            test(val_loader, num_query, query_name, gallery_name, num_classes)


