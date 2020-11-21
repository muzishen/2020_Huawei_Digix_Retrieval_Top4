import torch.nn.functional as F
import logging
from .softmax_loss import CrossEntropyLabelSmooth
from .triplet_loss import TripletLoss


def make_loss(cfg, num_classes):    # modified by gu
    feat_dim = 2048
    logger = logging.getLogger("reid_baseline.train")
    if 'triplet' in cfg.MODEL.METRIC_LOSS_TYPE:
        if cfg.MODEL.NO_MARGIN:
            triplet = TripletLoss(mining_method=cfg.SOLVER.HARD_EXAMPLE_MINING_METHOD)
            logger.info("using soft margin triplet loss for training, mining_method:{}".format(cfg.SOLVER.HARD_EXAMPLE_MINING_METHOD))
        else:
            triplet = TripletLoss(cfg.SOLVER.MARGIN, mining_method=cfg.SOLVER.HARD_EXAMPLE_MINING_METHOD)  # triplet loss
            logger.info("using Triplet Loss for training with margin:{}, mining_method:{}".format(cfg.SOLVER.MARGIN, cfg.SOLVER.HARD_EXAMPLE_MINING_METHOD))

    if cfg.MODEL.IF_LABELSMOOTH == 'on':
        id_loss_func = CrossEntropyLabelSmooth(num_classes=num_classes)
        logger.info("label smooth on, numclasses:{}".format(num_classes))
    else:
        id_loss_func = F.cross_entropy

    def loss_func(score, feat, target):   
        if cfg.MODEL.METRIC_LOSS_TYPE == 'triplet':
            return cfg.MODEL.ID_LOSS_WEIGHT * id_loss_func(score, target) + \
                    cfg.MODEL.TRIPLET_LOSS_WEIGHT * triplet(feat, target)[0]
        elif cfg.MODEL.METRIC_LOSS_TYPE == 'softmax':
            return cfg.MODEL.ID_LOSS_WEIGHT * id_loss_func(score, target)
        else:
            print('unexpected loss type')
    
    def two_head_loss(score, feat, channel_head_feature, target):   
        if cfg.MODEL.METRIC_LOSS_TYPE == 'triplet':
            return  cfg.MODEL.ID_LOSS_WEIGHT * id_loss_func(score, target) + \
                    cfg.MODEL.TRIPLET_LOSS_WEIGHT * triplet(feat, target)[0] + \
                    cfg.MODEL.TRIPLET_LOSS_WEIGHT * triplet(channel_head_feature, target)[0]

        elif cfg.MODEL.METRIC_LOSS_TYPE == 'softmax':
            return cfg.MODEL.ID_LOSS_WEIGHT * id_loss_func(score, target)
        else:
            print('unexpected loss type')

    if cfg.MODEL.CHANNEL_HEAD:
        loss_func = two_head_loss
        return loss_func
    else:
        loss_func = loss_func 
        return loss_func