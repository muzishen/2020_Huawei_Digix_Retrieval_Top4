import torch
from .ranger import Ranger
from .sgd_gc import SGD, SWA
import logging
def make_optimizer(cfg, model):
    logger = logging.getLogger("reid_baseline.train")
    params = []
    for key, value in model.named_parameters():
        if not value.requires_grad:
            continue
        lr = cfg.SOLVER.BASE_LR
        weight_decay = cfg.SOLVER.WEIGHT_DECAY
        if "bias" in key:
            lr = cfg.SOLVER.BASE_LR * cfg.SOLVER.BIAS_LR_FACTOR
            weight_decay = cfg.SOLVER.WEIGHT_DECAY_BIAS
        if cfg.SOLVER.LARGE_FC_LR:
            if "classifier" in key or "arcface" in key:
                lr = cfg.SOLVER.BASE_LR * cfg.SOLVER.FC_LR_TIMES
                logger.info('Using {} times learning rate for fc'.format(cfg.SOLVER.FC_LR_TIMES))
        if "gap" in key:
            lr = cfg.SOLVER.BASE_LR * 10
            weight_decay = 0

        params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]
    if cfg.SOLVER.OPTIMIZER_NAME == 'SGD':
        optimizer = getattr(torch.optim, cfg.SOLVER.OPTIMIZER_NAME)(params, momentum=cfg.SOLVER.MOMENTUM)
    elif cfg.SOLVER.OPTIMIZER_NAME == 'Ranger':
        optimizer = Ranger(params)
        logger.info('using Ranger for optimizer ')
    elif cfg.SOLVER.OPTIMIZER_NAME == 'SGD_GC':
        optimizer = SGD(params, momentum=cfg.SOLVER.MOMENTUM, weight_decay=cfg.SOLVER.WEIGHT_DECAY,
         use_gc=cfg.SOLVER.USE_GC, gc_conv_only=cfg.SOLVER.GC_CONV_ONLY)
    elif cfg.SOLVER.OPTIMIZER_NAME == 'SGD_GC_SWA':
        optimizer = SGD(params, momentum=cfg.SOLVER.MOMENTUM)
        optimizer = SWA(optimizer, swa_start=0, swa_freq=1)
    else:
        optimizer = getattr(torch.optim, cfg.SOLVER.OPTIMIZER_NAME)(params)
        
    return optimizer
