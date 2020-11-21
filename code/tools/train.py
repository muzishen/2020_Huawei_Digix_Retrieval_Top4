import os
from torch.backends import cudnn
from utils.logger import setup_logger
from model import make_model
from solver import make_optimizer, WarmupMultiStepLR, WarmupCosineAnnealingLR
from loss import make_loss
from processor import do_train
import random
import torch
import numpy as np
import os
import argparse
from config import cfg


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def train(train_loader, num_classes):
    parser = argparse.ArgumentParser(description="ReID Baseline Training")
    parser.add_argument(
        "--config_file", default="", help="path to config file", type=str
    )
    parser.add_argument("opts", help="Modify config options using the command-line", default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()

    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    set_seed(cfg.SOLVER.SEED)

    output_dir = cfg.OUTPUT_DIR
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    logger = setup_logger("reid_baseline", output_dir, if_train=True)
    logger.info("Saving model in the path :{}".format(cfg.OUTPUT_DIR))
    logger.info(args)

    if args.config_file != "":
        logger.info("Loaded configuration file {}".format(args.config_file))
        with open(args.config_file, 'r') as cf:
            config_str = "\n" + cf.read()
            logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))

    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.MODEL.DEVICE_ID


    if cfg.MODEL.PRETRAIN_CHOICE == 'finetune':
        model = make_model(cfg, num_class=num_classes)
        model.load_param_finetune(cfg.MODEL.PRETRAIN_PATH)
        print('Loading pretrained model for finetuning......')
    else:
        model = make_model(cfg, num_class=num_classes)

    loss_func = make_loss(cfg, num_classes=num_classes)

    optimizer = make_optimizer(cfg, model)
    
    scheduler = WarmupCosineAnnealingLR(optimizer, cfg.SOLVER.MAX_EPOCHS,  cfg.SOLVER.DELAY_ITERS, cfg.SOLVER.ETA_MIN_LR, 
                                cfg.SOLVER.WARMUP_FACTOR, cfg.SOLVER.WARMUP_EPOCHS, cfg.SOLVER.WARMUP_METHOD)
    logger.info("use WarmupCosineAnnealingLR, delay_step:{}".format(cfg.SOLVER.DELAY_ITERS))

    do_train(
        cfg,
        model,
        train_loader,
        optimizer,
        scheduler,  # modify for using self trained model
        loss_func
    )
