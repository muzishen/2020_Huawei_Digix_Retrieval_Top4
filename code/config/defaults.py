from yacs.config import CfgNode as CN
import os
# -----------------------------------------------------------------------------
# Convention about Training / Test specific parameters
# -----------------------------------------------------------------------------
# Whenever an argument can be either used for training or for testing, the
# corresponding name will be post-fixed by a _TRAIN for a training parameter,
# or _TEST for a test-specific parameter.
# For example, the number of images during training will be
# IMAGES_PER_BATCH_TRAIN, while the number of images for testing will be
# IMAGES_PER_BATCH_TEST

# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------

_C = CN()

_C.MODEL = CN()
# Model's Mode
_C.MODEL.MODE = 'train'
# Using cuda or cpu for training
_C.MODEL.DEVICE = "cuda"
# ID number of GPU
_C.MODEL.DEVICE_ID = '0'
# Name of backbone
_C.MODEL.NAME = 'resnet101_ibn_a'
# Last stride of backbone
_C.MODEL.LAST_STRIDE = 1
# Path to pretrained model of backbone
_C.MODEL.PRETRAIN_PATH = '/cache/resnet101_ibn_a.pth'
# Use ImageNet pretrained model to initialize backbone or use self trained model to initialize the whole model
# Options: 'imagenet' , 'self' , 'finetune'
_C.MODEL.PRETRAIN_CHOICE = 'imagenet'
# If train with BNNeck, options: 'bnneck' or 'no'
_C.MODEL.NECK = 'bnneck'
# If train loss include center loss, options: 'yes' or 'no'. Loss with center loss has different optimizer configuration
_C.MODEL.IF_WITH_CENTER = 'no'
# The loss type of metric loss
# options:['triplet'](without center loss) or ['center','triplet_center'](with center loss)
_C.MODEL.METRIC_LOSS_TYPE = 'triplet'
# For example, if loss type is cross entropy loss + triplet loss + center loss
# the setting should be: _C.MODEL.METRIC_LOSS_TYPE = 'triplet_center' and _C.MODEL.IF_WITH_CENTER = 'yes'
# If train with soft triplet loss, options: 'True', 'False'
_C.MODEL.NO_MARGIN = False
# If train with label smooth, options: 'on', 'off'
_C.MODEL.IF_LABELSMOOTH = 'on'
# If train with arcface loss, options: 'True', 'False'
_C.MODEL.COS_LAYER = False
# Channel head
_C.MODEL.CHANNEL_HEAD = True 
# Frozen layers of backbone
_C.MODEL.FROZEN = -1
# Frozen layers of backbone
_C.MODEL.POOLING_METHOD = 'GeM'
_C.MODEL.ID_LOSS_TYPE = 'softmax'
_C.MODEL.ID_LOSS_WEIGHT = 1.0
_C.MODEL.TRIPLET_LOSS_WEIGHT = 1.0
#_C.MODEL.WEIGHTED_TRIPLET = False
_C.MODEL.THRESH = 0.3
# -----------------------------------------------------------------------------
# INPUT
# -----------------------------------------------------------------------------
_C.INPUT = CN()

# Size of the image during training
_C.INPUT.SIZE_TRAIN = (576, 576)
# Size of the image during test
_C.INPUT.SIZE_TEST =  (576, 576)
# Random probability for image horizontal flip
_C.INPUT.PROB = 0.5
#cutout length
_C.INPUT.LENGTH = 256
# Random probability for random erasing
_C.INPUT.RE_PROB = 0.5
# Values to be used for image normalization
_C.INPUT.PIXEL_MEAN = [0.485, 0.456, 0.406]
# Values to be used for image normalization
_C.INPUT.PIXEL_STD = [0.229, 0.224, 0.225]

# Value of padding size
_C.INPUT.PADDING = 20

#filter_data
_C.INPUT.FILTER = False
_C.INPUT.LABEL_PATH = "/cache/label.txt"
_C.INPUT.FILTER_DATA_PATH = "/cache/train"
_C.INPUT.FILTER_DATA_NUM = 4
# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
_C.DATASETS = CN()
# List of the dataset names for training, as present in paths_catalog.py
_C.DATASETS.NAMES = ('digital')
# Root directory where datasets should be used (and downloaded if not found)
_C.DATASETS.ROOT_DIR = ('/cache')

_C.DATASETS.HARD_AUG = True
# Use Augmixup
_C.INPUT.AUGMIX = False
# Random color jitter
_C.INPUT.CJ = CN()
#_C.INPUT.CJ.ENABLED = True
_C.INPUT.CJ.PROB = 0.5
_C.INPUT.CJ.BRIGHTNESS = 0.15
_C.INPUT.CJ.CONTRAST = 0.15
_C.INPUT.CJ.SATURATION = 0.1
_C.INPUT.CJ.HUE = 0.1

# -----------------------------------------------------------------------------
# DataLoader
# -----------------------------------------------------------------------------
_C.DATALOADER = CN()
# Number of data loading threads
_C.DATALOADER.NUM_WORKERS = 16
# Sampler for data loading
_C.DATALOADER.SAMPLER = 'softmax_triplet'
# Number of instance for one batch
_C.DATALOADER.NUM_INSTANCE = 4

# ---------------------------------------------------------------------------- #
# Solver
# ---------------------------------------------------------------------------- #
_C.SOLVER = CN()
# Name of optimizer
_C.SOLVER.OPTIMIZER_NAME = 'SGD_GC'
# Number of max epoches
_C.SOLVER.MAX_EPOCHS = 75
# Base learning rate
_C.SOLVER.BASE_LR = 0.01
# Whether using larger learning rate for fc layer
_C.SOLVER.LARGE_FC_LR = True
#the time learning rate of fc layer
_C.SOLVER.FC_LR_TIMES = 2
# Factor of learning bias
_C.SOLVER.BIAS_LR_FACTOR = 1
# Momentum
_C.SOLVER.MOMENTUM = 0.9
# Margin of triplet loss sampler method 
_C.SOLVER.HARD_EXAMPLE_MINING_METHOD = 'batch_hard'
# Margin of triplet loss
_C.SOLVER.MARGIN = 0.3

# Settings of weight decay
_C.SOLVER.WEIGHT_DECAY = 0.0005
_C.SOLVER.WEIGHT_DECAY_BIAS = 0.0005
#setting gc
_C.SOLVER.USE_GC = True
_C.SOLVER.GC_CONV_ONLY = False

#lr_scheduler
#lr_scheduler method

# decay rate of learning rate
_C.SOLVER.GAMMA = 0.1
# decay step of learning rate
_C.SOLVER.STEPS = [40, 70]

#Cosine annealing learning rate options
_C.SOLVER.DELAY_ITERS = 30
_C.SOLVER.ETA_MIN_LR = 1e-7

# warm up factor
_C.SOLVER.WARMUP_FACTOR = 0.1
#  warm up epochs
_C.SOLVER.WARMUP_EPOCHS = 10
# method of warm up, option: 'constant','linear'
_C.SOLVER.WARMUP_METHOD = "linear"


_C.SOLVER.COSINE_MARGIN = 0.4
_C.SOLVER.COSINE_SCALE = 30


# epoch number of saving checkpoints
_C.SOLVER.CHECKPOINT_PERIOD = _C.SOLVER.MAX_EPOCHS
# iteration of display training log
_C.SOLVER.LOG_PERIOD = 600
# epoch number of validation
_C.SOLVER.EVAL_PERIOD = 10
# _C.SOLVER.FP16 = True
# Number of images per batch
# This is global, so if we have 8 GPUs and IMS_PER_BATCH = 16, each GPU will
# see 2 images per batch
_C.SOLVER.IMS_PER_BATCH = 40
_C.SOLVER.SEED = 1234

# ---------------------------------------------------------------------------- #
# TEST
# ---------------------------------------------------------------------------- #

_C.TEST = CN()
# Number of images per batch during test
_C.TEST.IMS_PER_BATCH = 256


# Path to trained model
_C.TEST.WEIGHT = _C.SOLVER.MAX_EPOCHS
# Which feature of BNNeck to be used for test, before or after BNNneck, options: 'before' or 'after'
_C.TEST.NECK_FEAT = 'after'
# Whether feature is nomalized before test, if yes, it is equivalent to cosine distance
_C.TEST.FEAT_NORM = 'yes'
_C.TEST.DIST_MAT = "dist_mat.npy"
# If test with re-ranking, options: 'True','False'
_C.TEST.RE_RANKING = False
# K1, K2, LAMBDA
_C.TEST.RE_RANKING_PARAMETER = [20, 6, 0.3]
# ---------------------------------------------------------------------------- #
# Misc options
# ---------------------------------------------------------------------------- #
# Path to checkpoint and saved log of trained model
_C.OUTPUT_DIR = "./log"
if not os.path.isdir(_C.OUTPUT_DIR):
    os.makedirs(_C.OUTPUT_DIR)