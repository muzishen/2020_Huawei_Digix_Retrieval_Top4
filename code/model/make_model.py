import torch
import torch.nn as nn
from .backbones.resnet import ResNet, BasicBlock, Bottleneck
from loss.metric_learning import Arcface, Cosface, AMSoftmax, CircleLoss
from .backbones.resnet_ibn_a import resnet50_ibn_a,resnet101_ibn_a
from .backbones.resnet_nl_ibn_a import resnet101_nl_ibn_a
from .backbones.se_resnet_ibn_a import se_resnet101_ibn_a
from .backbones.resnet_ibn_b import resnet101_ibn_b
from .backbones.resnest import resnest101
from .backbones.resnext_ibn_a import resnext101_ibn_a
import torch.nn.functional as F
from torch.nn.parameter import Parameter

class GeM(nn.Module):
    def __init__(self, p=3.0, eps=1e-6, freeze_p=True):
        super(GeM, self).__init__()
        self.p = p if freeze_p else Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return F.adaptive_avg_pool2d(x.clamp(min=self.eps).pow(self.p),
                            (1, 1)).pow(1. / self.p)

    def __repr__(self):
        if isinstance(self.p, float):
            p = self.p
        else:
            p = self.p.data.tolist()[0]
        return self.__class__.__name__ +\
               '(' + 'p=' + '{:.4f}'.format(p) +\
               ', ' + 'eps=' + str(self.eps) + ')'

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)

    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)


class Backbone(nn.Module):
    def __init__(self, num_classes, cfg):
        super(Backbone, self).__init__()
        last_stride = cfg.MODEL.LAST_STRIDE
        model_path = cfg.MODEL.PRETRAIN_PATH
        model_name = cfg.MODEL.NAME
        pretrain_choice = cfg.MODEL.PRETRAIN_CHOICE
        self.cos_layer = cfg.MODEL.COS_LAYER
        self.channel_head = cfg.MODEL.CHANNEL_HEAD
        self.neck = cfg.MODEL.NECK
        self.neck_feat = cfg.TEST.NECK_FEAT

        if model_name == 'resnet50':
            self.in_planes = 2048
            self.base = ResNet(last_stride=last_stride,
                               block=Bottleneck, frozen_stages=cfg.MODEL.FROZEN,
                               layers=[3, 4, 6, 3])
            print('using resnet50 as a backbone')
        elif model_name == 'resnet50_ibn_a':
            self.in_planes = 2048
            self.base = resnet50_ibn_a(last_stride)
            print('using resnet50_ibn_a as a backbone')
        elif model_name == 'resnet101_ibn_a':
            self.in_planes = 2048
            self.base = resnet101_ibn_a(last_stride, frozen_stages=cfg.MODEL.FROZEN)
            print('using resnet101_ibn_a as a backbone')
        elif model_name == 'resnet101_nl_ibn_a':
            self.in_planes = 2048
            self.base = resnet101_nl_ibn_a(last_stride, frozen_stages=cfg.MODEL.FROZEN)
        elif model_name == 'se_resnet101_ibn_a':
            self.in_planes = 2048
            self.base = se_resnet101_ibn_a(last_stride)
            print('using se_resnet101_ibn_a as a backbone')
        elif model_name == 'resnet101_ibn_b':
            self.in_planes = 2048
            self.base = resnet101_ibn_b(last_stride)
            print('using resnet101_ibn_b as a backbone')
        elif model_name == 'resnest101':
            self.in_planes = 2048
            self.base = resnest101(last_stride)
            print('using resnest as a backbone')
        elif model_name == 'resnext101_ibn_a':
            self.in_planes = 2048
            self.base = resnext101_ibn_a(last_stride)
            print('using resnext101_ibn_a as a backbone')
        else:
            print('unsupported backbone! but got {}'.format(model_name))

        if pretrain_choice == 'imagenet':
            self.base.load_param(model_path)
            print('Loading pretrained ImageNet model......from {}'.format(model_path))

        if cfg.MODEL.POOLING_METHOD == 'GeM':
            print('using GeM pooling')
            self.gap = GeM()
        else:
            self.gap = nn.AdaptiveAvgPool2d(1)

        self.num_classes = num_classes
        self.ID_LOSS_TYPE = cfg.MODEL.ID_LOSS_TYPE
        if self.ID_LOSS_TYPE == 'arcface':
            print('using {} with s:{}, m: {}'.format(self.ID_LOSS_TYPE,cfg.SOLVER.COSINE_SCALE,cfg.SOLVER.COSINE_MARGIN))
            self.classifier = Arcface(self.in_planes, self.num_classes,
                                      s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
        elif self.ID_LOSS_TYPE == 'cosface':
            print('using {} with s:{}, m: {}'.format(self.ID_LOSS_TYPE,cfg.SOLVER.COSINE_SCALE,cfg.SOLVER.COSINE_MARGIN))
            self.classifier = Cosface(self.in_planes, self.num_classes,
                                      s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
        elif self.ID_LOSS_TYPE == 'amsoftmax':
            print('using {} with s:{}, m: {}'.format(self.ID_LOSS_TYPE,cfg.SOLVER.COSINE_SCALE,cfg.SOLVER.COSINE_MARGIN))
            self.classifier = AMSoftmax(self.in_planes, self.num_classes,
                                        s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
        elif self.ID_LOSS_TYPE == 'circle':
            print('using {} with s:{}, m: {}'.format(self.ID_LOSS_TYPE,cfg.SOLVER.COSINE_SCALE,cfg.SOLVER.COSINE_MARGIN))
            self.classifier = CircleLoss(self.in_planes, self.num_classes,
                                        s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
        else:
            self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)
            self.classifier.apply(weights_init_classifier)

        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)

    def forward(self, x, label=None):  # label is unused if self.cos_layer == 'no'
        x = self.base(x)
        global_feat = self.gap(x)
        global_feat = global_feat.view(global_feat.shape[0], -1)  # flatten to (bs, 2048)
        feat = self.bottleneck(global_feat)
        
        if self.channel_head: 
            channel_head_feature = torch.mean(x, dim=1,keepdim=True)
            channel_head_feature = channel_head_feature.view(channel_head_feature.shape[0], -1)  # flatten to (bs, 2048)
           # print('channel_head_feature',channel_head_feature.shape)       
            #channel_head_feat =  self.bottleneck(channel_head_feature)

        if self.neck == 'no':
            feat = global_feat
        elif self.neck == 'bnneck':
            feat = self.bottleneck(global_feat)
            # if self.channel_head:
            #     channel_head_feat =  self.bottleneck(channel_head_feature)

        if self.training:
            if self.channel_head:
                if self.ID_LOSS_TYPE in ('arcface', 'cosface', 'amsoftmax', 'circle'):
                    cls_score = self.classifier(feat, label)
                else:
                    cls_score = self.classifier(feat)
                return cls_score, global_feat, channel_head_feature

            else:
                if self.ID_LOSS_TYPE in ('arcface', 'cosface', 'amsoftmax', 'circle'):
                    cls_score = self.classifier(feat, label)
                else:
                    cls_score = self.classifier(feat)
                return cls_score, global_feat
        else:
            if self.neck_feat == 'after':
                # print("Test with feature after BN")
                return feat
            else:
                # print("Test with feature before BN")
                return global_feat

    def load_param(self, trained_path):
        param_dict = {k.replace('module.', ""):
                v for k, v in torch.load(trained_path).items()}
        for i in param_dict:
            if 'classifier' in i or 'arcface' in i:
                continue
            self.state_dict()[i].copy_(param_dict[i])
        print('Loading pretrained model from {}'.format(trained_path))

    def load_param_finetune(self, model_path):
        param_dict = torch.load(model_path)
        for i in param_dict:
            self.state_dict()[i].copy_(param_dict[i])
        print('Loading pretrained model for finetuning from {}'.format(model_path))

def make_model(cfg, num_class):
    model = Backbone(num_class, cfg)
    return model
