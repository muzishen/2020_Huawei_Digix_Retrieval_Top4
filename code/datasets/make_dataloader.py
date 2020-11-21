import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader
import numpy as np
from .veri import VeRi
from .digital import Digital
from .bases import ImageDataset,AugMixDataset
from .preprocessing import RandomErasing, ImageNetPolicy,RandomPatch, Cutout
from .sampler import RandomIdentitySampler

__factory = {
    'veri': VeRi,
    'digital': Digital

}

def train_collate_fn(batch):
    """
    # collate_fn这个函数的输入就是一个list，list的长度是一个batch size，list中的每个元素都是__getitem__得到的结果
    """
    imgs, pids, _, _ = zip(*batch)
    pids = torch.tensor(pids, dtype=torch.int64)
    return torch.stack(imgs, dim=0), pids

def val_collate_fn(batch):
    imgs, pids, camids, _ = zip(*batch)
    return torch.stack(imgs, dim=0), pids, camids#, img_paths


def make_dataloader(cfg):
    if cfg.DATASETS.HARD_AUG:
        train_transforms = T.Compose([
            #ImageNetPolicy(),
            T.RandomRotation(30),
            T.Resize(cfg.INPUT.SIZE_TRAIN, interpolation=3),
            T.RandomVerticalFlip(p=cfg.INPUT.PROB),
            T.RandomHorizontalFlip(p=cfg.INPUT.PROB),           
            T.Pad(cfg.INPUT.PADDING),
            T.RandomCrop(cfg.INPUT.SIZE_TRAIN),
            #T.RandomResizedCrop(size=cfg.INPUT.SIZE_TRAIN, scale=(0.75,1.0), ratio=(0.75,1.3333), interpolation=3),
            T.RandomApply([T.ColorJitter(cfg.INPUT.CJ.BRIGHTNESS,cfg.INPUT.CJ.CONTRAST, cfg.INPUT.CJ.SATURATION, cfg.INPUT.CJ.HUE)], p=cfg.INPUT.CJ.PROB),
            #RandomPatch(prob_happen=cfg.INPUT.PROB,patch_max_area=0.1677),
            T.ToTensor(),
            T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD),           
            RandomErasing(probability=cfg.INPUT.RE_PROB, mean=cfg.INPUT.PIXEL_MEAN),
            #Cutout(n_holes=1, length=cfg.INPUT.LENGTH)
        ])
    elif cfg.INPUT.AUGMIX:
            train_transforms = T.Compose([
            #ImageNetPolicy(),
            T.RandomRotation(30),
            T.Resize(cfg.INPUT.SIZE_TRAIN),
            T.RandomVerticalFlip(p=cfg.INPUT.PROB),
            T.RandomHorizontalFlip(p=cfg.INPUT.PROB),           
            T.Pad(cfg.INPUT.PADDING),
            T.RandomCrop(cfg.INPUT.SIZE_TRAIN),
            #T.RandomResizedCrop(size=cfg.INPUT.SIZE_TRAIN, scale=(0.75,1.0), ratio=(0.75,1.3333), interpolation=3),
            T.RandomApply([T.ColorJitter(cfg.INPUT.CJ.BRIGHTNESS,cfg.INPUT.CJ.CONTRAST, cfg.INPUT.CJ.SATURATION, cfg.INPUT.CJ.HUE)], p=cfg.INPUT.CJ.PROB),
            RandomPatch(prob_happen=0.5,patch_max_area=0.1677),])
            # T.ToTensor(),
            # T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD),           
            # RandomErasing(probability=cfg.INPUT.RE_PROB, mean=cfg.INPUT.PIXEL_MEAN),
            
    else:
        train_transforms = T.Compose([
            T.Resize((580, 580), interpolation=3),
            #T.RandomHorizontalFlip(p=cfg.INPUT.PROB),
            T.Pad(cfg.INPUT.PADDING),
            #T.RandomCrop(cfg.INPUT.SIZE_TRAIN),
            T.RandomHorizontalFlip(p=cfg.INPUT.PROB),
            T.RandomResizedCrop(size=580, scale=(0.75,1.5), ratio=(0.75,1.3333), interpolation=3),
            T.RandomApply([T.ColorJitter(cfg.INPUT.CJ.BRIGHTNESS,cfg.INPUT.CJ.CONTRAST, cfg.INPUT.CJ.SATURATION, cfg.INPUT.CJ.HUE)], p=cfg.INPUT.CJ.PROB),
            RandomPatch(prob_happen=0.5,patch_max_area=0.1677),
            T.ToTensor(),
            T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD),
            
            #RandomErasing(probability=cfg.INPUT.RE_PROB, mean=cfg.INPUT.PIXEL_MEAN)
        ])
    val_transforms = T.Compose([
        T.Resize(cfg.INPUT.SIZE_TEST, interpolation=3),
        T.ToTensor(),
        T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD)
    ])

    num_workers = cfg.DATALOADER.NUM_WORKERS

    dataset = __factory[cfg.DATASETS.NAMES](root=cfg.DATASETS.ROOT_DIR)
    num_classes = dataset.num_train_pids

    preprocess = T.Compose([T.ToTensor(),T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD)])

    train_set = ImageDataset(dataset.train, train_transforms)

    if cfg.INPUT.AUGMIX:
        k = 3
        alpha = 1.
        js_loss = True
        train_set = AugMixDataset(train_set, preprocess, k, alpha, False)

    if 'triplet' in cfg.DATALOADER.SAMPLER:
        train_loader = DataLoader(
            train_set, batch_size=cfg.SOLVER.IMS_PER_BATCH,
            sampler=RandomIdentitySampler(dataset.train, cfg.SOLVER.IMS_PER_BATCH, cfg.DATALOADER.NUM_INSTANCE),
            num_workers=num_workers, collate_fn=train_collate_fn, pin_memory =True
        )
    elif cfg.DATALOADER.SAMPLER == 'softmax':
        print('using softmax sampler')
        train_loader = DataLoader(
            train_set, batch_size=cfg.SOLVER.IMS_PER_BATCH, shuffle=True, num_workers=num_workers,
            collate_fn=train_collate_fn, pin_memory =True
        )
    else:
        print('unsupported sampler! expected softmax or triplet but got {}'.format(cfg.SAMPLER))

    val_set = ImageDataset(dataset.query + dataset.gallery, val_transforms)
    query_name = np.asarray(dataset.query)[:, 0]
    gallery_name = np.asarray(dataset.gallery)[:, 0]
    val_loader = DataLoader(
        val_set, batch_size=cfg.TEST.IMS_PER_BATCH, shuffle=False, num_workers=num_workers,
        collate_fn=val_collate_fn, pin_memory =True
    )
    return train_loader, val_loader, len(dataset.query), num_classes, query_name, gallery_name


def make_dataloader_Pseudo(cfg):
    if cfg.DATASETS.HARD_AUG:
        train_transforms = T.Compose([
            T.Resize(cfg.INPUT.SIZE_TRAIN),
            T.RandomHorizontalFlip(p=cfg.INPUT.PROB),
            T.Pad(cfg.INPUT.PADDING),
            T.RandomCrop(cfg.INPUT.SIZE_TRAIN),
            T.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.2),
            T.transforms.RandomAffine(0, translate=None, scale=[0.9, 1.1], shear=None, resample=False, fillcolor=128),
            T.ToTensor(),
            T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD),
            RandomErasing(probability=cfg.INPUT.RE_PROB, mean=cfg.INPUT.PIXEL_MEAN)
        ])
        
    else:
        train_transforms = T.Compose([
            T.Resize(cfg.INPUT.SIZE_TRAIN),
            T.RandomHorizontalFlip(p=cfg.INPUT.PROB),
            T.Pad(cfg.INPUT.PADDING),
            T.RandomCrop(cfg.INPUT.SIZE_TRAIN),
            T.ToTensor(),
            T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD),
            RandomErasing(probability=cfg.INPUT.RE_PROB, mean=cfg.INPUT.PIXEL_MEAN)
        ])
    val_transforms = T.Compose([
        T.Resize(cfg.INPUT.SIZE_TEST),
        T.ToTensor(),
        T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD)
    ])

    num_workers = cfg.DATALOADER.NUM_WORKERS

    dataset = __factory[cfg.DATASETS.NAMES](root=cfg.DATASETS.ROOT_DIR)
    num_classes = dataset.num_train_pids

    train_set = ImageDataset(dataset.train, train_transforms)

    if 'triplet' in cfg.DATALOADER.SAMPLER:
        train_loader = DataLoader(
            train_set, batch_size=cfg.SOLVER.IMS_PER_BATCH,
            sampler=RandomIdentitySampler(dataset.train, cfg.SOLVER.IMS_PER_BATCH, cfg.DATALOADER.NUM_INSTANCE),
            num_workers=num_workers, collate_fn=train_collate_fn
        )
    elif cfg.DATALOADER.SAMPLER == 'softmax':
        print('using softmax sampler')
        train_loader = DataLoader(
            train_set, batch_size=cfg.SOLVER.IMS_PER_BATCH, shuffle=True, num_workers=num_workers,
            collate_fn=train_collate_fn
        )
    else:
        print('unsupported sampler! expected softmax or triplet but got {}'.format(cfg.SAMPLER))


    val_set_green = ImageDataset(dataset.query_green + dataset.gallery_green, val_transforms)
    val_loader_green = DataLoader(
        val_set_green, batch_size=cfg.TEST.IMS_PER_BATCH, shuffle=False, num_workers=num_workers,
        collate_fn=val_collate_fn
    )

    return train_loader, val_loader_green, len(dataset.query_green), num_classes, dataset, train_set, train_transforms
