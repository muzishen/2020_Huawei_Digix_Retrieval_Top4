from PIL import Image, ImageFile

from torch.utils.data import Dataset
import os.path as osp
import torch 
import random
import torch.nn.functional as F
from torch.distributions.dirichlet import Dirichlet
from torch.distributions.beta import Beta
from torchvision import transforms

from .augmentations import augmentations
#from augmentations import augmentations
ImageFile.LOAD_TRUNCATED_IMAGES = True


def read_image(img_path):
    """Keep reading image until succeed.
    This can avoid IOError incurred by heavy IO process."""
    got_img = False
    if not osp.exists(img_path):
        raise IOError("{} does not exist".format(img_path))
    while not got_img:
        try:
            img = Image.open(img_path).convert('RGB')
            got_img = True
        except IOError:
            print("IOError incurred when reading '{}'. Will redo. Don't worry. Just chill.".format(img_path))
            pass
    return img


class BaseDataset(object):
    """
    Base class of reid dataset
    """

    def get_imagedata_info(self, data):
        pids, cams = [], []
        for _, pid, camid in data:
            pids += [pid]
            cams += [camid]
        pids = set(pids)
        cams = set(cams)
        num_pids = len(pids)
        num_cams = len(cams)
        num_imgs = len(data)
        return num_pids, num_imgs, num_cams

    def print_dataset_statistics(self):
        raise NotImplementedError


class BaseImageDataset(BaseDataset):
    """
    Base class of image reid dataset
    """

    def print_dataset_statistics(self, train, query, gallery):
        num_train_pids, num_train_imgs, num_train_cams = self.get_imagedata_info(train)
        num_query_pids, num_query_imgs, num_query_cams = self.get_imagedata_info(query)
        num_gallery_pids, num_gallery_imgs, num_gallery_cams = self.get_imagedata_info(gallery)

        print("Dataset statistics:")
        print("  ----------------------------------------")
        print("  subset   | # ids | # images | # cameras")
        print("  ----------------------------------------")
        print("  train    | {:5d} | {:8d} | {:9d}".format(num_train_pids, num_train_imgs, num_train_cams))
        print("  query    | {:5d} | {:8d} | {:9d}".format(num_query_pids, num_query_imgs, num_query_cams))
        print("  gallery  | {:5d} | {:8d} | {:9d}".format(num_gallery_pids, num_gallery_imgs, num_gallery_cams))
        print("  ----------------------------------------")


class ImageDataset(Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img_path, pid, camid = self.dataset[index]
        img = read_image(img_path)

        if self.transform is not None:
            img = self.transform(img)

        return img, pid, camid, img_path.split('/')[-1]


class AugMixDataset(torch.utils.data.Dataset):
    """Dataset wrapper to perform AugMix augmentation.
       referenced from https://github.com/google-research/augmix/blob/master/cifar.py
    """
    def __init__(self, dataset, preprocess, k, alpha, no_jsd=False):
        self.dataset = dataset
        self.preprocess = preprocess
        self.k = k
        self.alpha = alpha
        self.no_jsd = no_jsd

    def __getitem__(self, i):
        x,  y , camid , img_path = self.dataset[i]
        if self.no_jsd:
            return augmentAndMix(x, self.k, self.alpha, self.preprocess), y, camid, img_path
        else:
            return (self.preprocess(x), 
                    augmentAndMix(x, self.k, self.alpha, self.preprocess),
                    augmentAndMix(x, self.k, self.alpha, self.preprocess)), y, camid , img_path

    def __len__(self):
        return len(self.dataset)

def augmentAndMix(x_orig, k, alpha, preprocess):
    # k : number of chains
    # alpha : sampling constant

    x_temp = x_orig # back up for skip connection

    x_aug = torch.zeros_like(preprocess(x_orig))
    mixing_weight_dist = Dirichlet(torch.empty(k).fill_(alpha))
    mixing_weights = mixing_weight_dist.sample()

    for i in range(k):
        sampled_augs = random.sample(augmentations, k)
        aug_chain_length = random.choice(range(1,k+1))
        aug_chain = sampled_augs[:aug_chain_length]

        for aug in aug_chain:
            severity = random.choice(range(1,6))
            x_temp = aug(x_temp, severity)

        x_aug += mixing_weights[i] * preprocess(x_temp)

    skip_conn_weight_dist = Beta(torch.tensor([alpha]), torch.tensor([alpha]))
    skip_conn_weight = skip_conn_weight_dist.sample()

    x_augmix = skip_conn_weight * x_aug + (1 - skip_conn_weight) * preprocess(x_orig)

    return x_augmix