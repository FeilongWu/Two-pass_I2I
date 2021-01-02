import os
from data.base_dataset import BaseDataset
from data.image_folder import make_dataset
from PIL import Image
import random
import numpy as np
import torch.utils.data as data
import torchvision.transforms as transforms



def get_params(opt, size):
    w, h = size
    new_h = h
    new_w = w
    if opt.preprocess == 'resize_and_crop':
        new_h = new_w = opt.load_size
    elif opt.preprocess == 'scale_width_and_crop':
        new_w = opt.load_size
        new_h = opt.load_size * h // w

    #x = random.randint(0, np.maximum(0, new_w - opt.crop_size))
    #y = random.randint(0, np.maximum(0, new_h - opt.crop_size))
    x=y=int((new_h - opt.crop_size)/2)
    flip = random.random() > 0.5
    #flip = False

    return {'crop_pos': (x, y), 'flip': flip}


def get_transform(opt, params=None, grayscale=False, method=Image.BICUBIC, convert=True):
    transform_list = []
    flip = False
    if grayscale:
        transform_list.append(transforms.Grayscale(1))
    if 'resize' in opt.preprocess:
        osize = [opt.load_size, opt.load_size]
        transform_list.append(transforms.Resize(osize, method))
    elif 'scale_width' in opt.preprocess:
        transform_list.append(transforms.Lambda(lambda img: __scale_width(img, opt.load_size, opt.crop_size, method)))

    if 'crop' in opt.preprocess:
        if params is None:
            transform_list.append(transforms.RandomCrop(opt.crop_size))
        else:
            transform_list.append(transforms.Lambda(lambda img: __crop(img, params['crop_pos'], opt.crop_size)))

    if opt.preprocess == 'none':
        transform_list.append(transforms.Lambda(lambda img: __make_power_2(img, base=4, method=method)))

    if not opt.no_flip:
        if params is None:
            transform_list.append(transforms.RandomHorizontalFlip())
        elif params['flip']:
            flip = True
            transform_list.append(transforms.Lambda(lambda img: __flip(img, params['flip'])))

    if convert:
        transform_list += [transforms.ToTensor()]
        if grayscale:
            transform_list += [transforms.Normalize((0.5,), (0.5,))]
        else:
            transform_list += [transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    return (transforms.Compose(transform_list), flip)


def __make_power_2(img, base, method=Image.BICUBIC):
    ow, oh = img.size
    h = int(round(oh / base) * base)
    w = int(round(ow / base) * base)
    if h == oh and w == ow:
        return img

    __print_size_warning(ow, oh, w, h)
    return img.resize((w, h), method)


def __scale_width(img, target_size, crop_size, method=Image.BICUBIC):
    ow, oh = img.size
    if ow == target_size and oh >= crop_size:
        return img
    w = target_size
    h = int(max(target_size * oh / ow, crop_size))
    return img.resize((w, h), method)


def __crop(img, pos, size):
    ow, oh = img.size
    x1, y1 = pos
    tw = th = size
    if (ow > tw or oh > th):
        return img.crop((x1, y1, x1 + tw, y1 + th))
    return img


def __flip(img, flip):
    if flip:
        return img.transpose(Image.FLIP_LEFT_RIGHT)
    return img


def __print_size_warning(ow, oh, w, h):
    """Print warning information about image size(only print once)"""
    if not hasattr(__print_size_warning, 'has_printed'):
        print("The image size needs to be a multiple of 4. "
              "The loaded image size was (%d, %d), so it was adjusted to "
              "(%d, %d). This adjustment will be done to all images "
              "whose sizes are not multiples of 4" % (ow, oh, w, h))
        __print_size_warning.has_printed = True


class AlignedPseudo1Dataset(BaseDataset):
    """A dataset class for paired image dataset.

    It assumes that the directory '/path/to/data/train' contains image pairs in the form of {A,B}.
    During test time, you need to prepare a directory '/path/to/data/test'.
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        self.dir_AB = os.path.join(opt.dataroot, opt.phase)  # get the image directory
        self.AB_paths = sorted(make_dataset(self.dir_AB, opt.max_dataset_size))  # get image paths
        assert(self.opt.load_size >= self.opt.crop_size)   # crop_size should be smaller than the size of loaded image
        self.input_nc = self.opt.output_nc if self.opt.direction == 'BtoA' else self.opt.input_nc
        self.output_nc = self.opt.input_nc if self.opt.direction == 'BtoA' else self.opt.output_nc

        if opt.direction == "AtoB":
            self.dir_real = os.path.join(opt.dataroot, opt.phase + 'B')  # create a path '/path/to/data/trainB'
        elif opt.direction == 'BtoA':
            self.dir_real = os.path.join(opt.dataroot, opt.phase + 'A')  # create a path '/path/to/data/trainA'
        else:
            assert 'Invalid direction: ' + opt.direction

        self.real_paths = sorted(make_dataset(self.dir_real, opt.max_dataset_size))   # load images from '/path/to/data/trainA(B)'
        self.real_size = len(self.real_paths)  # get the size of dataset A(B)
        self.transform_real = get_transform(self.opt, grayscale=(self.output_nc == 1))[0]
            

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor) - - an image in the input domain
            B (tensor) - - its corresponding image in the target domain
            A_paths (str) - - image paths
            B_paths (str) - - image paths (same as A_paths)
        """
        # read a image given a random integer index
        AB_path = self.AB_paths[index]
        AB = Image.open(AB_path).convert('RGB')
        # split AB image into A and B
        w, h = AB.size
        w2 = int(w / 2)
        A = AB.crop((0, 0, w2, h))
        B = AB.crop((w2, 0, w, h))

        # apply the same transform to both A and B
        transform_params = get_params(self.opt, A.size)
        #transform_params = {'crop_pos': (0, 0), 'flip': False}
        A_transform = get_transform(self.opt, transform_params, grayscale=(self.input_nc == 1))[0]
        temp = get_transform(self.opt, transform_params, grayscale=(self.output_nc == 1))
        B_transform = temp[0]
        flip = temp[1]

        A = A_transform(A)
        B = B_transform(B)

        if self.opt.serial_batches:   # make sure index is within then range
            index_real = index % self.real_size
        else:   # randomize the index for domain A(B) to avoid fixed pairs.
            index_real = random.randint(0, self.real_size - 1)
        real_path = self.real_paths[index_real]
        real_img = Image.open(real_path).convert('RGB')
        #real = self.transform_real(real_img)
        real = B_transform(real_img)

        return {'A': A, 'B': B, 'ground_truth': real, 'A_paths': AB_path, \
                'B_paths': AB_path, 'real_path': real_path, 'flip': flip}

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.AB_paths)
