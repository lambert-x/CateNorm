import random

import SimpleITK as sitk
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset


def loop_translate(a, d):
    n = np.ndarray(a.shape)
    for k in d:
        n[a == k] = d[k]
    return n


def get_datas(data_path, excluded_classes=None, n_classes=None):
    if type(data_path) != str:
        data_path = data_path.decode("utf-8")  # convert bytes to str
    path = data_path.split(",")

    image_path = path[0]
    label_path = path[1]

    itk_image = sitk.ReadImage(image_path)
    itk_mask = sitk.ReadImage(label_path)

    image = sitk.GetArrayFromImage(itk_image)
    mask = sitk.GetArrayFromImage(itk_mask)

    if excluded_classes is not None:
        mask_not_excluded = 1 - np.isin(mask, excluded_classes)
        mask = mask * mask_not_excluded
        post_mapping_dict = {}
        original_classes = list(range(n_classes + len(excluded_classes)))
        remain_classes = [item for item in original_classes if item not in excluded_classes]
        for new_value, value in enumerate(remain_classes):
            post_mapping_dict[value] = new_value
        mask = loop_translate(mask, post_mapping_dict)

    return image.transpose([1, 2, 0]), mask.transpose([1, 2, 0])




def extract_patch(image, mask, patch_size=(384, 384, 3), num_class=2, augmentation=False,
                  num_patches=1, training=True, mid_slice_idx=None, zoom_crop=False):
    """Extracts a patch of given resolution and size at a specific location."""
    image_patches = []
    mask_patches = []
    num_patches_now = 0
    while num_patches_now < num_patches:
        if training:
            z = random_patch_center_z(mask, patch_size=patch_size)  # define the centre of current patch
        else:
            z = mid_slice_idx
        image_patch = image[:, :, z - 1:z + 2]
        mask_patch = mask[:, :, z]
        image_patch = torch.from_numpy(image_patch.astype(np.float32)).permute(2, 0, 1)
        mask_patch = torch.from_numpy(mask_patch.astype(np.float32))


        # data augmentation
        if augmentation:
            if random.randint(0, 1) == 1:
                rand_degree = random.randint(-10, 10)
                image_patch = TF.rotate(image_patch.unsqueeze(0), rand_degree, resample=2).squeeze()
                mask_patch = TF.rotate(mask_patch.unsqueeze(0), rand_degree, resample=2).squeeze()

            if zoom_crop:
                if random.randint(0, 1) == 1:
                    zoom_crop_scale = random.uniform(1, 1.5)
                    image_patch = F.interpolate(image_patch.unsqueeze(0), scale_factor=zoom_crop_scale,
                                                mode='bilinear').squeeze()
                    image_patch = transforms.CenterCrop(384)(image_patch)
                    mask_patch = F.interpolate(mask_patch.unsqueeze(0).unsqueeze(0), scale_factor=zoom_crop_scale,
                                               mode='nearest').squeeze()
                    mask_patch = transforms.CenterCrop(384)(mask_patch)


            if num_class == 2:  # for multi-class dataset, flip could mislead the classification
                if random.randint(0, 1) == 1:
                    image_patch = torch.flip(image_patch, dims=[-1])
                    mask_patch = torch.flip(mask_patch, dims=[-1])


                if random.randint(0, 1) == 1:
                    image_patch = translate(image_patch)
                    mask_patch = translate(mask_patch.unsqueeze(0)).squeeze()

        image_patches.append(image_patch)
        mask_patches.append(mask_patch)
        num_patches_now += 1

    image_patches = image_patches[0]
    mask_patches = mask_patches[0]

    return image_patches, mask_patches


def random_patch_center_z(mask, patch_size):
    z = np.random.randint(patch_size[2] // 2, mask.shape[2] - patch_size[2] // 2)
    return z


def crop_center(img, new_height, new_width=None):
    height = img.shape[0]
    width = img.shape[1]
    if not new_width:
        new_width = new_height
    starty = height // 2 - (new_height // 2)
    startx = width // 2 - (new_width // 2)
    return img[starty:starty + new_height, startx:startx + new_width, ...]


def translate(img, shift_std=4):
    shift = int(np.around(np.random.randn(1)[0] * shift_std))
    if shift == 0:
        return img
    else:
        i = random.randint(1, 2)
        img = torch.roll(img, shifts=shift, dims=i)
        return img


class SiteSet(Dataset):
    def __init__(self, datalist, iters=2400, training=True, augmentation=False, source='Overall', zoom_crop=False,
                 whitening=True, batchsize=4, site_num=3, n_classes=2, excluded_classes=None
                 ):
        images = []
        masks = []

        case_idx = 0
        self.test_idxlist = []
        self.site = source
        self.training = training
        self.whitening = whitening
        self.n_classes = n_classes
        if excluded_classes is not None and training:
            post_mapping_dict = {}
            original_classes = list(range(n_classes + len(excluded_classes)))
            remain_classes = [item for item in original_classes if item not in excluded_classes]
            for new_value, value in enumerate(remain_classes):
                post_mapping_dict[value] = new_value
            print('Post mapping dict:', post_mapping_dict)
        if self.site == 'Overall':
            for sidx, sitelist in enumerate(datalist):
                site_images = []
                site_masks = []
                np.random.shuffle(sitelist)
                for case in sitelist:
                    if len(case) > 0:
                        case_data = get_datas(case, excluded_classes=excluded_classes, n_classes=n_classes)
                        image = case_data[0]
                        mask = case_data[1]
                        if not training:
                            for mid_slice in range(1, image.shape[2] - 1):
                                self.test_idxlist.append([case_idx, mid_slice])
                            case_idx += 1
                        site_images.append(image)
                        site_masks.append(mask)
                images.append(site_images)
                masks.append(site_masks)
        else:
            np.random.shuffle(datalist)
            for case in datalist:
                if len(case) > 0:
                    case_data = get_datas(case, excluded_classes=excluded_classes, n_classes=n_classes)
                    image = case_data[0]
                    mask = case_data[1]
                    if not training:
                        for mid_slice in range(1, image.shape[2] - 1):
                            self.test_idxlist.append([case_idx, mid_slice])
                        case_idx += 1
                    images.append(image)
                    masks.append(mask)
        if whitening:
            if self.site == 'Overall':
                for site_idx, site_images in enumerate(images):
                    for image_idx, image in enumerate(site_images):
                        mean = image.mean()
                        std = image.std()
                        images[site_idx][image_idx] = (image - mean) / std
            else:
                for image_idx, image in enumerate(images):
                    mean = image.mean()
                    std = image.std()
                    images[image_idx] = (image - mean) / std
        self.images = images
        self.masks = masks

        self.iteration = iters
        self.batchsize = batchsize
        self.site_num = site_num
        self.augmentation = augmentation
        self.zoom_crop = zoom_crop
        if self.training:
            if self.site == 'Overall':
                set_length = self.iteration // site_num
                index_list = {}
                for i in range(site_num):
                    site_length = len(self.images[i])
                    index_origin = np.arange(len(self.images[i]))
                    index_list[i] = index_origin.copy()
                    for turn in range(set_length // site_length):
                        np.random.shuffle(index_origin)
                        index_list[i] = np.hstack((index_list[i], index_origin.copy()))
                self.index_list = index_list
            else:
                set_length = self.iteration
                site_length = len(self.images)
                index_origin = np.arange(len(self.images))
                index_list = index_origin.copy()
                for i in range(set_length // site_length):
                    np.random.shuffle(index_origin)
                    index_list = np.hstack((index_list, index_origin.copy()))
                self.index_list = index_list

    def __getitem__(self, idx):
        if self.training:
            # multiple sites
            if self.site == 'Overall':
                site_idx = (idx % (self.site_num * self.batchsize)) // self.batchsize
                idx = ((idx // (self.site_num * self.batchsize)) * self.batchsize + (
                        (idx % (self.site_num * self.batchsize)) - site_idx * self.batchsize))
                idx = self.index_list[site_idx][idx]
                image, mask = extract_patch(self.images[site_idx][idx], self.masks[site_idx][idx],
                                            augmentation=self.augmentation, num_class=self.n_classes,
                                            zoom_crop=self.zoom_crop)
            else:
                # only one site
                idx = self.index_list[idx]
                site_idx_dict = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5}
                site_idx = site_idx_dict[self.site]
                image, mask = extract_patch(self.images[idx], self.masks[idx],
                                            augmentation=self.augmentation, num_class=self.n_classes,
                                            zoom_crop=self.zoom_crop)
            return image, mask, site_idx
        else:
            case_idx = self.test_idxlist[idx]
            image, mask = extract_patch(self.images[case_idx[0]], self.masks[case_idx[0]],
                                        augmentation=False, num_class=self.n_classes, training=False,
                                        mid_slice_idx=case_idx[1])
            return image, mask

    def __len__(self):
        if self.training:
            length = self.iteration
        else:
            length = len(self.test_idxlist)
        return length
