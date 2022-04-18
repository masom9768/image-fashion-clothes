"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

###############################################################################
# Code from
# https://github.com/pytorch/vision/blob/master/torchvision/datasets/folder.py
# Modified the original code so that it also loads images from the current
# directory as well as the subdirectories
###############################################################################
import torch.utils.data as data
from PIL import Image
import os

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP', '.tiff', '.webp'
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def make_dataset_rec(dir, images):
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    for root, dnames, fnames in sorted(os.walk(dir, followlinks=True)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)


def make_dataset(dir, recursive=False, read_cache=False, write_cache=False):
    images = []

    if read_cache:
        possible_filelist = os.path.join(dir, 'files.list')
        if os.path.isfile(possible_filelist):
            with open(possible_filelist, 'r') as f:
                images = f.read().splitlines()
                return images

    if recursive:
        make_dataset_rec(dir, images)
    else:
        assert os.path.isdir(dir) or os.path.islink(dir), '%s is not a valid directory' % dir

        for root, dnames, fnames in sorted(os.walk(dir)):
            for fname in fnames:
                if is_image_file(fname):
                    path = os.path.join(root, fname)
                    images.append(path)

    if write_cache:
        filelist_cache = os.path.join(dir, 'files.list')
        with open(filelist_cache, 'w') as f:
            for path in images:
                f.write("%s\n" % path)
            print('wrote filelist cache at %s' % filelist_cache)

    return images
    """def make_dataset(dir):
	images = []
	assert os.path.isdir(dir), '%s is not a valid directory' % dir
	new_root = './fashion_data'
	if not os.path.exists(new_root):
		os.mkdir(new_root)

	train_root = './fashion_data/train'
	if not os.path.exists(train_root):
		os.mkdir(train_root)

	test_root = './fashion_data/test'
	if not os.path.exists(test_root):
		os.mkdir(test_root)

	train_images = []
	train_f = open('./fashion_data/train.lst', 'r')
	for lines in train_f:
		lines = lines.strip()
		if lines.endswith('.jpg'):
			train_images.append(lines)

	test_images = []
	test_f = open('./fashion_data/test.lst', 'r')
	for lines in test_f:
		lines = lines.strip()
		if lines.endswith('.jpg'):
			test_images.append(lines)

	print(train_images, test_images)

	# print(train_images, test_images)

	for root, _, fnames in sorted(os.walk(dir)):
		for fname in fnames:
			if is_image_file(fname):
				path = os.path.join(root, fname)
				path1 = path.replace('fashion_data/img', 'fashion')
				path_names = path1.split('/') 
				# path_names[2] = path_names[2].replace('_', '')
				path_names[3] = path_names[3].replace('_', '')
				path_names[4] = path_names[4].split('_')[0] + "_" + "".join(path_names[4].split('_')[1:])
				path_names = "".join(path_names)
				print(path_names)
				new_path = path_names
				img = Image.open(path)
				#imgcrop = img.crop((40, 0, 216, 256))
				if new_path in train_images:
					img.save(os.path.join(train_root, path_names))
				elif new_path in test_images:
					img.save(os.path.join(test_root, path_names))
# 					
make_dataset('fashion_data/img')"""



def default_loader(path):
    return Image.open(path).convert('RGB')


class ImageFolder(data.Dataset):

    def __init__(self, root, transform=None, return_paths=False,
                 loader=default_loader):
        imgs = make_dataset(root)
        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in: " + root + "\n"
                               "Supported image extensions are: " +
                               ",".join(IMG_EXTENSIONS)))

        self.root = root
        self.imgs = imgs
        self.transform = transform
        self.return_paths = return_paths
        self.loader = loader

    def __getitem__(self, index):
        path = self.imgs[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.return_paths:
            return img, path
        else:
            return img

    def __len__(self):
        return len(self.imgs)
