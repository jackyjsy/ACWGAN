import torch
import os
import random
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from PIL import Image
import numpy as np
import h5py
import scipy.io as sio

def to_categorical(y, num_classes):
    """ 1-hot encodes a tensor """
    y = np.asarray(y)
    y = y.squeeze()
    # print(type(y))
    y = np.eye(num_classes, dtype='uint8')[y]
    return y

class CelebDataset(Dataset):
    def __init__(self, image_path, seg_path, metadata_path, transform, transform_seg1, transform_seg2, mode):
        self.image_path = image_path
        self.seg_path = seg_path
        self.transform = transform
        self.transform_seg1 = transform_seg1
        self.transform_seg2 = transform_seg2
        self.mode = mode
        self.lines = open(metadata_path, 'r').readlines()
        self.num_data = int(self.lines[0])
        self.attr2idx = {}
        self.idx2attr = {}

        print ('Start preprocessing dataset..!')
        self.preprocess()
        print ('Finished preprocessing dataset..!')

        if self.mode == 'train':
            self.num_data = len(self.train_filenames)
        elif self.mode == 'test':
            self.num_data = len(self.test_filenames)

    def preprocess(self):
        attrs = self.lines[1].split()
        for i, attr in enumerate(attrs):
            self.attr2idx[attr] = i
            self.idx2attr[i] = attr

        self.selected_attrs = ['Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Male', 'Young']
        self.train_filenames = []
        self.train_labels = []
        self.test_filenames = []
        self.test_labels = []

        lines = self.lines[2:]
        random.shuffle(lines)   # random shuffling
        for i, line in enumerate(lines):

            splits = line.split()
            filename = splits[0]
            values = splits[1:]

            label = []
            for idx, value in enumerate(values):
                attr = self.idx2attr[idx]

                if attr in self.selected_attrs:
                    if value == '1':
                        label.append(1)
                    else:
                        label.append(0)

            if (i+1) < 2000:
                self.test_filenames.append(filename)
                self.test_labels.append(label)
            else:
                self.train_filenames.append(filename)
                self.train_labels.append(label)

    def __getitem__(self, index):
        image = Image.open(os.path.join(self.image_path, self.train_filenames[index]))
        label = self.train_labels[index]

        return self.transform(image), torch.FloatTensor(label)

    def __len__(self):
        return self.num_data


class FashionDataset(Dataset):
    def __init__(self, image_path, metadata_path, transform, transform_seg, mode):
        self.image_path = image_path
        self.transform = transform
        self.transform_seg = transform_seg
        self.mode = mode
        print ('Start preprocessing dataset..!')
        self.f = h5py.File(os.path.join(image_path, 'G2.h5'), 'r')
        self.seg_data = self.f['b_']
        self.image_data = self.f['ih']
        self.image_mean = self.f['ih_mean']
        self.cond_data = sio.loadmat(os.path.join(metadata_path, 'encode_hn2_rnn_100_2_full.mat'))['hn2']
        self.label_data = sio.loadmat(os.path.join(metadata_path, 'language_original.mat'))['color_']
        self.num_data = len(self.image_data)
        self.preprocess()
        print ('Finished preprocessing dataset..!')
    def preprocess(self):
        self.label_data=torch.LongTensor(self.label_data.astype(int))
        self.label_data=self.label_data.squeeze()-1
        c_dim=17
        out = torch.zeros(self.num_data, c_dim)
        out[np.arange(self.num_data), self.label_data] = 1
        self.label_data_onehot = out
    def __getitem__(self, index):

        image = self.image_data[index] + self.image_mean
        image = image - np.amin(image)
        image = image / np.amax(image)
        image = (image - 0.5) * 2
        # label = self.cond_data[index]
        label_onehot = self.label_data_onehot[index]
        label = self.label_data[index]
        num_s = 7

        image = torch.FloatTensor(image)
        label_onehot = torch.FloatTensor(label_onehot)
        # print(label)
        # print(label_onehot)
        return image, label, label_onehot

    def __len__(self):
        return self.num_data


def get_loader(image_path, seg_path, metadata_path, crop_size, image_size, batch_size, dataset='CelebA', mode='train'):
    """Build and return data loader."""

    if dataset == 'CelebA':
        transform = transforms.Compose([
            transforms.CenterCrop(crop_size),
            transforms.Scale(image_size),
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        transform_seg1 = transforms.Compose([
            transforms.CenterCrop(crop_size),
            transforms.Scale(image_size)])
        transform_seg2 = transforms.Compose([
            transforms.ToTensor()])
    elif dataset == 'Fashion':
        transform = transforms.Compose([
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        transform_seg = transforms.Compose([
            transforms.ToTensor(),
            ])

    if dataset == 'CelebA':
        dataset = CelebDataset(image_path, seg_path, metadata_path, transform, transform_seg1, transform_seg2, mode)
    elif dataset == 'Fashion':
        dataset = FashionDataset(image_path, metadata_path, transform, transform_seg, mode)

    shuffle = False
    if mode == 'train':
        shuffle = True

    data_loader = DataLoader(dataset=dataset,
                             batch_size=batch_size,
                             shuffle=shuffle)
    return data_loader