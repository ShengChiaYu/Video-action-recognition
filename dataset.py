import os
import sys
import glob

import cv2
import numpy as np
# np.random.seed(1216)

import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

from PIL import Image

class TrimmedVideos(Dataset):
    sample_fps = 1
    image_size = 224
    frames_num = 10
    def __init__(self, root, transform=None, train=True, problem=1, test=False):
        """ Intialize the Face dataset """
        self.root = root
        self.transform = transform
        self.labels = []
        self.videos = {}
        self.train = train
        self.problem = problem
        self.test = test

        # read label file: gt_train.csv or gt_valid.csv
        if self.test:
            f = open(root[1], 'r')
        else:
            if self.train:
                f = open(os.path.join(root, 'label', 'gt_train.csv'), 'r')
            else:
                f = open(os.path.join(root, 'label', 'gt_valid.csv'), 'r')

        lines = f.readlines()
        col_labels = (lines[0].split())[0].split(',')
        for i in range(1,len(lines)):
            dict = {}
            for j in range(len(col_labels)):
                dict[col_labels[j]] = (lines[i].split())[0].split(',')[j]
            self.labels.append(dict)

        # read directory of videos
        if self.test:
            video_folders = sorted(glob.glob(os.path.join(root[0], '**/')))
        else:
            if self.train:
                video_folders = sorted(glob.glob(os.path.join(root, 'video', 'train', '**/')))
            else:
                video_folders = sorted(glob.glob(os.path.join(root, 'video', 'valid', '**/')))

        for i in range(len(video_folders)):
            folder_name = video_folders[i].split('/')[-2]
            self.videos[folder_name] = {}

            if self.test:
                videos = sorted(glob.glob(os.path.join(root[0], '{}/*.mp4'.format(folder_name))))
            else:
                if self.train:
                    videos = sorted(glob.glob(os.path.join(root, 'video', 'train', '{}/*.mp4'.format(folder_name))))
                else:
                    videos = sorted(glob.glob(os.path.join(root, 'video', 'valid', '{}/*.mp4'.format(folder_name))))

            for j in range(len(videos)):
                video_dir = videos[j].split('/')[-1]
                video_name = video_dir.split('F')[0][:-1]
                self.videos[folder_name][video_name] = video_dir

        # number of samples
        self.len = len(self.labels)
        print('Number of samples:',self.len)


    def __getitem__(self, index):
        """ Get a sample from the dataset """
        # read label
        label = self.labels[index]
        if self.test:
            video_dir = self.videos[label['Video_category']][label['Video_name']]
            video_dir = os.path.join(self.root[0], label['Video_category'], video_dir)
        else:
            if self.train:
                video_dir = self.videos[ label['Video_category'] ][ label['Video_name'] ]
                video_dir = os.path.join(self.root, 'video', 'train', label['Video_category'], video_dir)
            else:
                video_dir = self.videos[label['Video_category']][label['Video_name']]
                video_dir = os.path.join(self.root, 'video', 'valid', label['Video_category'], video_dir)

        # read video
        video = cv2.VideoCapture(video_dir)
        length = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        get_frame = int(length / self.frames_num)
        # fps = video.get(cv2.CAP_PROP_FPS)
        # get_frame = int(fps / self.sample_fps)

        # exit if video not opened.
        if not video.isOpened():
            print("Failed to open video")
            sys.exit()

        count = get_frame
        frames = torch.Tensor()
        while video.isOpened():
            ret, frame = video.read()
            if self.problem == 1:
                criterion = ret
            if self.problem == 2:
                criterion = (ret and int(frames.size(0) / 3) < self.frames_num)
            if criterion:
                if count == get_frame:
                    frame = cv2.resize(frame, (self.image_size,self.image_size))
                    if self.transform is not None:
                        frame = self.transform(frame)
                    # print(frame)
                    frames = torch.cat((frames, frame), dim=0)
                    count = 1
                else:
                    count += 1
            else:
                break
        frame_num = int(frames.size(0) / 3)
        image_size = frames[0].size()

        if self.test:
            return frames.view(frame_num, 3, image_size[0], image_size[1]), 0
        else:
            return frames.view(frame_num, 3, image_size[0], image_size[1]), int(label['Action_labels'])


    def __len__(self):
        """ Total number of samples in the dataset """
        return self.len


def collate_fn(batch):
    frames = torch.Tensor()
    labels = []
    lens = []
    for i in range(len(batch)):
        frame, label = batch[i]
        frames = torch.cat((frames, frame), dim=0)
        if isinstance(label, list):
            labels += label
        else:
            labels.append(label)
        lens.append(frame.size(0))

    labels = torch.Tensor(labels).long()
    lens = torch.Tensor(lens).long()

    return frames, labels, lens


def TrimmedVideos_test():
    dataroot = os.path.join(os.getcwd(), 'hw4_data', 'TrimmedVideos')
    transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ])
    dataset = TrimmedVideos(root=dataroot, transform=transform, train=False, problem=2)
    # frames, label = dataset.__getitem__(3)
    # print(frames[0].size())
    # print(frames.size())
    # print(frames[3])
    # print(label)

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=6,
                                             shuffle=False, num_workers=1,
                                             collate_fn=collate_fn)
    dataiter = iter(dataloader)
    frames, labels, lens = dataiter.next()

    print('Frame tensor in each batch:', frames.shape, frames.dtype)
    print('Label tensor in each batch:', labels.shape, labels.dtype)
    print('Len tensor in each batch:', lens.shape, lens.dtype)
    print(labels)
    print(lens)


class FullLengthVideos(Dataset):
    image_size = 224
    sample_size = 500
    def __init__(self, root, transform=None, train=True, test=False):
        """ Intialize the Face dataset """
        self.root = root
        self.transform = transform
        self.videos = []
        self.labels = []
        self.train = train
        self.test = test

        if self.test:
            video_path = self.root

            # read videos
            video_folders = sorted(glob.glob(os.path.join(video_path, '**/')))

            for folder in video_folders:
                videos = sorted(glob.glob(os.path.join(folder, '*.jpg')))
                self.videos.append(np.array(videos)) # [[fn1, fn2...], ...]
        else:
            # train or valid
            if self.train:
                video_path = os.path.join(self.root, 'videos', 'train')
                label_path = os.path.join(self.root, 'labels', 'train')

            else:
                video_path = os.path.join(self.root, 'videos', 'valid')
                label_path = os.path.join(self.root, 'labels', 'valid')

            # read videos
            video_folders = sorted(glob.glob(os.path.join(video_path, '**/')))

            for folder in video_folders:
                videos = sorted(glob.glob(os.path.join(folder, '*.jpg')))
                self.videos.append(np.array(videos)) # [[fn1, fn2...], ...]

            # read labels
            label_txts = sorted(glob.glob(os.path.join(label_path, '*.txt')))

            for txt in label_txts:
                f = open(txt, 'r')
                lines = f.readlines()
                labels = [int(label.split()[0]) for label in lines]
                self.labels.append(np.array(labels)) # [[1,0,0,...], ...]

        # for i in range(len(self.videos)):
        #     print(self.videos[i].shape, self.labels[i].shape)

        self.video_category = sorted(os.listdir(video_path))
        # print(self.video_category)

        # number of samples
        self.len = len(self.videos)
        print('Number of samples:',self.len)


    def __getitem__(self, index):
        """ Get a sample from the dataset """
        length = len(self.videos[index])
        rand_frame_num = np.sort(np.random.choice(length, self.sample_size, replace=False))

        if not self.test:
            # np.set_printoptions(precision=2)
            unique, counts = np.unique(self.labels[index], return_counts=True)
            ratio = counts/length
            weight = []
            j = 0
            for i in range(11):
                if i == unique[j]:
                    weight.append(1-ratio[j])
                    if j < len(unique)-1:
                        j += 1
                else:
                    weight.append(0)
            weight = torch.Tensor(weight)

            labels = torch.Tensor(self.labels[index][rand_frame_num]).long()

        videos = self.videos[index][rand_frame_num]
        frames_list = []
        for frame_name in videos:
            frame = Image.open(frame_name)
            width, height = frame.size[0], frame.size[1]

            frame = frame.crop(((width-height)/2, 0, width-(width-height)/2, height))
            frame = frame.resize((self.image_size, self.image_size))
            # frame = cv2.imread(frame_name)
            # frame = cv2.resize(frame, (self.image_size, self.image_size))
            # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            if self.transform is not None:
                frame = self.transform(frame)
            frames_list.append(frame.unsqueeze(0))

        frames = torch.cat(frames_list, dim=0)

        if self.test:
            return frames, rand_frame_num, length
        else:
            return frames, self.labels[index], rand_frame_num, labels, length, weight


    def __len__(self):
        """ Total number of samples in the dataset """
        return self.len


def p3_collate_fn(batch):
    frames = torch.Tensor()
    labels = torch.Tensor()
    rand_frame_nums = torch.Tensor()
    for i in range(len(batch)):
        frame, label, rand_frame_num = batch[i]
        frames = torch.cat((frames, frame), dim=0)
        labels = torch.cat((labels, label), dim=0)
        rand_frame_nums = torch.cat((rand_frame_nums, rand_frame_num), dim=0)

    return frames, labels, rand_frame_nums


def FullLengthVideos_test():
    dataroot = os.path.join(os.getcwd(), 'hw4_data', 'FullLengthVideos')
    transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ])
    dataset = FullLengthVideos(root=dataroot, transform=transform, train=False)
    # for i in range(len(dataset.video_category)):
    #     print(dataset.video_category[i])
    frames, all_labels, rand_frame_nums, labels, length, weight = dataset.__getitem__(0)
    print(frames.shape, frames.dtype)
    print(all_labels.shape, all_labels.dtype)
    print(rand_frame_nums.shape, rand_frame_nums.dtype)
    print(labels.shape, labels.dtype)
    print(length)
    print(weight.shape, weight.dtype)

    # dataloader = torch.utils.data.DataLoader(dataset, batch_size=1,
    #                                          shuffle=False, num_workers=6)
    # dataiter = iter(dataloader)
    # frames, all_labels, rand_frame_nums, labels, length = dataiter.next()
    #
    # print('Frame tensor in each batch:', frames.shape, frames.dtype)
    # print('all_labels tensor in each batch:', len(all_labels))
    # print('rand_frame_nums tensor in each batch:', rand_frame_nums.shape, rand_frame_nums.dtype)
    # print('labels tensor in each batch:', labels.shape, labels.dtype)
    # print('length tensor in each batch:', length.shape, length.dtype)
    # print(labels)
    # print(lens)


if __name__ == '__main__':
    # TrimmedVideos_test()
    FullLengthVideos_test()
