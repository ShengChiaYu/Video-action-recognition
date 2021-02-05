import torch.optim as optim
import torch.utils.data
import torch.nn as nn
from torchvision import datasets
from torchvision import transforms

import numpy as np
import matplotlib.pyplot as plt
import random
import os
import sys

# from sklearn.manifold import TSNE

from tqdm import tqdm
from dataset import TrimmedVideos, collate_fn, FullLengthVideos, p3_collate_fn
from utils import AverageMeter
from model import p1_classifier, p2_rnn, p3_rnn


def load_data(batchSize, problem, video_root, label_root):
    torch.manual_seed(1216)

    # dataroot = os.path.join(os.getcwd(), 'hw4_data', 'TrimmedVideos')
    video_root = os.path.join(os.getcwd(), video_root)
    label_root = os.path.join(os.getcwd(), label_root)
    dataroot = [video_root, label_root]

    transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ])
    valid_dataset = TrimmedVideos(root=dataroot, transform=transform, train=False, problem=problem, test=True)

    valid_loader = torch.utils.data.DataLoader(
        dataset=valid_dataset,
        batch_size=batchSize,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        drop_last=False,
        collate_fn=collate_fn,
        )

    return valid_loader


def load_checkpoint(checkpoint_path, model, optimizer):
    state = torch.load(checkpoint_path)
    model.load_state_dict(state['state_dict'])
    # optimizer.load_state_dict(state['optimizer'])
    print('model loaded from %s' % checkpoint_path)


def test(cls, test_loader, batchSize, filename, problem, output_root):
    cls.eval()
    # correct = 0
    load_path = os.path.join(os.getcwd(), output_root, filename)
    f = open(load_path, 'w')
    test_pbar = tqdm(total=len(test_loader), ncols=100, leave=True)
    count = 1
    with torch.no_grad(): # This will free the GPU memory used for back-prop
        for i, (frames, labels, lens) in enumerate(test_loader):
            frames, labels, lens = frames.cuda(), labels.cuda(), lens.cuda()
            if problem == 1:
                _, class_output = cls(frames, lens)
            if problem == 2:
                _, class_output = cls(frames)

            pred = class_output.max(1, keepdim=True)[1] # get the index of the max log-probability
            for label in pred:
                if count == len(test_loader.dataset):
                    f.write('{}'.format(label.data.item()))
                else:
                    f.write('{}\n'.format(label.data.item()))
                count += 1

            # correct += pred.eq(labels.view_as(pred)).sum().item()
            test_pbar.update()
    # test_acc = 100.*correct / len(test_loader.dataset)
    # test_pbar.set_postfix({'acc':'{:.4f}'.format(test_acc)})

    f.close()
    test_pbar.close()


def p1_predict(video_root, label_root, output_root):
    batchSize = 4
    learningRate = 1e-3
    problem = 1
    valid_loader = load_data(batchSize, problem, video_root, label_root)

    # models
    cls = p1_classifier().cuda()
    optimizer = optim.SGD(cls.parameters(),  lr=learningRate, momentum=0.9)

    checkpoint_path = os.path.join(os.getcwd(), 'p1_best.pth')
    load_checkpoint(checkpoint_path, cls, optimizer)

    test(cls, valid_loader, batchSize, 'p1_valid.txt', problem, output_root)


def p2_predict(video_root, label_root, output_root):
    batchSize = 10
    learningRate = 1e-3
    problem = 2
    valid_loader = load_data(batchSize, problem, video_root, label_root)

    # models
    cls = p2_rnn().cuda()
    optimizer = optim.SGD(cls.parameters(),  lr=learningRate, momentum=0.9)

    checkpoint_path = os.path.join(os.getcwd(), 'p2_best_4.pth')
    load_checkpoint(checkpoint_path, cls, optimizer)

    test(cls, valid_loader, batchSize, 'p2_result.txt', problem, output_root)


def p3_predict(video_root, output_root):
    torch.manual_seed(1216)
    np.random.seed(1216)
    batchSize = 1
    num_epochs = 40
    learningRate = 1e-3
    image_size = 224

    # dataroot = os.path.join(os.getcwd(), 'FullLengthVideos')
    dataroot = os.path.join(os.getcwd(), video_root)
    transform = transforms.Compose([
                    # transforms.Resize(image_size),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ])

    valid_dataset = FullLengthVideos(root=dataroot, transform=transform, train=False, test=True)

    valid_loader = torch.utils.data.DataLoader(
        dataset=valid_dataset,
        batch_size=batchSize,
        shuffle=False,
        num_workers=6,
        pin_memory=True,
        drop_last=False,
        # collate_fn=p3_collate_fn,
        )

    # models
    cls = p3_rnn().cuda()
    params_to_update = []
    for name, param in cls.named_parameters():
        if param.requires_grad:
            params_to_update.append(param)
            # print(name)
    # optimizer = optim.SGD(cls.parameters(),  lr=learningRate, momentum=0.9)
    optimizer = optim.Adam(params_to_update, lr=learningRate)
    checkpoint_path = os.path.join(os.getcwd(), 'p3_best.pth')
    load_checkpoint(checkpoint_path, cls, optimizer)

    cls.eval()
    # correct = 0
    # count = 0
    test_pbar = tqdm(total=len(valid_loader), ncols=100, leave=True)
    with torch.no_grad(): # This will free the GPU memory used for back-prop
        for i, (frames, rand_frame_nums, length) in enumerate(valid_loader):
            frames = frames.cuda()

            filename = valid_dataset.video_category[i] + '.txt'
            load_path = os.path.join(os.getcwd(), output_root, filename)
            f = open(load_path, 'w')

            class_output = cls(frames[0])
            pred = class_output.max(1, keepdim=True)[1] # get the index of the max log-probability

            # print(pred.view(20,-1))
            check_point = 0
            label = 0
            for j in range(length):
                if j == rand_frame_nums[0][check_point]:
                    label = pred[check_point].data.item()
                    if check_point < rand_frame_nums.shape[1]-1:
                        check_point += 1
                # if label == all_labels[0][j]:
                #    correct += 1
                f.write('{}\n'.format(label))
            # count += length.data.item()

            test_pbar.update()
            f.close()

    # test_acc = 100.*correct / count
    # test_pbar.set_postfix({'acc':'{:.4f}'.format(test_acc)})

    test_pbar.close()


def p1_plot_tsne():
    batchSize = 4
    learningRate = 1e-3
    problem = 1
    valid_loader = load_data(batchSize, problem)

    # models
    cls = p1_classifier().cuda()
    optimizer = optim.SGD(cls.parameters(),  lr=learningRate, momentum=0.9)

    checkpoint_path = os.path.join(os.getcwd(), 'models', 'p1_best.pth')
    load_checkpoint(checkpoint_path, cls, optimizer)

    # generate features
    cls.eval()
    test_pbar = tqdm(total=len(valid_loader), ncols=100, leave=True)
    features = torch.Tensor().cuda()
    gt = torch.Tensor().long().cuda()
    with torch.no_grad():
        for i, (frames, labels, lens) in enumerate(valid_loader):
            frames, labels, lens = frames.cuda(), labels.cuda(), lens.cuda()

            feature, _ = cls(frames, lens)
            features = torch.cat((features, feature), dim=0)
            gt = torch.cat((gt, labels), dim=0)
            test_pbar.update()
    test_pbar.close()

    # plot tsne
    tsne = TSNE(n_components=2, random_state=0)

    features = np.array(features.cpu())
    gt = np.array(gt.cpu())

    features = tsne.fit_transform(features)
    features.tofile('prediction/p1_tsne_labels.dat')
    # features = np.fromfile('prediction/p1_tsne_labels.dat', dtype=int)

    # plot labels
    plt.figure(figsize=(6, 5))
    plt.title('p1_tsne_labels')
    for i in range(11):
        num = gt == i
        if i >= 7:
            plt.scatter(features[num,0], features[num,1], c=[plt.cm.Set1(i-7)], label=str(i))
        else:
            plt.scatter(features[num,0], features[num,1], c=[plt.cm.Set2(i)], label=str(i))
    plt.legend()
    save_path = os.path.join(os.getcwd(), 'prediction', 'p1_tsne_labels.jpg')
    plt.savefig(save_path, bbox_inches='tight')
    print('output jpg path:{}'.format(save_path))


def p2_plot_tsne():
    batchSize = 10
    learningRate = 1e-3
    problem = 2
    valid_loader = load_data(batchSize, problem)

    # models
    cls = p2_rnn().cuda()
    optimizer = optim.SGD(cls.parameters(),  lr=learningRate, momentum=0.9)

    checkpoint_path = os.path.join(os.getcwd(), 'models', 'p2_best_4.pth')
    load_checkpoint(checkpoint_path, cls, optimizer)

    # generate features
    cls.eval()
    test_pbar = tqdm(total=len(valid_loader), ncols=100, leave=True)
    features = torch.Tensor().cuda()
    gt = torch.Tensor().long().cuda()
    with torch.no_grad():
        for i, (frames, labels, lens) in enumerate(valid_loader):
            frames, labels, lens = frames.cuda(), labels.cuda(), lens.cuda()

            feature, _ = cls(frames)
            features = torch.cat((features, feature), dim=0)
            gt = torch.cat((gt, labels), dim=0)
            test_pbar.update()
    test_pbar.close()

    # plot tsne
    tsne = TSNE(n_components=2, random_state=0)

    features = np.array(features.cpu())
    gt = np.array(gt.cpu())

    features = tsne.fit_transform(features)
    features.tofile('prediction/p2_tsne_labels.dat')
    # features = np.fromfile('prediction/p1_tsne_labels.dat', dtype=int)

    # plot labels
    plt.figure(figsize=(6, 5))
    plt.title('p2_tsne_labels')
    for i in range(11):
        num = gt == i
        if i >= 7:
            plt.scatter(features[num,0], features[num,1], c=[plt.cm.Set1(i-7)], label=str(i))
        else:
            plt.scatter(features[num,0], features[num,1], c=[plt.cm.Set2(i)], label=str(i))
    plt.legend()
    save_path = os.path.join(os.getcwd(), 'prediction', 'p2_tsne_labels.jpg')
    plt.savefig(save_path, bbox_inches='tight')
    print('output jpg path:{}'.format(save_path))


def main():
    if sys.argv[1] == 'p1':
        p1_predict(sys.argv[2], sys.argv[3], sys.argv[4])

    elif sys.argv[1] == 'p2':
        p2_predict(sys.argv[2], sys.argv[3], sys.argv[4])

    elif sys.argv[1] == 'p3':
        p3_predict(sys.argv[2], sys.argv[3])

    elif sys.argv[1] == 'p1_tsne':
        p1_plot_tsne()

    elif sys.argv[1] == 'p2_tsne':
        p2_plot_tsne()


if __name__ == '__main__':
    main()
