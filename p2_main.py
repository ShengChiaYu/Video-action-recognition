import torch.optim as optim
import torch.utils.data
import torch.nn as nn
from torchvision import datasets
from torchvision import transforms

import numpy as np
import random
import os
import sys

from tqdm import tqdm
from dataset import TrimmedVideos, collate_fn
from utils import AverageMeter
from model import p2_rnn


def test(cls, test_loader, batchSize):
    criterion = nn.CrossEntropyLoss()
    cls.eval()
    correct = 0
    losses = AverageMeter()
    test_pbar = tqdm(total=len(test_loader), ncols=100, leave=True)
    with torch.no_grad(): # This will free the GPU memory used for back-prop
        for i, (frames, labels, lens) in enumerate(test_loader):
            frames, labels, lens = frames.cuda(), labels.cuda(), lens.cuda()

            class_output = cls(frames)
            loss = criterion(class_output, labels)
            pred = class_output.max(1, keepdim=True)[1] # get the index of the max log-probability

            losses.update(loss.data.item(), batchSize)
            correct += pred.eq(labels.view_as(pred)).sum().item()
            test_pbar.update()

            if i % 2 == 0:
                test_pbar.set_postfix({'loss':'{:.4f}'.format(losses.avg)})
    test_acc = 100.*correct / len(test_loader.dataset)
    test_pbar.set_postfix({'loss':'{:.4f}'.format(losses.avg),
                            'acc':'{:.4f}'.format(test_acc),
                            })
    test_pbar.close()
    # print('\nLoss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(losses.avg, correct, len(test_loader.dataset), test_acc))

    return test_acc, losses

def save_checkpoint(checkpoint_path, model, optimizer):
    state = {'state_dict': model.state_dict(),
             # 'state_dict_cls': classifier.state_dict(),
             'optimizer' : optimizer.state_dict(),
             # 'optimizer_cls' : optimizer_cls.state_dict()
             }
    torch.save(state, checkpoint_path)
    print('model saved to %s' % checkpoint_path)

def load_checkpoint(checkpoint_path, model, optimizer):
    pretrained_dict = torch.load(checkpoint_path)['state_dict']
    model_dict = model.state_dict()

    # 1. filter out unnecessary keys
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    # 2. overwrite entries in the existing state dict
    model_dict.update(pretrained_dict)
    # 3. load the new state dict
    model.load_state_dict(model_dict)

    # optimizer.load_state_dict(state['optimizer'])
    print('model loaded from %s' % checkpoint_path)

def main():
    torch.manual_seed(1216)
    batchSize = 10
    num_epochs = 40
    learningRate = 1e-3

    dataroot = os.path.join(os.getcwd(), 'hw4_data', 'TrimmedVideos')
    transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ])
    train_dataset = TrimmedVideos(root=dataroot, transform=transform, train=True, problem=2)
    valid_dataset = TrimmedVideos(root=dataroot, transform=transform, train=False, problem=2)

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=batchSize,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        drop_last=False,
        collate_fn=collate_fn,
        )

    valid_loader = torch.utils.data.DataLoader(
        dataset=valid_dataset,
        batch_size=batchSize,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        drop_last=False,
        collate_fn=collate_fn,
        )

    # models
    cls = p2_rnn().cuda()
    optimizer = optim.SGD(cls.parameters(),  lr=learningRate, momentum=0.9)
    checkpoint_path = os.path.join(os.getcwd(), 'models', 'p1_best.pth')
    load_checkpoint(checkpoint_path, cls, optimizer)

    # fix cnn weight
    for name, child in cls.named_children():
        if name == 'CNN':
            for param in child.parameters():
                param.requires_grad = False

    print('Training parameters')
    for name, param in cls.named_parameters():
        if param.requires_grad:
            print(name)

    best_acc = 0
    err_log_path = os.path.join(os.getcwd(), 'models', 'p2_err_4.txt')
    err_log = open(err_log_path, 'w')
    for epoch in range(num_epochs):
        criterion = nn.CrossEntropyLoss()
        cls.train()

        print ('\nEpoch = {}'.format(epoch+1))
        err_log.write('Epoch = {}, '.format(epoch+1))

        correct = 0
        losses = AverageMeter()
        train_pbar = tqdm(total=len(train_loader), ncols=100, leave=True)
        for i, (frames, labels, lens) in enumerate(train_loader):

            frames, labels, lens = frames.cuda(), labels.cuda(), lens.cuda()
            # print(frames.size())
            cls.zero_grad()

            _, class_output = cls(frames)
            loss = criterion(class_output, labels)
            loss.backward()

            pred = class_output.max(1, keepdim=True)[1]
            correct += pred.eq(labels.view_as(pred)).sum().item()

            optimizer.step()

            losses.update(loss.data.item(), batchSize)
            train_pbar.update()

            if i % 2 == 0:
                train_pbar.set_postfix({'loss':'{:.4f}'.format(losses.avg)})

        train_acc = 100.*correct / len(train_loader.dataset)
        train_pbar.set_postfix({'loss':'{:.4f}'.format(losses.avg),
                                'acc':'{:.4f}'.format(train_acc),
                                })
        train_pbar.close()

        valid_acc, valid_loss = test(cls, valid_loader, batchSize)

        if valid_acc > best_acc:
            best_acc = valid_acc
            checkpoint_path = os.path.join(os.getcwd(), 'models', 'p2_best_4.pth')
            save_checkpoint(checkpoint_path, cls, optimizer)

        err_log.write('Loss: {:.4f}/{:.4f}, Accuracy: {:.2f}/{:.2f}\n'.format(losses.avg, valid_loss.avg, train_acc, valid_acc))
        err_log.flush()

    err_log.write('Best valid_acc: {:.2f}\n'.format(best_acc))
    err_log.close()

if __name__ == '__main__':
    main()
