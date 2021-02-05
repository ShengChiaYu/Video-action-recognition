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
from dataset import p3_collate_fn, FullLengthVideos
from utils import AverageMeter
from model import p3_rnn


def test(cls, test_loader, batchSize):
    criterion = nn.CrossEntropyLoss()
    cls.eval()
    correct = 0
    count = 0
    losses = AverageMeter()
    test_pbar = tqdm(total=len(test_loader), ncols=100, leave=True)
    with torch.no_grad(): # This will free the GPU memory used for back-prop
        for i, (frames, all_labels, rand_frame_nums, labels, length, weight) in enumerate(test_loader):
            frames, labels, weight = frames.cuda(), labels.cuda(), weight.cuda()

            class_output = cls(frames[0])
            criterion.weight = weight
            loss = criterion(class_output, labels[0])

            pred = class_output.max(1, keepdim=True)[1] # get the index of the max log-probability

            print(pred.view(20,-1))
            check_point = 0
            label = 0
            for j in range(length):
                if j == rand_frame_nums[0][check_point]:
                    label = pred[check_point].data.item()
                    if check_point < rand_frame_nums.shape[1]-1:
                        check_point += 1
                if label == all_labels[0][j]:
                    correct += 1
            count += length.data.item()

            losses.update(loss.data.item(), batchSize)
            test_pbar.update()

            test_acc = 100.*correct / count
            test_pbar.set_postfix({'loss':'{:.4f}'.format(losses.avg),
                                    'acc':'{:.4f}'.format(test_acc),
                                    })
    test_acc = 100.*correct / count
    test_pbar.set_postfix({'loss':'{:.4f}'.format(losses.avg),
                            'acc':'{:.4f}'.format(test_acc),
                            })
    test_pbar.close()

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
    torch.cuda.device(1)
    # np.random.seed(1216)
    batchSize = 1
    num_epochs = 40
    learningRate = 5e-5
    image_size = 224

    dataroot = os.path.join(os.getcwd(), 'hw4_data', 'FullLengthVideos')
    transform = transforms.Compose([
                    # transforms.Resize(image_size),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ])
    train_dataset = FullLengthVideos(root=dataroot, transform=transform, train=True)
    valid_dataset = FullLengthVideos(root=dataroot, transform=transform, train=False)

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=batchSize,
        shuffle=False,
        num_workers=6,
        pin_memory=True,
        drop_last=False,
        # collate_fn=p3_collate_fn,
        )

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

    # fix cnn weight
    for name, child in cls.named_children():
        if name == 'CNN':
            for param in child.parameters():
                param.requires_grad = False

    print('Training parameters')
    params_to_update = []
    for name, param in cls.named_parameters():
        if param.requires_grad:
            params_to_update.append(param)
            print(name)

    # optimizer = optim.SGD(params_to_update,  lr=learningRate, momentum=0.9)
    optimizer = optim.Adam(params_to_update, lr=learningRate)
    checkpoint_path = os.path.join(os.getcwd(), 'models', 'p3_best.pth')
    load_checkpoint(checkpoint_path, cls, optimizer)

    best_acc = 56.79
    err_log_path = os.path.join(os.getcwd(), 'models', 'p3_err.txt')
    err_log = open(err_log_path, 'w')
    for epoch in range(num_epochs):
        criterion = nn.CrossEntropyLoss()
        cls.train()

        print ('\nEpoch = {}'.format(epoch+1))
        err_log.write('Epoch = {}, '.format(epoch+1))

        correct = 0
        count = 0
        losses = AverageMeter()
        train_pbar = tqdm(total=len(train_loader), ncols=100, leave=True)
        for i, (frames, all_labels, rand_frame_nums, labels, length, weight) in enumerate(train_loader):

            frames, labels, weight = frames.cuda(), labels.cuda(), weight.cuda()

            cls.zero_grad()

            class_output = cls(frames[0])
            criterion.weight = weight
            loss = criterion(class_output, labels[0])
            loss.backward()

            pred = class_output.max(1, keepdim=True)[1]
            print(pred.view(20,-1))

            check_point = 0
            label = 0
            for j in range(length):
                if j == rand_frame_nums[0][check_point]:
                    label = pred[check_point].data.item()
                    if check_point < rand_frame_nums.shape[1]-1:
                        check_point += 1
                if label == all_labels[0][j]:
                    correct += 1
            count += length.data.item()

            optimizer.step()

            losses.update(loss.data.item(), batchSize)
            train_pbar.update()

            train_acc = 100.*correct / count
            train_pbar.set_postfix({'loss':'{:.4f}'.format(losses.avg),
                                    'acc':'{:.4f}'.format(train_acc),
                                    })

        train_acc = 100.*correct / count
        train_pbar.set_postfix({'loss':'{:.4f}'.format(losses.avg),
                                'acc':'{:.4f}'.format(train_acc),
                                })
        train_pbar.close()

        valid_acc, valid_loss = test(cls, valid_loader, batchSize)

        if valid_acc > best_acc:
            best_acc = valid_acc
            checkpoint_path = os.path.join(os.getcwd(), 'models', 'p3_best.pth')
            save_checkpoint(checkpoint_path, cls, optimizer)

        err_log.write('Loss: {:.4f}/{:.4f}, Accuracy: {:.2f}/{:.2f}\n'.format(losses.avg, valid_loss.avg, train_acc, valid_acc))
        err_log.flush()

    err_log.write('Best valid_acc: {:.2f}\n'.format(best_acc))
    err_log.close()

if __name__ == '__main__':
    main()
