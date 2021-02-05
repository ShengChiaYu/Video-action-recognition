import matplotlib.pyplot as plt
import os
import sys
import argparse


def read_log(filename):
    load_path = os.path.join(os.getcwd(), 'models', filename)
    f = open(load_path, 'r')
    lines = f.readlines()
    # print(float(lines[0].split()[4].split('/')[1].split(',')[0]))
    # sys.exit()
    train_loss = []
    valid_loss = []
    train_acc = []
    valid_acc = []
    for i in range(len(lines)):
        train_loss.append(float(lines[i].split()[4].split('/')[0]))
        valid_loss.append(float(lines[i].split()[4].split('/')[1].split(',')[0]))
        train_acc.append(float(lines[i].split()[6].split('/')[0]))
        valid_acc.append(float(lines[i].split()[6].split('/')[1].split(',')[0]))

    f.close()

    return train_loss, train_acc, valid_loss, valid_acc


def plot(title, train, valid):
    plt.figure()
    plt.title(title)
    plt.plot(train, label='train')
    plt.plot(valid, label='valid')
    plt.legend()
    save_path = os.path.join(os.getcwd(), 'models', '{}.jpg'.format(title))
    plt.savefig(save_path)
    print('output jpg path:{}'.format(save_path))

def main():
    train_loss, train_acc, valid_loss, valid_acc = read_log('p3_err.txt')
    plot('p3_train_valid_loss', train_loss, valid_loss)
    plot('p3_train_valid_acc', train_acc, valid_acc)

if __name__ == '__main__':
    main()
