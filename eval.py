import os
import sys
import numpy as np
import cv2
import glob

bgr_color_code = [
                (87,201,0), # forest green
                (0,255,255), # yellow
                (87,227,207), # banana
                (255,105,65), # blue
                (0,0,255), # red
                (0,97,255), # orange
                (42,42,128), # brown
                (33,36,41), # black
                (240,32,160), # purple
                (192,192,192), # gray
                (203,192,255), # pink
                ]

def p1_p2(txtfile):
    # txtfile = 'p2_valid.txt'
    # txtfile = 'p3_result.txt'

    # load_path = os.path.join(os.getcwd(), 'prediction', txtfile)
    load_path = os.path.join(os.getcwd(), 'output', txtfile)
    pred = open(load_path, 'r')
    pred_lines = pred.readlines()

    # load_path = os.path.join(os.getcwd(), 'hw4_data', 'TrimmedVideos', 'label', 'gt_valid.csv')
    load_path = os.path.join(os.getcwd(), 'TrimmedVideos', 'label', 'gt_valid.csv')
    gt = open(load_path, 'r')
    gt_lines = gt.readlines()

    correct = 0
    for i in range(len(pred_lines)):
        if int(pred_lines[i].split()[0]) == int(gt_lines[i+1].split(',')[5]):
            correct += 1

    test_acc = correct / len(pred_lines) * 100
    print('{}/{}, test_acc = {:.2f}%'.format(correct, len(pred_lines), test_acc))

    pred.close()
    gt.close()


def p3(num, plot=False):
    num = int(num)
    video_category = [
    'OP01-R02-TurkeySandwich.txt',
    'OP01-R04-ContinentalBreakfast.txt',
    'OP01-R07-Pizza.txt',
    'OP03-R04-ContinentalBreakfast.txt',
    'OP04-R04-ContinentalBreakfast.txt',
    'OP05-R04-ContinentalBreakfast.txt',
    'OP06-R03-BaconAndEggs.txt',
    ]
    print(video_category[num])

    load_path = os.path.join(os.getcwd(), 'output', video_category[num])
    pred = open(load_path, 'r')
    pred_lines = pred.readlines()

    load_path = os.path.join(os.getcwd(), 'FullLengthVideos', 'labels', 'valid', video_category[num])
    gt = open(load_path, 'r')
    gt_lines = gt.readlines()

    # read videos
    video_path = os.path.join(os.getcwd(), 'FullLengthVideos', 'videos', 'valid')
    video_folders = sorted(glob.glob(os.path.join(video_path, '**/')))
    videos = sorted(glob.glob(os.path.join(video_folders[num], '*.jpg')))
    print(video_folders[num])

    # plot sequence images, predicted labels and ground truth labels
    frame_start = int(len(pred_lines) * 0.2)
    frame_end = int(len(pred_lines) * 0.6)
    frame_num = 10
    l_size = 1 # line size
    wh_ratio = 8

    frame_length = (frame_end - frame_start) * l_size
    img_h = int(frame_length/wh_ratio)
    img = np.zeros((img_h, frame_length, 3), np.uint8)
    img.fill(255) # fill white background

    bar_h = int(img_h/5)
    frame_h = img_h - bar_h*2
    frame_w = int(frame_length / frame_num)
    get_frame = int(frame_length / frame_w)

    correct = 0
    zeros_num = 0
    count = 0
    for i in range(len(pred_lines)):
        pred_label = int(pred_lines[i].split()[0])
        gt_label = int(gt_lines[i].split()[0])

        if pred_label == gt_label:
            correct += 1
        if gt_label == 0:
            zeros_num += 1

        if i >= frame_start and i < frame_end:
            j = (i - frame_start) * l_size
            cv2.line(img, (j, 0), (j, bar_h), bgr_color_code[gt_label], l_size)
            cv2.line(img, (j, img_h-bar_h), (j, img_h), bgr_color_code[pred_label], l_size)

            if j % frame_w == 0 and count < get_frame:
                frame = cv2.imread(videos[int(i+frame_w/2/l_size)])
                if count == (get_frame-1):
                    frame = cv2.resize(frame, (frame_length-j, frame_h))
                    img[bar_h: bar_h+frame_h, j:frame_length, :] = frame
                else:
                    frame = cv2.resize(frame, (frame_w, frame_h))
                    img[bar_h: bar_h+frame_h, j:j+frame_w, :] = frame
                count += 1


    zero_ratio = zeros_num / len(pred_lines) * 100
    test_acc = correct / len(pred_lines) * 100
    print('zero_ratio = {:.2f}%, {}/{}, test_acc = {:.2f}%'.format(zero_ratio, correct, len(pred_lines), test_acc))

    if plot:
        save_path = os.path.join(os.getcwd(), 'prediction', video_category[num].split('.')[0]+'.jpg')
        cv2.imwrite(save_path, img)


    pred.close()
    gt.close()


def legend():
    h = 100
    img = np.zeros((h, h*len(bgr_color_code), 3), np.uint8)
    img.fill(255) # fill white background

    for i in range(len(bgr_color_code)):
        cv2.circle(img, (int((0.5+i)*h), int(h/4)), int(h/4), bgr_color_code[i], -1)
        cv2.putText(img, '{}'.format(i), (int((i+0.3)*h), int(h*0.9)), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 0), 1, cv2.LINE_AA)

    save_path = os.path.join(os.getcwd(), 'prediction', 'legend.jpg')
    cv2.imwrite(save_path, img)



if __name__ == '__main__':
    if sys.argv[1] == 'p12':
        p1_p2(sys.argv[2])
    elif sys.argv[1] == 'p3':
        p3(sys.argv[2])
    elif sys.argv[1] == 'legend':
        legend()
