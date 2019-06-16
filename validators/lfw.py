import optparse
import os

import matplotlib.pyplot as plt
import numpy as np
from skimage.io import imread
from skimage.transform import resize
from tensorflow.python.keras import Model
from tensorflow.python.keras.backend import l2_normalize
from tensorflow.python.keras.layers import Lambda

from helpers.helpers import get_images, mkdir_p
from metric_learning.resnet34 import Resnet34


def cosin_metric(x1, x2):
    return np.dot(x1, x2) / (np.linalg.norm(x1) * np.linalg.norm(x2))


def cal_accuracy(y_score, y_true):
    y_score = np.asarray(y_score)
    y_true = np.asarray(y_true)
    best_acc = 0
    best_th = 0
    for i in range(len(y_score)):
        th = y_score[i]
        y_test = (y_score >= th)
        acc = np.mean((y_test == y_true).astype(int))
        if acc > best_acc:
            best_acc = acc
            best_th = th

    return (best_acc, best_th)


def read_pairs_file(path):
    pairs = []
    with open(path, 'r') as f:
        for line in f.readlines()[1:]:
            pair = line.strip().split()
            pairs.append(pair)
    return np.array(pairs)


def create_pairs(base_dir, pairs):
    nrof_skipped_pairs = 0
    path_list = []
    issame_list = []
    for pair in pairs:
        try:
            if len(pair) == 3:
                path0 = add_extension(os.path.join(base_dir, pair[0], pair[0] + '_' + '%04d' % int(pair[1])))
                path1 = add_extension(os.path.join(base_dir, pair[0], pair[0] + '_' + '%04d' % int(pair[2])))
                issame = True
            elif len(pair) == 4:
                path0 = add_extension(os.path.join(base_dir, pair[0], pair[0] + '_' + '%04d' % int(pair[1])))
                path1 = add_extension(os.path.join(base_dir, pair[2], pair[2] + '_' + '%04d' % int(pair[3])))
                issame = False
            if os.path.exists(path0) and os.path.exists(path1):  # Only add the pair if both paths exist
                path_list.append((path0, path1))
                issame_list.append(issame)
            else:
                nrof_skipped_pairs += 1
        except:
            continue
    if nrof_skipped_pairs > 0:
        print('Skipped %d image pairs' % nrof_skipped_pairs)

    return path_list, issame_list


def add_extension(path):
    if os.path.exists(path + '.jpg'):
        return path + '.jpg'
    elif os.path.exists(path + '.png'):
        return path + '.png'
    else:
        raise RuntimeError('No file "%s" with extension png or jpg.' % path)


def main():
    pairs = read_pairs_file(options.pairs)
    pairs, positive = create_pairs(options.dataset, pairs)
    resnet = Resnet34(128, 128, arch=arch)
    resnset = resnet.create_model()
    if not os.path.exists(options.weights):
        print('can not find weights')
    else:
        print('weights are found')
        resnset.load_weights(options.weights, by_name=True)
    l2_layer = Lambda(lambda x: l2_normalize(x, 1))(resnset.output)
    resnset = Model(inputs=[resnset.input], outputs=[l2_layer])
    pairs = np.array(pairs)
    first_images = pairs[:, 0]
    second_images = pairs[:, 1]
    first_images = get_images(first_images, 128)
    second_images = get_images(second_images, 128)

    first_inferences = resnset.predict(first_images)
    second_inferences = resnset.predict(second_images)

    if flipped:
        first_images_flipped = np.flip(first_images, 2)
        second_images_flipped = np.flip(second_images, 2)
        first_inferences_flipped = resnset.predict(first_images_flipped)
        second_inferences_flipped = resnset.predict(second_images_flipped)
        first_inferences = (first_inferences + first_inferences_flipped) / 2
        second_inferences = (second_inferences + second_inferences_flipped) / 2

    # distances = np.linalg.norm(first_inferences - second_inferences, axis=1).flatten()
    distances = []
    for u, v in zip(first_inferences, second_inferences):
        metric = cosin_metric(u, v)
        print(metric)
        distances.append(metric)
    is_same = np.array(positive).flatten()

    best_acc, best_th = cal_accuracy(distances, is_same)
    print('best acc', best_acc)
    print('best thr', best_th)


def read_images(paths):
    images = []
    for path in paths:
        image = np.array(resize(imread(path), (128, 128)))
        images.append(image)
    return np.array(images)


def save_wrong_answers(img1, img2, dist, count, is_positive):
    folder_name = '/home/root/lfw_errors'
    mkdir_p(folder_name)
    if is_positive:
        positive = 'positive'
    else:
        positive = 'false'
    f = plt.figure()
    f.add_subplot(1, 2, 1)
    plt.axis('off')
    plt.imshow(((img1 / 2 + 0.5) * 255).astype(int))
    f.add_subplot(1, 2, 2)
    plt.axis('off')
    plt.imshow(((img2 / 2 + 0.5) * 255).astype(int))
    name = f'{count} thr {dist} {positive}.jpg'
    plt.savefig(os.path.join(folder_name, name))
    plt.clf()


def save_right_answers(img1, img2, dist, count):
    folder_name = '/home/root/lfw_ok'
    mkdir_p(folder_name)
    f = plt.figure()
    f.add_subplot(1, 2, 1)
    plt.axis('off')
    plt.imshow(((img1 / 2 + 0.5) * 255).astype(int))
    f.add_subplot(1, 2, 2)
    plt.axis('off')
    plt.imshow(((img2 / 2 + 0.5) * 255).astype(int))
    name = f'{count} thr {dist}.jpg'
    plt.savefig(os.path.join(folder_name, name))
    plt.clf()


if __name__ == '__main__':
    parser = optparse.OptionParser()
    parser.add_option('--dataset')
    parser.add_option('--pairs')
    parser.add_option('--weights')
    parser.add_option('--arch', default='resnet')
    parser.add_option('--flipped', default=False)
    parser.add_option('--step', type='float')
    (options, args) = parser.parse_args()
    flipped = bool(options.flipped)
    arch = options.arch
    main()
