import optparse
import os

import numpy as np
from keras.layers import BatchNormalization
from skimage.io import imread
from skimage.transform import resize
from tensorflow.keras import Model
from tensorflow.keras.layers import Input
from tensorflow.python.keras.layers import Conv2D, MaxPool2D, Dense, BatchNormalization, Activation, \
    GlobalAveragePooling2D

from metric_learning.resnet34 import level4, level3, level2, level1, level0

parser = optparse.OptionParser()
parser.add_option('--dataset')
parser.add_option('--pairs')
parser.add_option('--thr')
parser.add_option('--weights')
(options, args) = parser.parse_args()


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
    resnset = create_resnet()
    if not os.path.exists(options.weights):
        print('can not find weights')
        exit(1)
    resnset.load_weights(options.weights, by_name=True)
    thr = float(options.thr)
    count = len(pairs)
    right_answers = 0
    for idx, pair in enumerate(pairs):
        im0 = resize(imread(pair[0]), (128, 128))
        im1 = resize(imread(pair[1]), (128, 128))
        imgs = np.array([im0, im1])
        inferences = resnset.predict(imgs)
        dist = np.linalg.norm(inferences[0] - inferences[1])
        if dist > thr:
            cur_positive = False
        else:
            cur_positive = True

        if positive[idx] == cur_positive:
            right_answers += 1

    print('accuracy: ', right_answers / count)


def create_resnet():
    image_input = Input(shape=(128, 128, 3))
    prev = Conv2D(37, (7, 7), (2, 2))(image_input)
    prev = Activation('relu')(prev)
    prev = BatchNormalization()(prev)
    prev = MaxPool2D(pool_size=(3, 3), strides=(2, 2))(prev)

    prev = level4(prev)
    prev = level3(prev)
    prev = level2(prev)
    prev = level1(prev)
    prev = level0(prev)
    prev = GlobalAveragePooling2D()(prev)
    output = Dense(128, use_bias=False)(prev)
    return Model(image_input, output)


if __name__ == '__main__':
    main()
