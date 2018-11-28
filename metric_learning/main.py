import os
import random

import cv2
import numpy as np
from tensorflow import keras
from tensorflow.keras import Model
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer, Input
from tensorflow.python.keras import optimizers, losses
from tensorflow.python.keras.layers import Conv2D, MaxPool2D, Dense, BatchNormalization, Activation, \
    GlobalAveragePooling2D
from matplotlib import pyplot as plt

from metric_learning.resnet34 import level0, level1, level2, level3, level4

import optparse

class_name_max = 10
output_len = 128

parser = optparse.OptionParser()
parser.add_option('--dataset')
(options, args) = parser.parse_args()


class TripletLossLayer(Layer):
    def __init__(self, alpha=0.2, **kwargs):
        self.alpha = alpha
        super(TripletLossLayer, self).__init__(**kwargs)

    def triplet_loss(self, inputs):
        threshold = 0.6
        margin = 0.12
        a, p, n = inputs
        p_dist = K.sqrt(K.sum(K.square(a - p), axis=1))
        n_dist = K.sqrt(K.sum(K.square(a - n), axis=1))
        p_loss = K.clip(p_dist - threshold + margin, 0, np.inf)
        n_loss = K.clip(-n_dist + threshold + margin, 0, np.inf)
        return K.mean(p_loss) + K.mean(n_loss)

    def call(self, inputs):
        loss = self.triplet_loss(inputs)
        self.add_loss(loss)
        return loss


def scheduler(epoch):
    lr = 0.1
    return lr / (((epoch // 100) + 1) * 10)


def get_data(path=r'C:\datasets\vgg2'):
    folders = os.listdir(path)
    data = []
    short_labels = []
    class_name = 0
    for folder in folders:
        cur = path + os.pathsep + folder
        files = os.listdir(cur)
        for file in files:
            file = cur + os.pathsep + file
            img = cv2.imread(file)
            img = cv2.resize(img, (50, 50))
            short_labels.append(class_name)
            img = np.array(img)
            img = img.astype('float32')
            img /= 256
            data.append(img)
        class_name = class_name + 1
        if class_name >= class_name_max:
            break
    return np.array(data), np.array(short_labels)


def create_triplet(data, labels):
    labels = np.array(labels)
    random.shuffle(labels)
    data = np.array(data)
    random.shuffle(data)
    a = []
    p = []
    n = []

    for d, l in zip(data, labels):
        a.append(d)
        idxs = np.argwhere(labels != l)
        idx = random.choice(idxs)
        n.append(data[idx][0])
        idxs = np.argwhere(labels == l)
        idx = random.choice(idxs)
        p.append(data[idx][0])
    return np.array(a), np.array(p), np.array(n)


def create_triplet_generator(data, labels, batch_size):
    while True:
        labels = np.array(labels)
        random.shuffle(labels)
        data = np.array(data)
        random.shuffle(data)
        a = []
        p = []
        n = []

        i = 0
        for d, l in zip(data, labels):
            if i >= batch_size:
                break
            i += 1
            a.append(d)
            idxs = np.argwhere(labels != l)
            idx = random.choice(idxs)
            n.append(data[idx][0])
            idxs = np.argwhere(labels == l)
            idx = random.choice(idxs)
            p.append(data[idx][0])
        features = [np.array(a), np.array(p), np.array(n)]

        yield features, None


def get_test_image(path):
    img = cv2.imread(path)
    img = cv2.resize(img, (50, 50))
    img = np.array(img)
    img = img.astype('float32')
    img /= 256
    img = np.expand_dims(img, axis=0)
    return img


def distance(emb1, emb2):
    return np.sum(np.square(emb1 - emb2))


def mean_pred(y_true, y_pred):
    return K.mean(y_pred)


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
    prev = Dense(output_len, use_bias=False)(prev)
    return Model(image_input, prev)


class CenterLossLayer(Layer):

    def __init__(self, alpha=0.5, **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha

    def build(self, input_shape):
        self.centers = self.add_weight(name='centers',
                                       shape=(class_name_max, 128),
                                       initializer='uniform',
                                       trainable=False)
        # self.counter = self.add_weight(name='counter',
        #                                shape=(1,),
        #                                initializer='zeros',
        #                                trainable=False)  # just for debugging
        super().build(input_shape)

    def call(self, x, mask=None):
        # x[0] is Nx2, x[1] is Nx10 onehot, self.centers is 10x2
        delta_centers = K.dot(K.transpose(x[1]), (K.dot(x[1], self.centers) - x[0]))  # 10x2
        center_counts = K.sum(K.transpose(x[1]), axis=1, keepdims=True) + 1  # 10x1
        delta_centers /= center_counts
        new_centers = self.centers - self.alpha * delta_centers
        self.add_update((self.centers, new_centers), x)

        # self.add_update((self.counter, self.counter + 1), x)

        self.result = x[0] - K.dot(x[1], self.centers)
        self.result = K.sum(self.result ** 2, axis=1, keepdims=True)  # / K.dot(x[1], center_counts)
        return self.result  # Nx1

    def compute_output_shape(self, input_shape):
        return K.int_shape(self.result)


def zero_loss(y_true, y_pred):
    return 0.5 * K.sum(y_pred, axis=0)


def train_resnet():
    aux_input = Input((class_name_max,))
    resnet = create_resnet()
    main = Dense(class_name_max, activation='softmax', name='main_out')(resnet.output)
    side = CenterLossLayer(alpha=0.5, name='centerlosslayer')([resnet.output, aux_input])

    model = Model(inputs=[resnet.input, aux_input], outputs=[main, side])
    optim = optimizers.SGD(lr=1e-3, momentum=0.9)
    model.compile(optimizer=optim,
                  loss=[losses.categorical_crossentropy, zero_loss],
                  loss_weights=[1, 0.08])
    f, l = get_data(options.dataset)
    labels_one_hot = keras.utils.to_categorical(l, class_name_max)
    dummy = np.zeros((f.shape[0], 1))
    model.fit([f, labels_one_hot], [labels_one_hot, dummy], epochs=50, verbose=2, batch_size=32)
    model.save_weights('resnet2d.h5')


def test_reznet():
    res_net = create_resnet()

    a = get_test_image(r'C:\datasets\vgg2\n000149\0003_01.jpg')
    p = get_test_image(r'C:\datasets\vgg2\n000149\0014_01.jpg')
    n = get_test_image(r'C:\datasets\vgg2\n000009\0011_01.jpg')
    a = res_net.predict(a)
    p = res_net.predict(p)
    n = res_net.predict(n)
    p_dist = np.linalg.norm(a - p)
    n_dist = np.linalg.norm(a - n)
    print(p_dist)
    print(n_dist)

    res_net.load_weights('resnet_weights.h5', by_name=True)
    a = get_test_image(r'C:\datasets\vgg2\n000149\0003_01.jpg')
    p = get_test_image(r'C:\datasets\vgg2\n000149\0014_01.jpg')
    n = get_test_image(r'C:\datasets\vgg2\n000009\0011_01.jpg')
    a = res_net.predict(a)
    p = res_net.predict(p)
    n = res_net.predict(n)
    p_dist = np.linalg.norm(a - p)
    n_dist = np.linalg.norm(a - n)
    print(p_dist)
    print(n_dist)


def test_distance():
    d, l = get_data(r'C:\datasets\test')
    a, p, n = create_triplet(d, l)
    res_net = create_resnet()

    res = res_net.predict(a)
    res2 = res_net.predict(p)
    res3 = res_net.predict(n)
    p_dist = np.linalg.norm(res - res2, axis=1)
    n_dist = np.linalg.norm(res - res3, axis=1)
    print(p_dist.mean())
    print(n_dist.mean())

    res_net.load_weights('resnet_weights_full.h5', by_name=True)

    res = res_net.predict(a)
    res2 = res_net.predict(p)
    res3 = res_net.predict(n)
    p_dist = np.linalg.norm(res - res2, axis=1)
    n_dist = np.linalg.norm(res - res3, axis=1)
    print(p_dist.mean())
    print(n_dist.mean())


def print_result():
    d, l = get_data()
    resnet = create_resnet()

    resnet.load_weights('resnet2d.h5', by_name=True)
    dot = resnet.predict(d)
    c = ['#ff0000', '#ffff00', '#00ff00', '#00ffff', '#0000ff',
         '#ff00ff', '#990000', '#999900', '#009900', '#009999']
    print(dot)
    for d, l in zip(dot, l):
        plt.scatter(d[0], d[1], c=c[l], s=10)
    plt.title('after')
    # plt.title('before')
    plt.show()


if __name__ == '__main__':
    train_resnet()
    # test_reznet()
    # test_distance()
    # print_result()
