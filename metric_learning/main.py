import optparse
import os

import cv2
import dlib
import numpy as np
from skimage.io import imread
from skimage.transform import resize
from sklearn.model_selection import train_test_split
from tensorflow.keras import Model
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer, Input
from tensorflow.python.keras import optimizers, losses
from tensorflow.python.keras.callbacks import LearningRateScheduler, ModelCheckpoint
from tensorflow.python.keras.layers import Conv2D, MaxPool2D, Dense, BatchNormalization, Activation, \
    GlobalAveragePooling2D, Dropout
from tensorflow.python.keras.utils import to_categorical

from metric_learning.generator import Generator
from metric_learning.resnet34 import level4, level1, level0, level3, level2

output_len = 128
input_image_size = 128

parser = optparse.OptionParser()
parser.add_option('--dataset', type='string')
parser.add_option('--classes', type='int')
parser.add_option('--lr', type='float')
parser.add_option('--center', type='float')
parser.add_option('--batch', type='int')
parser.add_option('--epochs', type='int')
parser.add_option('--verbose', type='int')
parser.add_option('--alpha', type='float')
parser.add_option('--generator', action='store_true', dest='fit_generator')
parser.add_option('--prev_weights', type='string')
parser.add_option('--weights', type='string')
parser.add_option('--mode', type='string')
parser.add_option('--urls', type='string')
parser.add_option('--thr', type='float')

(options, args) = parser.parse_args()

lr = options.lr
center_weight = options.center
batch_size = options.batch
epochs = options.epochs
class_name_max = options.classes
verbose = options.verbose
alpha = options.alpha
fit_generator = options.fit_generator


def get_images(files):
    images = []
    for file in files:
        img = cv2.imread(file)
        img = cv2.resize(img, (input_image_size, input_image_size))
        img = np.array(img) / 255
        images.append(img)

    return np.array(images)


def step_decay(epoch):
    if epoch < 60:
        return 0.05
    elif 60 <= epoch < 80:
        return 0.005
    else:
        return 0.0005


def create_resnet():
    image_input = Input(shape=(input_image_size, input_image_size, 3))
    prev = Dropout(0.1)(image_input)
    prev = Conv2D(37, (7, 7), (2, 2))(prev)
    prev = Activation('relu')(prev)
    prev = BatchNormalization()(prev)
    prev = MaxPool2D(pool_size=(3, 3), strides=(2, 2))(prev)

    prev = level4(prev)
    prev = Dropout(0.1)(prev)
    prev = level3(prev)
    prev = level2(prev)
    prev = level1(prev)
    prev = Dropout(0.1)(prev)
    prev = level0(prev)
    prev = GlobalAveragePooling2D()(prev)
    output = Dense(output_len, use_bias=False)(prev)
    return Model(image_input, output)


class CenterLossLayer(Layer):

    def __init__(self, alpha=0.5, **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha

    def build(self, input_shape):
        self.centers = self.add_weight(name='centers',
                                       shape=(class_name_max, output_len),
                                       initializer='uniform',
                                       trainable=False)
        super().build(input_shape)

    def call(self, x, mask=None):
        delta_centers = K.dot(K.transpose(x[1]), (K.dot(x[1], self.centers) - x[0]))
        center_counts = K.sum(K.transpose(x[1]), axis=1, keepdims=True) + 1
        delta_centers /= center_counts
        new_centers = self.centers - self.alpha * delta_centers
        self.add_update((self.centers, new_centers), x)
        self.result = x[0] - K.dot(x[1], self.centers)
        self.result = K.sum(self.result ** 2, axis=1, keepdims=True)
        return self.result

    def compute_output_shape(self, input_shape):
        return K.int_shape(self.result)


def zero_loss(y_true, y_pred):
    return 0.5 * K.sum(y_pred, axis=0)


def train_resnet():
    aux_input = Input((class_name_max,))
    resnet = create_resnet()
    main = Dense(class_name_max, activation='softmax', name='main_out')(resnet.output)
    side = CenterLossLayer(alpha=alpha, name='centerlosslayer')([resnet.output, aux_input])

    model = Model(inputs=[resnet.input, aux_input], outputs=[main, side])
    optim = optimizers.Adam(lr=lr)
    model.compile(optimizer=optim,
                  loss=[losses.categorical_crossentropy, zero_loss],
                  loss_weights=[1, center_weight], metrics=['accuracy'])
    all_files, all_labels = get_files(options.dataset)
    p = np.random.permutation(len(all_files))
    all_files = all_files[p]
    all_labels = all_labels[p]

    filepath = "weights-improvement-{epoch:02d}-{val_main_out_acc:.2f}.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_main_out_acc', verbose=1, save_best_only=True, mode='max')
    callbacks = [checkpoint]

    if lr < 0:
        callbacks.append(LearningRateScheduler(step_decay))

    if options.prev_weights and os.path.exists(options.prev_weights):
        model.load_weights(options.prev_weights)

    if fit_generator:
        x_train, x_test, y_train, y_test = train_test_split(all_files, all_labels, test_size=0.1, random_state=1)
        training_generator = Generator(x_train, y_train, input_image_size, batch_size, class_name_max)
        test_generator = Generator(x_test, y_test, input_image_size, batch_size, class_name_max)

        model.fit_generator(training_generator,
                            epochs=epochs,
                            steps_per_epoch=len(y_train) // batch_size,
                            verbose=verbose,
                            validation_data=test_generator,
                            validation_steps=len(y_test) // batch_size,
                            callbacks=callbacks
                            )
    else:
        images = get_images(all_files)
        dummy = np.zeros((np.array(images).shape[0], 1))
        hot_encoded_labels = to_categorical(all_labels, class_name_max)
        model.fit([images, hot_encoded_labels],
                  [hot_encoded_labels, dummy],
                  epochs=epochs,
                  validation_split=0.2,
                  batch_size=batch_size,
                  verbose=verbose)
    model.save_weights('resnet2d.h5')


def get_files(path):
    files_count = 0
    all_files = []
    all_labels = []
    folders = os.listdir(path)
    for current_label, folder in enumerate(folders):
        if current_label >= class_name_max:
            break
        cur = path + os.path.sep + folder
        files = os.listdir(cur)
        for idx, val in enumerate(files):
            files[idx] = cur + os.path.sep + files[idx]
        current_folder_files_count = len(files)

        current_folder_labels = [current_label] * current_folder_files_count
        all_labels.extend(current_folder_labels)

        files_count += current_folder_files_count
        all_files.extend(files)
        current_label += 1
    return np.array(all_files), np.array(all_labels)


def find_distance(image_urls):
    images = []
    paths = []
    for image_url in image_urls:
        image = imread(image_url)

        face_detector = dlib.get_frontal_face_detector()
        faces = face_detector(image, 1)
        if len(faces) == 0:
            continue

        face = faces[0]
        x = face.left()
        y = face.top()
        w = face.right() - x
        h = face.bottom() - y
        image = image[abs(y):y + h, abs(x):abs(x) + w]
        image = np.array(resize(image, (128, 128))) / 255
        images.append(image)
        paths.append(image_url)

    test_distance(np.array(images), paths)


def test_distance(images, paths):
    resnet = create_resnet()
    if (options.weights is not None) and os.path.exists(options.weights):
        resnet.load_weights(options.weights, by_name=True)
    else:
        raise Exception('Cant find weights !')
    inferences = resnet.predict(images)
    for idx, inference in enumerate(inferences):
        for idx2, inference2 in enumerate(inferences):
            if idx != idx2:
                dist = np.linalg.norm(inference - inference2)
                print(f'{idx} {idx2} dist = {dist} {paths[idx]} {paths[idx2]}')


if __name__ == '__main__':
    if options.mode == 'train':
        train_resnet()
    elif options.mode == 'test':
        urls = options.urls.split(',')
        find_distance(urls)
