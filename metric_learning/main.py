import optparse
import os

import cv2
import dlib
import numpy as np
from imutils.face_utils import FaceAligner
from keras.optimizers import adam
from keras.utils import to_categorical
from skimage.io import imread
from skimage.transform import resize
from tensorflow.python.keras import Model
from tensorflow.python.keras import backend as K
from tensorflow.python.keras import optimizers, losses
from tensorflow.python.keras.callbacks import LearningRateScheduler, ModelCheckpoint
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.layers import Layer, Input

from metric_learning.generator import Generator
from metric_learning.resnet34 import Resnet34

output_len = 128
input_image_size = 128


def get_images(files):
    images = []
    for file in files:
        img = cv2.imread(file)
        img = cv2.resize(img, (input_image_size, input_image_size))
        images.append(img)

    return np.array(images)


def step_decay(epoch):
    if epoch < 70:
        return 0.01
    elif 70 <= epoch < 150:
        return 0.005
    elif 150 <= epoch < 250:
        return 0.0001
    else:
        return 0.00005


def create_resnet():
    resnet = Resnet34(input_image_size, output_len, drop=drop, arch=arch)
    return resnet.create_model()


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
    side = CenterLossLayer(name='centerlosslayer')([resnet.output, aux_input])

    model = Model(inputs=[resnet.input, aux_input], outputs=[main, side])
    optim = optimizers.Adam(lr=lr)
    model.compile(optimizer=optim,
                  loss=[losses.categorical_crossentropy, zero_loss],
                  loss_weights=[1, center_weight], metrics=['accuracy'])

    train_features, train_labels = get_files(train)
    test_features, test_labels = get_files(test)
    # p = np.random.permutation(len(all_files))
    # all_files = all_files[p]
    # all_labels = all_labels[p]

    filepath = "weights-improvement-{val_loss:.2f}-epch = {epoch:02d}- acc={val_main_out_acc:.2f}.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=False, mode='max')
    callbacks = [checkpoint]

    if lr < 0:
        callbacks.append(LearningRateScheduler(step_decay))

    if options.prev_weights and os.path.exists(options.prev_weights):
        model.load_weights(options.prev_weights)

    training_generator = Generator(train_features, train_labels, batch_size, class_name_max)
    test_generator = Generator(test_features, test_labels, batch_size, class_name_max)

    model.fit_generator(training_generator,
                        epochs=epochs,
                        verbose=verbose,
                        validation_data=test_generator,
                        callbacks=callbacks
                        )


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

        image = face_align(image)
        image = np.array(resize(image, (128, 128)))
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


def face_align(img):
    predictor = dlib.shape_predictor("../shape_predictor_68_face_landmarks.dat")
    face_aligner = FaceAligner(predictor=predictor, desiredFaceHeight=128, desiredFaceWidth=128)
    detector = dlib.get_frontal_face_detector()

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    rects = detector(img_gray, 2)
    aligned_face = face_aligner.align(img, img_gray, rects[0])
    return aligned_face


def integration_test():
    model = create_resnet()
    num_classes = 10
    from keras.datasets import cifar10
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    y_train = to_categorical(y_train, num_classes)
    y_test = to_categorical(y_test, num_classes)
    opt = adam(lr=lr, decay=1e-6)

    model.compile(loss='categorical_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'])

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255

    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=2,
              validation_data=(x_test, y_test),
              shuffle=True)


if __name__ == '__main__':
    parser = optparse.OptionParser()
    parser.add_option('--train', type='string')
    parser.add_option('--test', type='string')
    parser.add_option('--classes', type='int')
    parser.add_option('--lr', type='float')
    parser.add_option('--center', type='float')
    parser.add_option('--k_r', type='float', default=0.)
    parser.add_option('--b_r', type='float', default=0.)
    parser.add_option('--batch', type='int')
    parser.add_option('--epochs', type='int')
    parser.add_option('--verbose', type='int')
    parser.add_option('--alpha', type='float')
    parser.add_option('--aug', action='store_true', dest='aug', default=False)
    parser.add_option('--arch', default='resnet')
    parser.add_option('--prev_weights', type='string')
    parser.add_option('--weights', type='string')
    parser.add_option('--mode', type='string')
    parser.add_option('--urls', type='string')
    parser.add_option('--drop', type='float', default=0.)

    (options, args) = parser.parse_args()

    lr = options.lr
    center_weight = options.center
    batch_size = options.batch
    epochs = options.epochs
    class_name_max = options.classes
    verbose = options.verbose
    arch = options.arch
    drop = options.drop
    train = options.train
    test = options.test
    if options.mode == 'train':
        train_resnet()
    elif options.mode == 'test':
        urls = options.urls.split(',')
        find_distance(urls)
    elif options.mode == 'integr':
        integration_test()
