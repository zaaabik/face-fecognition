import optparse
import os

import cv2
import dlib
import numpy as np
from imutils.face_utils import FaceAligner
from matplotlib import pyplot as plt, cm
from skimage.io import imread
from skimage.transform import resize
from sklearn.model_selection import train_test_split
from tensorflow.python.keras import Model
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.layers import Layer, Input
from tensorflow.python.keras import optimizers, losses
from tensorflow.python.keras.callbacks import LearningRateScheduler, ModelCheckpoint
from tensorflow.python.keras.layers import Dense, Dropout
from tensorflow.python.keras.utils import to_categorical
from vis.utils import utils
from vis.visualization import visualize_cam, overlay

from metric_learning.callbacks import ValidateOnLfw
from metric_learning.generator import Generator
from metric_learning.resnet34 import Resnet34

output_len = 128
input_image_size = 128

parser = optparse.OptionParser()
parser.add_option('--dataset', type='string')
parser.add_option('--classes', type='int')
parser.add_option('--lr', type='float')
parser.add_option('--center', type='float')
parser.add_option('--k_r', type='float', default=0.)
parser.add_option('--b_r', type='float', default=0.)
parser.add_option('--batch', type='int')
parser.add_option('--epochs', type='int')
parser.add_option('--verbose', type='int')
parser.add_option('--alpha', type='float')
parser.add_option('--generator', action='store_true', dest='fit_generator')
parser.add_option('--aug', action='store_true', dest='aug', default=False)
parser.add_option('--sgd', action='store_true', dest='sgd')
parser.add_option('--arch', default='resnet')
parser.add_option('--prev_weights', type='string')
parser.add_option('--weights', type='string')
parser.add_option('--mode', type='string')
parser.add_option('--urls', type='string')
parser.add_option('--thr', type='float')
parser.add_option('--pairs', type='string')
parser.add_option('--filter_idx', type='int', default=None)
parser.add_option('--lfw', type='string')
parser.add_option('--drop', type='float', default=0.)
parser.add_option('--weights')

(options, args) = parser.parse_args()

lr = options.lr
center_weight = options.center
batch_size = options.batch
epochs = options.epochs
class_name_max = options.classes
verbose = options.verbose
alpha = options.alpha
fit_generator = options.fit_generator
kernel_regularization = options.k_r
bias_regularization = options.b_r
arch = options.arch
sgd = options.sgd
aug = options.aug
drop = options.drop
lfw = options.lfw
pairs = options.pairs
filter_idx = options.filter_idx
weights = options.weights


def visualize(image_urls, filter_idx):
    images = []
    for image_url in image_urls:
        image = imread(image_url)
        image = face_align(image)
        images.append(image)

    model = create_metric_resnet_without_centerloss()
    model.load_weights(weights, by_name=True)
    model.summary()
    penultimate_layer = utils.find_layer_idx(model, 'conv2d_19')
    layer_idx = utils.find_layer_idx(model, 'dense')

    for modifier in [None, 'guided', 'relu']:
        plt.figure()
        f, ax = plt.subplots(1, len(images))
        plt.suptitle("vanilla" if modifier is None else modifier)
        for i, img in enumerate(images):
            # 20 is the imagenet index corresponding to `ouzel`
            grads = visualize_cam(model, layer_idx, filter_indices=filter_idx,
                                  seed_input=img, penultimate_layer_idx=penultimate_layer,
                                  backprop_modifier=modifier)
            # Lets overlay the heatmap onto original image.
            jet_heatmap = np.uint8(cm.jet(grads) * 255)[..., 0]
            ax[i].imshow(overlay(jet_heatmap, img))
            plt.savefig(f'visualize {modifier}')


def get_images(files):
    images = []
    for file in files:
        img = cv2.imread(file)
        img = cv2.resize(img, (input_image_size, input_image_size))
        img = np.array(img) / 255
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
    resnet = Resnet34(kernel_regularization, bias_regularization, input_image_size, output_len, drop=drop, arch=arch)
    return resnet.create_model()


def create_metric_resnet():
    aux_input = Input((class_name_max,))
    resnet = create_resnet()
    main = Dropout(drop)(resnet.output)
    main = Dense(class_name_max, activation='softmax', name='main_out')(main)
    side = CenterLossLayer(alpha=alpha, name='centerlosslayer')([resnet.output, aux_input])

    model = Model(inputs=[resnet.input, aux_input], outputs=[main, side])
    return model


def create_metric_resnet_without_centerloss():
    resnet = create_resnet()
    main = Dropout(drop)(resnet.output)
    main = Dense(class_name_max, activation='softmax', name='main_out')(main)

    model = Model(inputs=[resnet.input], outputs=[main])
    return model


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
    optim = optimizers.Adam(lr=lr)
    if sgd:
        optim = optimizers.SGD(lr=lr, momentum=0.9, nesterov=True)
    model = create_metric_resnet()
    model.compile(optimizer=optim,
                  loss=[losses.categorical_crossentropy, zero_loss],
                  loss_weights=[1, center_weight], metrics=['accuracy'])

    all_files, all_labels = get_files(options.dataset)
    p = np.random.permutation(len(all_files))
    all_files = all_files[p]
    all_labels = all_labels[p]

    filepath = "weights-improvement-{epoch:02d}-{val_main_out_acc:.2f}.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_main_out_acc', verbose=1, save_best_only=False, mode='max')
    validate_on_lfw = ValidateOnLfw(pairs, lfw, class_name_max)
    callbacks = [checkpoint, validate_on_lfw]

    if lr < 0:
        callbacks.append(LearningRateScheduler(step_decay))

    if options.prev_weights and os.path.exists(options.prev_weights):
        model.load_weights(options.prev_weights)

    if fit_generator:
        x_train, x_test, y_train, y_test = train_test_split(all_files, all_labels, test_size=0.2)
        training_generator = Generator(x_train, y_train, input_image_size, batch_size, class_name_max, aug)
        test_generator = Generator(x_test, y_test, input_image_size, batch_size, class_name_max, False)

        history = model.fit_generator(training_generator,
                                      epochs=epochs,
                                      verbose=verbose,
                                      validation_data=test_generator,
                                      callbacks=callbacks
                                      )

        plt.plot(history.history['val_main_out_acc'])
        plt.plot(history.history['main_out_acc'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['validation data', 'train data'], loc='upper left')
        plt.savefig(f'training k_r={kernel_regularization} b_r={bias_regularization} lr={lr}.png')
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
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    face_aligner = FaceAligner(predictor=predictor, desiredFaceHeight=128, desiredFaceWidth=128)
    detector = dlib.get_frontal_face_detector()

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    rects = detector(img_gray, 2)
    aligned_face = face_aligner.align(img, img_gray, rects[0])
    return aligned_face


if __name__ == '__main__':
    print(aug)
    if options.mode == 'train':
        train_resnet()
    elif options.mode == 'test':
        urls = options.urls.split(',')
        find_distance(urls)
    elif options.mode == 'visualize':
        urls = options.urls.split(',')
        visualize(urls, filter_idx)
