import optparse
import os

from tensorflow.keras import Model
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer, Input
from tensorflow.python.keras import optimizers, losses
from tensorflow.python.keras.callbacks import LearningRateScheduler
from tensorflow.python.keras.layers import Conv2D, Dense, Activation, \
    GlobalAveragePooling2D

from metric_learning.generator import Generator

output_len = 128
input_image_size = 128

parser = optparse.OptionParser()
parser.add_option('--dataset')
parser.add_option('--classes')
parser.add_option('--lr')
parser.add_option('--center')
parser.add_option('--batch')
parser.add_option('--epochs')
parser.add_option('--verbose')
(options, args) = parser.parse_args()

batch_size = int(options.batch)
lr = float(options.lr)
center_weight = float(options.center)
epochs = int(options.epochs)
class_name_max = int(options.classes)
verbose = int(options.verbose)


def step_decay(epoch):
    if epoch < 60:
        return 0.05
    elif 60 <= epoch < 80:
        return 0.005
    else:
        return 0.0005


def create_resnet():
    image_input = Input(shape=(input_image_size, input_image_size, 3))
    prev = Conv2D(37, (7, 7), (2, 2))(image_input)
    prev = Activation('relu')(prev)
    prev = GlobalAveragePooling2D()(prev)
    prev = Dense(output_len, use_bias=False)(prev)
    return Model(image_input, prev)


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
    side = CenterLossLayer(alpha=0.95, name='centerlosslayer')([resnet.output, aux_input])

    model = Model(inputs=[resnet.input, aux_input], outputs=[main, side])
    optim = optimizers.SGD(lr=lr, momentum=0.9)
    model.compile(optimizer=optim,
                  loss=[losses.categorical_crossentropy, zero_loss],
                  loss_weights=[1, center_weight], metrics=['accuracy'])
    all_files, all_labels = get_files(options.dataset)
    dataset_len = len(all_files)
    callbacks = []
    if lr < 0:
        callbacks.append(LearningRateScheduler(step_decay))

    model.fit_generator(Generator(all_files, all_labels, input_image_size, batch_size, class_name_max),
                        epochs=epochs,
                        steps_per_epoch=dataset_len // batch_size,
                        verbose=verbose,
                        callbacks=callbacks
                        )
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
    return all_files, all_labels


if __name__ == '__main__':
    train_resnet()
