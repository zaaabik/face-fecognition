from keras.applications.resnet50 import ResNet50
from tensorflow.keras import Model
from tensorflow.keras.layers import Input

from tensorflow.python.keras.layers import Conv2D, MaxPool2D, Dense, BatchNormalization, Activation, \
    GlobalAveragePooling2D, Dropout, Lambda
from tensorflow.python.keras.layers import add, AvgPool2D
from tensorflow.python.keras.regularizers import l2

import tensorflow.keras.backend as k

default_drop = 0.3
class Resnet34:
    def __init__(self, kernel_regularization, bias_regularization, input_size, output_size, drop=0., arch='resnet'):
        self.kernel_regularization = l2(kernel_regularization)
        self.bias_regularization = l2(bias_regularization)
        self.input_size = input_size
        self.output_size = output_size
        self.arch = arch
        self.drop = drop

    def conv_block(self, feat_maps_out, prev, strides):
        prev = Conv2D(feat_maps_out, (3, 3), strides=strides, padding='same', kernel_initializer='he_normal',
                      kernel_regularizer=self.kernel_regularization, bias_regularizer=self.bias_regularization)(prev)
        prev = BatchNormalization()(prev)  # Specifying the axis and mode allows for later merging
        prev = Activation('relu')(prev)
        prev = Conv2D(feat_maps_out, (3, 3), padding='same', kernel_regularizer=self.kernel_regularization,
                      bias_regularizer=self.bias_regularization)(prev)
        prev = BatchNormalization()(prev)  # Specifying the axis and mode allows for later merging
        return prev

    def skip_block(self, feat_maps_out, prev):
        if prev.shape[-1] != feat_maps_out:
            # This adds in a 1x1 convolution on shortcuts that map between an uneven amount of channels
            prev = Conv2D(feat_maps_out, (1, 1), padding='same')(prev)
        return prev

    def Residual_down(self, output, prev):
        skip = self.skip_block(output, prev)
        skip = AvgPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(skip)
        conv = self.conv_block(output, prev, (2, 2))
        return add([skip, conv])

    def Residual(self, output, prev):
        skip = self.skip_block(output, prev)
        conv = self.conv_block(output, prev, (1, 1))
        return add([skip, conv])

    def Ares(self, prev, n):
        prev = self.Residual(n, prev)
        prev = Activation('relu')(prev)
        return prev

    def Ares_down(self, prev, n):
        prev = self.Residual_down(n, prev)
        prev = Activation('relu')(prev)
        return prev

    def level0(self, prev):
        prev = self.Ares_down(prev, 256)
        return prev

    def level1(self, prev):
        prev = self.Ares(prev, 128)
        prev = self.Ares(prev, 256)
        prev = self.Ares(prev, 256)
        return prev

    def level2(self, prev):
        prev = self.Ares(prev, 128)
        prev = self.Ares(prev, 128)
        prev = self.Ares(prev, 128)
        return prev

    def level3(self, prev):
        prev = self.Ares(prev, 64)
        prev = self.Ares(prev, 64)
        prev = self.Ares(prev, 64)
        prev = self.Ares(prev, 64)
        return prev

    def level4(self, prev):
        prev = self.Ares(prev, 32)
        prev = self.Ares(prev, 32)
        prev = self.Ares(prev, 32)
        return prev

    def create_model(self):
        if self.arch == 'app':
            print('app')
            resnet = ResNet50(classes=128, pooling='max', input_shape=(128, 128, 3), weights=None)
            return resnet
        elif self.arch == 'resnet':
            print('resnet')
            image_input = Input(shape=(self.input_size, self.input_size, 3))
            prev = Conv2D(37, (7, 7), (2, 2), kernel_initializer='he_normal')(image_input)
            prev = Activation('relu')(prev)
            prev = BatchNormalization()(prev)
            prev = MaxPool2D(pool_size=(3, 3), strides=(2, 2))(prev)

            prev = self.level4(prev)
            prev = Dropout(default_drop)(prev)
            prev = self.level3(prev)
            prev = Dropout(default_drop)(prev)
            # prev = self.level2(prev)
            # prev = self.level1(prev)
            prev = self.level0(prev)
            prev = Dropout(default_drop)(prev)
            prev = GlobalAveragePooling2D()(prev)
            output = Dense(self.output_size, use_bias=False)(prev)
            return Model(image_input, output)
        elif self.arch == 'test':
            print('test')
            return self.__test_model()

    def __test_model(self):
        input_layer = Input(shape=(self.input_size, self.input_size, 3))

        prev = Conv2D(32, kernel_size=(3, 3), padding='same', kernel_initializer='he_normal')(input_layer)
        prev = Activation('relu')(prev)
        prev = MaxPool2D(pool_size=(2, 2))(prev)

        prev = Conv2D(64, (3, 3), kernel_initializer='he_normal')(prev)
        prev = Activation('relu')(prev)
        prev = MaxPool2D(pool_size=(2, 2))(prev)

        prev = Conv2D(128, kernel_size=(3, 3), padding='same', kernel_initializer='he_normal')(prev)
        prev = Activation('relu')(prev)
        prev = MaxPool2D(pool_size=(2, 2))(prev)

        prev = Conv2D(32, kernel_size=(3, 3), kernel_initializer='he_normal')(prev)
        prev = Activation('relu')(prev)
        prev = MaxPool2D(pool_size=(2, 2))(prev)

        prev = GlobalAveragePooling2D()(prev)

        prev = Dense(self.output_size, use_bias=False)(prev)
        return Model(input_layer, prev)
