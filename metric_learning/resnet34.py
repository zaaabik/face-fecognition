from tensorflow.python.keras import Sequential, Input
from tensorflow.python.keras.layers import Convolution2D, MaxPool2D, BatchNormalization, Conv2D, add, AvgPool2D, Activation


def resnet34(input_shape, output):
    model = Sequential()
    model.add(Input(shape=input_shape))
    model.add(Convolution2D(37, (7, 7), (2, 2), activation='relu'))
    model.add(MaxPool2D((3, 3), (2, 3)))


def conv_block(feat_maps_out, prev, strides):
    prev = Conv2D(feat_maps_out, (3, 3), strides=strides, padding='same')(prev)
    prev = BatchNormalization()(prev)  # Specifying the axis and mode allows for later merging
    prev = Activation('relu')(prev)
    prev = Conv2D(feat_maps_out, (3, 3), padding='same')(prev)
    prev = BatchNormalization()(prev)  # Specifying the axis and mode allows for later merging
    return prev


def skip_block(feat_maps_out, prev):
    if prev.shape[-1] != feat_maps_out:
        # This adds in a 1x1 convolution on shortcuts that map between an uneven amount of channels
        prev = Conv2D(feat_maps_out, (1, 1), padding='same')(prev)
    return prev


def Residual_down(output, prev):
    skip = skip_block(output, prev)
    skip = AvgPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(skip)
    conv = conv_block(output, prev, (2, 2))
    return add([skip, conv])


def Residual(output, prev):
    skip = skip_block(output, prev)
    conv = conv_block(output, prev, (1, 1))
    return add([skip, conv])


def Ares(prev, n):
    prev = Residual(n, prev)
    prev = Activation('relu')(prev)
    return prev


def Ares_down(prev, n):
    prev = Residual_down(n, prev)
    prev = Activation('relu')(prev)
    return prev


def level0(prev):
    prev = Ares_down(prev, 256)
    return prev


def level1(prev):
    prev = Ares(prev, 128)
    prev = Ares(prev, 256)
    prev = Ares(prev, 256)
    return prev


def level2(prev):
    prev = Ares(prev, 128)
    prev = Ares(prev, 128)
    prev = Ares(prev, 128)
    return prev


def level3(prev):
    prev = Ares(prev, 64)
    prev = Ares(prev, 64)
    prev = Ares(prev, 64)
    prev = Ares(prev, 64)
    return prev


def level4(prev):
    prev = Ares(prev, 32)
    prev = Ares(prev, 32)
    prev = Ares(prev, 32)
    return prev
