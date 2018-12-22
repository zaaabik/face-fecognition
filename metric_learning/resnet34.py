from keras.constraints import maxnorm
from tensorflow.python.keras.layers import BatchNormalization, Conv2D, add, AvgPool2D, \
    Activation
from tensorflow.python.keras.regularizers import l2

max_norm = 4
kernel_regularizer = l2(0.01)
bias_regularizer = l2(0.01)


def conv_block(feat_maps_out, prev, strides):
    prev = Conv2D(feat_maps_out, (3, 3), strides=strides, padding='same', kernel_constraint=maxnorm(max_norm),
                  kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer)(prev)
    prev = BatchNormalization()(prev)  # Specifying the axis and mode allows for later merging
    prev = Activation('relu')(prev)
    prev = Conv2D(feat_maps_out, (3, 3), padding='same', kernel_constraint=maxnorm(max_norm),
                  kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer)(prev)
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
