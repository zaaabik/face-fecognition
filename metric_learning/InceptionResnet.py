from __future__ import absolute_import
from __future__ import print_function

from functools import partial

from keras_applications.imagenet_utils import _obtain_input_shape
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.layers import Activation
from tensorflow.python.keras.layers import AveragePooling2D
from tensorflow.python.keras.layers import BatchNormalization
from tensorflow.python.keras.layers import Concatenate
from tensorflow.python.keras.layers import Conv2D
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.layers import Dropout
from tensorflow.python.keras.layers import GlobalAveragePooling2D
from tensorflow.python.keras.layers import Input
from tensorflow.python.keras.layers import Lambda
from tensorflow.python.keras.layers import MaxPooling2D
from tensorflow.python.keras.models import Model


def convolution_block(x,
                      filters,
                      kernel_size,
                      strides=1,
                      padding='same',
                      activation='relu',
                      using_bias=False,
                      name=None):
    x = Conv2D(filters,
               kernel_size,
               strides=strides,
               padding=padding,
               use_bias=using_bias,
               name=name)(x)
    if using_bias:
        bn_axis = 1 if K.image_data_format() == 'channels_first' else 3
        x = BatchNormalization(axis=bn_axis, scale=False)(x)
    if activation is not None:
        x = Activation(activation)(x)
    return x


def _generate_layer_name(name, branch_idx=None, prefix=None):
    if prefix is None:
        return None
    if branch_idx is None:
        return '_'.join((prefix, name))
    return '_'.join((prefix, 'Branch', str(branch_idx), name))


def _inception_resnet_block(x, scale, block_type, block_idx, activation='relu'):
    channel_axis = 1 if K.image_data_format() == 'channels_first' else 3

    if block_type == 'Block35':
        branch0 = convolution_block(x, 32, 1)
        branch1 = convolution_block(x, 32, 1)
        branch1 = convolution_block(branch1, 32, 3)
        branch_2 = convolution_block(x, 32, 1)
        branch_2 = convolution_block(branch_2, 48, 3)
        branch_2 = convolution_block(branch_2, 64, 3)
        branches = [branch0, branch1, branch_2]
    elif block_type == 'Block17':
        branch0 = convolution_block(x, 192, 1)
        branch1 = convolution_block(x, 128, 1)
        branch1 = convolution_block(branch1, 160, [1, 7])
        branch1 = convolution_block(branch1, 192, [7, 1])
        branches = [branch0, branch1]
    elif block_type == 'Block8':
        branch0 = convolution_block(x, 192, 1)
        branch1 = convolution_block(x, 192, 1)
        branch1 = convolution_block(branch1, 224, [1, 3])
        branch1 = convolution_block(branch1, 256, [3, 1])
        branches = [branch0, branch1]

    mixed = Concatenate(axis=channel_axis)(branches)
    up = convolution_block(mixed,
                           K.int_shape(x)[channel_axis],
                           1,
                           activation=None,
                           using_bias=True)
    x = Lambda(lambda inputs, scale: inputs[0] + inputs[1] * scale,
               output_shape=K.int_shape(x)[1:],
               arguments={'scale': scale}
               )([x, up])
    if activation is not None:
        x = Activation(activation)(x)
    return x


def InceptionResNetV2(input_shape=None):
    img_input = Input(shape=input_shape)
    x = convolution_block(img_input, 32, 3, strides=2, padding='valid')
    x = convolution_block(x, 32, 3, padding='valid')
    x = convolution_block(x, 64, 3)
    x = MaxPooling2D(3, strides=2)(x)
    x = convolution_block(x, 80, 1, padding='valid')
    x = convolution_block(x, 192, 3, padding='valid')
    x = MaxPooling2D(3, strides=2)(x)

    channel_axis = 1 if K.image_data_format() == 'channels_first' else 3
    name_fmt = partial(_generate_layer_name)
    branch_0 = convolution_block(x, 96, 1)
    branch_1 = convolution_block(x, 48, 1)
    branch_1 = convolution_block(branch_1, 64, 5)
    branch_2 = convolution_block(x, 64, 1)
    branch_2 = convolution_block(branch_2, 96, 3)
    branch_2 = convolution_block(branch_2, 96, 3)
    branch_pool = AveragePooling2D(3,
                                   strides=1,
                                   padding='same',
                                   name=name_fmt('AvgPool_0a_3x3', 3))(x)
    branch_pool = convolution_block(branch_pool, 64, 1)
    branches = [branch_0, branch_1, branch_2, branch_pool]
    x = Concatenate(axis=channel_axis)(branches)

    for block_idx in range(1, 11):
        x = _inception_resnet_block(x,
                                    scale=0.17,
                                    block_type='Block35',
                                    block_idx=block_idx)

    branch_0 = convolution_block(x,
                                 384,
                                 3,
                                 strides=2,
                                 padding='valid')
    branch_1 = convolution_block(x, 256, 1)
    branch_1 = convolution_block(branch_1, 256, 3)
    branch_1 = convolution_block(branch_1,
                                 384,
                                 3,
                                 strides=2,
                                 padding='valid')
    branch_pool = MaxPooling2D(3,
                               strides=2,
                               padding='valid')(x)
    branches = [branch_0, branch_1, branch_pool]
    x = Concatenate(axis=channel_axis)(branches)

    for block_idx in range(1, 21):
        x = _inception_resnet_block(x,
                                    scale=0.1,
                                    block_type='Block17',
                                    block_idx=block_idx)

    name_fmt = partial(_generate_layer_name)
    branch_0 = convolution_block(x, 256, 1)
    branch_0 = convolution_block(branch_0,
                                 384,
                                 3,
                                 strides=2,
                                 padding='valid')
    branch_1 = convolution_block(x, 256, 1)
    branch_1 = convolution_block(branch_1,
                                 288,
                                 3,
                                 strides=2,
                                 padding='valid')
    branch_2 = convolution_block(x, 256, 1)
    branch_2 = convolution_block(branch_2, 288, 3)
    branch_2 = convolution_block(branch_2,
                                 320,
                                 3,
                                 strides=2,
                                 padding='valid')
    branch_pool = MaxPooling2D(3,
                               strides=2,
                               padding='valid')(x)
    branches = [branch_0, branch_1, branch_2, branch_pool]
    x = Concatenate(axis=channel_axis)(branches)

    for block_idx in range(1, 10):
        x = _inception_resnet_block(x,
                                    scale=0.2,
                                    block_type='Block8',
                                    block_idx=block_idx)
    x = _inception_resnet_block(x,
                                scale=1.,
                                block_type='Block8',
                                block_idx=10)
    x = convolution_block(x, 1536, 1)
    x = GlobalAveragePooling2D()(x)

    inputs = img_input

    x = Dropout(0.4)(x)
    x = Dense(128, use_bias=False)(x)
    model = Model(inputs, x, name='inception_resnet_v2')
    return model
