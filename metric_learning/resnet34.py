from keras.applications.resnet50 import ResNet50
from keras.regularizers import l2
from tensorflow.keras import Model
from tensorflow.keras.layers import Input
from tensorflow.python.keras.layers import Conv2D, MaxPool2D, Dense, BatchNormalization, Activation, \
    GlobalAveragePooling2D, ZeroPadding2D, MaxPooling2D, AveragePooling2D, Flatten
from tensorflow.python.keras.layers import add, AvgPool2D

from metric_learning.resnet_arch import ResNet18

default_drop = 0.05
default_kernel_size = 3


def resnet_layer(inputs,
                 num_filters=16,
                 kernel_size=3,
                 strides=1,
                 activation='relu',
                 batch_normalization=True,
                 conv_first=True):
    """2D Convolution-Batch Normalization-Activation stack builder

    # Arguments
        inputs (tensor): input tensor from input image or previous layer
        num_filters (int): Conv2D number of filters
        kernel_size (int): Conv2D square kernel dimensions
        strides (int): Conv2D square stride dimensions
        activation (string): activation name
        batch_normalization (bool): whether to include batch normalization
        conv_first (bool): conv-bn-activation (True) or
            bn-activation-conv (False)

    # Returns
        x (tensor): tensor as input to the next layer
    """
    conv = Conv2D(num_filters,
                  kernel_size=kernel_size,
                  strides=strides,
                  padding='same',
                  kernel_initializer='he_normal',
                  kernel_regularizer=l2(1e-4))

    x = inputs
    if conv_first:
        x = conv(x)
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
    else:
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
        x = conv(x)
    return x


def resnet_v2(input_shape, depth, out_size=128):
    if (depth - 2) % 9 != 0:
        raise ValueError('depth should be 9n+2 (eg 56 or 110 in [b])')
    num_filters_in = 16
    num_res_blocks = int((depth - 2) / 9)

    inputs = Input(shape=input_shape)
    x = resnet_layer(inputs=inputs,
                     num_filters=num_filters_in,
                     conv_first=True)

    for stage in range(3):
        for res_block in range(num_res_blocks):
            activation = 'relu'
            batch_normalization = True
            strides = 1
            if stage == 0:
                num_filters_out = num_filters_in * 4
                if res_block == 0:
                    activation = None
                    batch_normalization = False
            else:
                num_filters_out = num_filters_in * 2
                if res_block == 0:
                    strides = 2

            y = resnet_layer(inputs=x,
                             num_filters=num_filters_in,
                             kernel_size=1,
                             strides=strides,
                             activation=activation,
                             batch_normalization=batch_normalization,
                             conv_first=False)
            y = resnet_layer(inputs=y,
                             num_filters=num_filters_in,
                             conv_first=False)
            y = resnet_layer(inputs=y,
                             num_filters=num_filters_out,
                             kernel_size=1,
                             conv_first=False)
            if res_block == 0:
                x = resnet_layer(inputs=x,
                                 num_filters=num_filters_out,
                                 kernel_size=1,
                                 strides=strides,
                                 activation=None,
                                 batch_normalization=False)
            x = add([x, y])

        num_filters_in = num_filters_out

    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = AveragePooling2D(pool_size=8)(x)
    y = Flatten()(x)
    outputs = Dense(out_size)(y)

    model = Model(inputs=inputs, outputs=outputs)
    return model


def conv_block(input_tensor,
               kernel_size,
               filters,
               stage,
               block,
               strides=(2, 2),
               norm=None):
    """A block that has a conv layer at shortcut.

    # Arguments
        input_tensor: input tensor
        kernel_size: default 3, the kernel size of
            middle conv layer at main path
        filters: list of integers, the filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
        strides: Strides for the first conv layer in the block.

    # Returns
        Output tensor for the block.

    Note that from stage 3,
    the first conv layer at main path is with strides=(2, 2)
    And the shortcut should have strides=(2, 2) as well
    """
    filters1, filters2, filters3 = filters
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    bn_axis = -1

    x = Conv2D(filters1, (1, 1), strides=strides,
               kernel_initializer='he_normal',
               kernel_regularizer=norm,
               name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters2, kernel_size, padding='same',
               kernel_initializer='he_normal',
               kernel_regularizer=norm,
               name=conv_name_base + '2b')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters3, (1, 1),
               kernel_initializer='he_normal',
               name=conv_name_base + '2c')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    shortcut = Conv2D(filters3, (1, 1), strides=strides,
                      kernel_initializer='he_normal',
                      kernel_regularizer=norm,
                      name=conv_name_base + '1')(input_tensor)
    shortcut = BatchNormalization(
        axis=bn_axis, name=bn_name_base + '1')(shortcut)

    x = add([x, shortcut])
    x = Activation('relu')(x)
    return x


def identity_block(input_tensor, kernel_size, filters, stage, block):
    """The identity block is the block that has no conv layer at shortcut.

    # Arguments
        input_tensor: input tensor
        kernel_size: default 3, the kernel size of
            middle conv layer at main path
        filters: list of integers, the filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names

    # Returns
        Output tensor for the block.
    """
    filters1, filters2, filters3 = filters
    bn_axis = -1
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Conv2D(filters1, (1, 1),
               kernel_initializer='he_normal',
               name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters2, kernel_size,
               padding='same',
               kernel_initializer='he_normal',
               name=conv_name_base + '2b')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters3, (1, 1),
               kernel_initializer='he_normal',
               name=conv_name_base + '2c')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    x = add([x, input_tensor])
    x = Activation('relu')(x)
    return x


class Resnet34:
    def __init__(self, input_size, output_size, drop=0., arch='resnet'):
        self.input_size = input_size or 128
        self.output_size = output_size
        self.arch = arch
        self.drop = drop

    def conv_block(self, feat_maps_out, prev, strides):
        prev = Conv2D(feat_maps_out, (3, 3), strides=strides, padding='same', kernel_initializer='he_normal')(prev)
        prev = BatchNormalization()(prev)  # Specifying the axis and mode allows for later merging
        prev = Activation('relu')(prev)
        prev = Conv2D(feat_maps_out, (3, 3), padding='same')(prev)
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
            prev = self.level3(prev)
            prev = self.level2(prev)
            prev = self.level1(prev)
            prev = self.level0(prev)
            prev = GlobalAveragePooling2D()(prev)
            output = Dense(self.output_size)(prev)
            return Model(image_input, output)
        elif self.arch == 'test':
            print('test')
            return self.__test_model()
        elif self.arch == 'test3':
            print('test3')
            return self.__test_model3()
        elif self.arch == 'resnet20':
            print('resnet20')
            return self.__resnet20()

    def __test_model(self):
        img_input = Input(shape=(self.input_size, self.input_size, 3))
        x = ZeroPadding2D(padding=(3, 3), name='conv1_pad')(img_input)
        x = Conv2D(32, (5, 5),
                   strides=(2, 2),
                   padding='valid',
                   kernel_initializer='he_normal',
                   name='conv1')(x)
        x = BatchNormalization(name='bn_conv1')(x)
        x = Activation('relu')(x)
        x = ZeroPadding2D(padding=(1, 1), name='pool1_pad')(x)
        x = MaxPooling2D((3, 3), strides=(2, 2))(x)

        x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1))
        x = identity_block(x, 3, [64, 64, 256], stage=2, block='b')
        x = identity_block(x, 3, [64, 64, 256], stage=2, block='c')

        x = conv_block(x, 3, [128, 128, 512], stage=3, block='a')
        x = identity_block(x, 3, [128, 128, 512], stage=3, block='b')
        x = identity_block(x, 3, [128, 128, 512], stage=3, block='c')
        x = identity_block(x, 3, [128, 128, 512], stage=3, block='d')

        x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a')
        x = identity_block(x, 3, [256, 256, 1024], stage=4, block='b')
        x = identity_block(x, 3, [256, 256, 1024], stage=4, block='c')
        x = identity_block(x, 3, [256, 256, 1024], stage=4, block='d')
        x = identity_block(x, 3, [256, 256, 1024], stage=4, block='e')
        x = identity_block(x, 3, [256, 256, 1024], stage=4, block='f')

        x = conv_block(x, 3, [512, 512, 2048], stage=5, block='a')
        x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b')
        x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c')

        x = GlobalAveragePooling2D()(x)
        x = Dense(self.output_size)(x)
        return Model(img_input, x)

    def __test_model3(self):
        model = ResNet18((self.input_size, self.input_size, 3), 128, dropout=self.drop)
        x = Dense(self.output_size)(model.output)
        return Model(model.input, x)

    def __resnet20(self):
        model = resnet_v2((self.input_size, self.input_size, 3), 20, self.output_size)
        return model
