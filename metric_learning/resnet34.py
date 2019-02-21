from keras.applications.resnet50 import ResNet50
from tensorflow.keras import Model
from tensorflow.keras.layers import Input
from tensorflow.python.keras.layers import Conv2D, MaxPool2D, Dense, BatchNormalization, Activation, \
    GlobalAveragePooling2D, Dropout, ZeroPadding2D, MaxPooling2D, Flatten
from tensorflow.python.keras.layers import add, AvgPool2D
from tensorflow.python.keras.regularizers import l2

from metric_learning.resnet_arch import ResNet18

default_drop = 0.3
default_kernel_size = 3


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


def identity_block(input_tensor, kernel_size, filters, stage, block, norm):
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
               kernel_regularizer=norm,
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
               kernel_regularizer=norm,
               name=conv_name_base + '2c')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    x = add([x, input_tensor])
    x = Activation('relu')(x)
    return x


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
        elif self.arch == 'test2':
            print('test2')
            return self.__test_model2()
        elif self.arch == 'test3':
            print('test3')
            return self.__test_model3()

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
        x = Dropout(default_drop)(x)

        x = conv_block(x, 3, [16, 16, 64], stage=2, block='a', strides=(1, 1), norm=self.kernel_regularization)
        x = identity_block(x, 3, [16, 16, 64], stage=2, block='b', norm=self.kernel_regularization)
        x = identity_block(x, 3, [16, 16, 64], stage=2, block='c', norm=self.kernel_regularization)
        x = MaxPooling2D((2, 2), strides=(2, 2))(x)
        x = Dropout(default_drop)(x)

        x = conv_block(x, 3, [16, 16, 64], stage=3, block='a', strides=(1, 1))
        x = identity_block(x, 3, [16, 16, 64], stage=3, block='b', norm=self.kernel_regularization)
        x = identity_block(x, 3, [16, 16, 64], stage=3, block='c', norm=self.kernel_regularization)
        x = MaxPooling2D((2, 2), strides=(2, 2))(x)
        x = Dropout(default_drop)(x)

        x = GlobalAveragePooling2D()(x)
        x = Dense(self.output_size, use_bias=False)(x)
        return Model(img_input, x)

    def __test_model2(self):
        img_input = Input(shape=(self.input_size, self.input_size, 3))
        x = Conv2D(64, 3, 3, activation='relu', name='conv1_1')(img_input)
        x = MaxPooling2D((2, 2), strides=(2, 2))(x)

        x = Conv2D(128, 3, 3, activation='relu', name='conv2_1')(x)
        x = MaxPooling2D((2, 2), strides=(2, 2))(x)

        x = Flatten()(x)
        x = Dense(4096, activation='relu', name='fc6')(x)
        x = Dropout(0.5)(x)
        x = Dense(4096, activation='relu', name='fc7')(x)
        return Model(img_input, x)

    def __test_model3(self):
        model = ResNet18((self.input_size, self.input_size, 3), 128, dropout=self.drop)
        x = Dense(self.output_size)(model.output)
        return Model(model.input, x)
