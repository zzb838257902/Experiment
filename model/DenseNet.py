import numpy as np
from tensorflow import keras
from tensorflow.keras.models import Model, save_model, load_model
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization, LeakyReLU, concatenate, Activation
from tensorflow.keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, GlobalAveragePooling2D
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import GaussianNoise
from tensorflow.keras import backend as K
from tensorflow.keras import regularizers
import matplotlib.pyplot as plt
import utils

seed_value = 1

# 1. Set `PYTHONHASHSEED` environment variable at a fixed value
import os

os.environ['PYTHONHASHSEED'] = str(seed_value)

# 2. Set `python` built-in pseudo-random generator at a fixed value
import random

random.seed(seed_value)

# 3. Set `numpy` pseudo-random generator at a fixed value
import numpy as np

np.random.seed(seed_value)

# 4. Set `tensorflow` pseudo-random generator at a fixed value
import tensorflow as tf

tf.compat.v1.set_random_seed(seed_value)
# data
acc_dir = r'accuracy_denseNet121.txt'
csv_dir = r'Normal_all.csv'

x_train, x_test, y_train, y_test = utils.load_handle_data(csv_dir)
x_train = x_train.values.reshape(x_train.shape[0], 985, 1, 1).astype('float32')
x_test = x_test.values.reshape(x_test.shape[0], 985, 1, 1).astype('float32')
y_train = np.array(y_train)
y_test = np.array(y_test)


def conv_block(ip, nb_filter, bottleneck=False, dropout_rate=None, weight_decay=1e-4):
    ''' Apply BatchNorm, Relu, 3x3 Conv2D, optional bottleneck block and dropout
        Args:
            ip: Input keras tensor
            nb_filter: number of filters
            bottleneck: add bottleneck block
            dropout_rate: dropout rate
            weight_decay: weight decay factor
        Returns: keras tensor with batch_norm, relu and convolution2d added (optional bottleneck)
    '''
    concat_axis = 1 if K.image_data_format() == 'channel_first' else -1

    # x = BatchNormalization(axis=concat_axis, epsilon=1.1e-5)(ip)
    x = Activation('relu')(ip)

    if bottleneck:
        inter_channel = nb_filter * 4
        x = Conv2D(inter_channel, (1, 1), kernel_initializer='glorot_uniform', padding='same', use_bias=False,
                   kernel_regularizer=regularizers.l2(weight_decay))(x)
        # x = BatchNormalization(axis=concat_axis, epsilon=1.1e-5)(x)
        x = Activation('relu')(x)

    x = Conv2D(nb_filter, (3, 3), kernel_initializer='glorot_uniform', padding='same', use_bias=False)(x)

    if dropout_rate:
        x = Dropout(dropout_rate)(x)

    return x


def transition_block(ip, nb_filter, compression=1.0, weight_decay=1e-4):
    '''Apply BatchNorm, ReLU, Conv2d, optional compressoin, dropout and Maxpooling2D
        Args:
            ip: keras tensor
            nb_filter: number of filters
            compression: caculated as 1 - reduction. Reduces the number of features maps in the transition block
            dropout_rate: dropout rate
            weight_decay: weight decay factor
        Returns:
            keras tensor, after applying batch_norm, relu-conv, dropout, maxpool
    '''
    concat_axis = 1 if K.image_data_format() == 'channels_first' else -1

    # x = BatchNormalization(axis=concat_axis, epsilon=1.1e-5)(ip)
    x = Activation('relu')(ip)
    x = Conv2D(int(nb_filter * compression), (1, 1), kernel_initializer='glorot_uniform', padding='same', use_bias=False,
               kernel_regularizer=regularizers.l2(weight_decay))(x)
    x = MaxPooling2D((2, 1), strides=2)(x)

    return x


def dense_block(x, nb_layers, nb_filter, growth_rate, bottleneck=False, dropout_rate=None, weight_decay=1e-4,
                grow_nb_filters=True, return_concat_list=False):
    '''Build a dense_block where the output of ench conv_block is fed t subsequent ones
        Args:
            x: keras tensor
            nb_layser: the number of layers of conv_block to append to the model
            nb_filter: number of filters
            growth_rate: growth rate
            bottleneck: bottleneck block
            dropout_rate: dropout rate
            weight_decay: weight decay factor
            grow_nb_filters: flag to decide to allow number of filters to grow
            return_concat_list: return the list of feature maps along with the actual output
        Returns:
            keras tensor with nb_layers of conv_block appened
    '''

    concat_axis = 1 if K.image_data_format() == 'channels_first' else -1

    x_list = [x]

    for i in range(nb_layers):
        cb = conv_block(x, growth_rate, bottleneck, dropout_rate, weight_decay)
        x_list.append(cb)
        x = concatenate([x, cb], axis=concat_axis)

        if grow_nb_filters:
            nb_filter += growth_rate

    if return_concat_list:
        return x, nb_filter, x_list
    else:
        return x, nb_filter


def create_dense_net(nb_classes, img_input, include_top, depth=40, nb_dense_block=3, growth_rate=12, nb_filter=-1,
                     nb_layers_per_block=[1], bottleneck=False, reduction=0.0, dropout_rate=None, weight_decay=1e-4,
                     subsample_initial_block=False, activation='softmax'):
    ''' Build the DenseNet model
        Args:
            nb_classes: number of classes
            img_input: tuple of shape (channels, rows, columns) or (rows, columns, channels)
            include_top: flag to include the final Dense layer
            depth: number or layers
            nb_dense_block: number of dense blocks to add to end (generally = 3)
            growth_rate: number of filters to add per dense block
            nb_filter: initial number of filters. Default -1 indicates initial number of filters is 2 * growth_rate
            nb_layers_per_block: list, number of layers in each dense block
            bottleneck: add bottleneck blocks
            reduction: reduction factor of transition blocks. Note : reduction value is inverted to compute compression
            dropout_rate: dropout rate
            weight_decay: weight decay rate
            subsample_initial_block: Set to True to subsample the initial convolution and
                    add a MaxPool2D before the dense blocks are added.
            subsample_initial:
            activation: Type of activation at the top layer. Can be one of 'softmax' or 'sigmoid'.
                    Note that if sigmoid is used, classes must be 1.
        Returns: keras tensor with nb_layers of conv_block appended
    '''

    concat_axis = 1 if K.image_data_format() == 'channel_first' else -1

    if type(nb_layers_per_block) is not list:
        print('nb_layers_per_block should be a list!!!')
        return 0

    final_nb_layer = nb_layers_per_block[-1]
    nb_layers = nb_layers_per_block[:-1]

    if nb_filter <= 0:
        nb_filter = 2 * growth_rate
    compression = 1.0 - reduction
    if subsample_initial_block:
        initial_kernel = (7, 7)
        initial_strides = 1
    else:
        initial_kernel = (3, 3)
        initial_strides = 1

    x = Conv2D(nb_filter, initial_kernel, kernel_initializer='glorot_uniform', padding='same',
               strides=initial_strides, use_bias=False, kernel_regularizer=regularizers.l2(weight_decay))(img_input)
    if subsample_initial_block:
        # x = BatchNormalization(axis=concat_axis, epsilon=1.1e-5)(x)
        x = Activation('relu')(x)
        x = MaxPooling2D((3, 1), strides=1, padding='same')(x)

    for block_index in range(nb_dense_block - 1):
        x, nb_filter = dense_block(x, nb_layers[block_index], nb_filter, growth_rate, bottleneck=bottleneck,
                                   dropout_rate=dropout_rate, weight_decay=weight_decay)
        x = transition_block(x, nb_filter, compression=compression, weight_decay=weight_decay)
        nb_filter = int(nb_filter * compression)

    # ×îºóÒ»¸öblockÃ»ÓÐtransition_block
    x, nb_filter = dense_block(x, final_nb_layer, nb_filter, growth_rate, bottleneck=bottleneck,
                               dropout_rate=dropout_rate, weight_decay=weight_decay)

    # x = BatchNormalization(axis=concat_axis, epsilon=1.1e-5)(x)
    x = Activation('relu')(x)
    x = GlobalAveragePooling2D()(x)

    if include_top:
        x = Dense(nb_classes, activation='sigmoid')(x)

    return x


def binary_focal_loss(gamma=2, alpha=0.25):
    """
    Binary form of focal loss.
    ÊÊÓÃÓÚ¶þ·ÖÀàÎÊÌâµÄfocal loss

    focal_loss(p_t) = -alpha_t * (1 - p_t)**gamma * log(p_t)
        where p = sigmoid(x), p_t = p or 1 - p depending on if the label is 1 or 0, respectively.
    References:
        https://arxiv.org/pdf/1708.02002.pdf
    Usage:
     model.compile(loss=[binary_focal_loss(alpha=.25, gamma=2)], metrics=["accuracy"], optimizer=adam)
    """
    alpha = tf.constant(alpha, dtype=tf.float32)
    gamma = tf.constant(gamma, dtype=tf.float32)

    def binary_focal_loss_fixed(y_true, y_pred):
        """
        y_true shape need be (None,1)
        y_pred need be compute after sigmoid
        """
        y_true = tf.cast(y_true, tf.float32)
        alpha_t = y_true * alpha + (K.ones_like(y_true) - y_true) * (1 - alpha)

        p_t = y_true * y_pred + (K.ones_like(y_true) - y_true) * (K.ones_like(y_true) - y_pred) + K.epsilon()
        focal_loss = - alpha_t * K.pow((K.ones_like(y_true) - p_t), gamma) * K.log(p_t)
        return K.mean(focal_loss)
    return binary_focal_loss_fixed


input_shape = (985, 1, 1)
inputs = Input(shape=input_shape)
x = create_dense_net(nb_classes=1, img_input=inputs, include_top=True, depth=121, nb_dense_block=4,
                     growth_rate=32, nb_filter=64, nb_layers_per_block=[6, 12, 24, 16], bottleneck=True, reduction=0.5,
                     dropout_rate=0.3, weight_decay=1e-5, subsample_initial_block=True, activation='softmax')
model = Model(inputs, x, name='densenet121')

print(model.summary())
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['binary_accuracy'])
# checkpoint = ModelCheckpoint('DenseNet121.h5', monitor='val_loss', verbose=1, save_best_only=True, mode='min')
# callbacks_list = [checkpoint]
model.summary()
# history = model.fit(x_train, y_train, batch_size=32, epochs=800, validation_split=0.2, callbacks=callbacks_list, verbose=1, shuffle=True)
#model.save('DenseNet121.h5')
# print("H.histroy keys", history.history.keys())
#plot
# acc = history.history['binary_accuracy']  # »ñÈ¡ÑµÁ·¼¯×¼È·ÐÔÊý¾Ý
#val_acc = history.history['val_binary_accuracy']  # »ñÈ¡ÑéÖ¤¼¯×¼È·ÐÔÊý¾Ý
# loss = history.history['loss']  # »ñÈ¡ÑµÁ·¼¯´íÎóÖµÊý¾Ý
#val_loss = history.history['val_loss']  # »ñÈ¡ÑéÖ¤¼¯´íÎóÖµÊý¾Ý
# epochs = range(1, len(acc) + 1)
# with open(acc_dir, 'a', newline='', encoding='utf-8') as f:
#     for i in range(len(acc)):
#         f.write('val loss='+str(val_loss[i])+','+'val accuracy='+str(val_acc[i]))
#         f.write('\n')
# plt.plot(epochs, acc, 'r', label='Trainning acc')  # ÒÔepochsÎªºá×ø±ê£¬ÒÔÑµÁ·¼¯×¼È·ÐÔÎª×Ý×ø±ê
# plt.plot(epochs, val_acc, 'b', label='Vaildation acc')  # ÒÔepochsÎªºá×ø±ê£¬ÒÔÑéÖ¤¼¯×¼È·ÐÔÎª×Ý×ø±ê
# plt.legend()  # »æÖÆÍ¼Àý£¬¼´±êÃ÷Í¼ÖÐµÄÏß¶Î´ú±íºÎÖÖº¬Òå
# plt.show()
# plt.plot(epochs, loss, 'r', label='Trainning loss')  # ÒÔepochsÎªºá×ø±ê£¬ÒÔÑµÁ·¼¯×¼È·ÐÔÎª×Ý×ø±ê
# plt.plot(epochs, val_loss, 'b', label='Vaildation loss')  # ÒÔepochsÎªºá×ø±ê£¬ÒÔÑéÖ¤¼¯×¼È·ÐÔÎª×Ý×ø±ê
# plt.legend()  # »æÖÆÍ¼Àý£¬¼´±êÃ÷Í¼ÖÐµÄÏß¶Î´ú±íºÎÖÖº¬Òå
# plt.show()
for ii in range(1600):
    print("Epoch:", ii + 1)
    model.fit(x_train, y_train, batch_size=32, epochs=1, verbose=1, shuffle=True)
    score = model.evaluate(x_test, y_test, verbose=1)
    print('Test loss =', score[0])
    print('Test accuracy =', score[1])
    with open(acc_dir, 'a', newline='', encoding='utf-8') as f:
        f.write('Test loss='+str(score[0])+','+'Test accuracy='+str(score[1]))
        f.write('\n')
