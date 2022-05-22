from tensorflow import keras
from tensorflow.keras.models import Model,Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, AveragePooling2D, ZeroPadding2D
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization, add, Input
from tensorflow.keras import regularizers
import os
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import ModelCheckpoint
import utils

seed_value = 0

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

acc_dir = r'accuracy_resnet50.txt'
csv_dir = r'Normal_all.csv'
x_train, x_test, y_train, y_test = utils.load_handle_data(csv_dir)
x_train = x_train.values.reshape(x_train.shape[0], 985, 1, 1).astype('float32')
x_test = x_test.values.reshape(x_test.shape[0], 985, 1, 1).astype('float32')
y_train = np.array(y_train)
y_test = np.array(y_test)



def Conv2d_BN(x, nb_filter, kernel_size, strides=(1, 1), padding='same', name=None):
    if name is not None:
        bn_name = name + '_bn'
        conv_name = name + '_conv'
    else:
        bn_name = None
        conv_name = None

    x = Conv2D(nb_filter, kernel_size, padding=padding, strides=strides, name=conv_name)(x)
    x = BatchNormalization(axis=3, name=bn_name)(x)
    x = Activation('relu')(x)
    return x


def Conv_Block(inpt, nb_filter, kernel_size, strides=(1, 1), with_conv_shortcut=False):
    x = Conv2d_BN(inpt, nb_filter=nb_filter[0], kernel_size=(1, 1), strides=strides, padding='same')
    x = Conv2d_BN(x, nb_filter=nb_filter[1], kernel_size=(3, 3), padding='same')
    x = Conv2d_BN(x, nb_filter=nb_filter[2], kernel_size=(1, 1), padding='same')
    if with_conv_shortcut:
        shortcut = Conv2d_BN(inpt, nb_filter=nb_filter[2], strides=strides, kernel_size=kernel_size)
        x = add([x, shortcut])
        return x
    else:
        x = add([x, inpt])
        return x


def creatcnn():
    inpt = Input(shape=(985, 1, 1))
    x = ZeroPadding2D((3, 3))(inpt)
    x = Conv2d_BN(x, nb_filter=64, kernel_size=(7, 7), strides=2, padding='valid')
    x = MaxPooling2D(pool_size=(3, 1), strides=(2, 1), padding='same')(x)

    x = Conv_Block(x, nb_filter=[64, 64, 256], kernel_size=(3, 3), strides=(1, 1), with_conv_shortcut=True)
    x = Conv_Block(x, nb_filter=[64, 64, 256], kernel_size=(3, 3))
    x = Conv_Block(x, nb_filter=[64, 64, 256], kernel_size=(3, 3))

    x = Conv_Block(x, nb_filter=[128, 128, 512], kernel_size=(3, 3), strides=(2, 2), with_conv_shortcut=True)
    x = Conv_Block(x, nb_filter=[128, 128, 512], kernel_size=(3, 3))
    x = Conv_Block(x, nb_filter=[128, 128, 512], kernel_size=(3, 3))
    x = Conv_Block(x, nb_filter=[128, 128, 512], kernel_size=(3, 3))

    x = Conv_Block(x, nb_filter=[256, 256, 1024], kernel_size=(3, 3), strides=(2, 2), with_conv_shortcut=True)
    x = Conv_Block(x, nb_filter=[256, 256, 1024], kernel_size=(3, 3))
    x = Conv_Block(x, nb_filter=[256, 256, 1024], kernel_size=(3, 3))
    x = Conv_Block(x, nb_filter=[256, 256, 1024], kernel_size=(3, 3))
    x = Conv_Block(x, nb_filter=[256, 256, 1024], kernel_size=(3, 3))
    x = Conv_Block(x, nb_filter=[256, 256, 1024], kernel_size=(3, 3))

    x = Conv_Block(x, nb_filter=[512, 512, 2048], kernel_size=(3, 3), strides=(2, 2), with_conv_shortcut=True)
    x = Conv_Block(x, nb_filter=[512, 512, 2048], kernel_size=(3, 3))
    x = Conv_Block(x, nb_filter=[512, 512, 2048], kernel_size=(3, 3))
    x = AveragePooling2D(pool_size=(7, 1))(x)
    x = Flatten()(x)
    x = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=inpt, outputs=x)
    return model

if __name__ == '__main__':

    model = creatcnn()
    print(model.summary())
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['binary_accuracy'])
    checkpoint = ModelCheckpoint('resnet50_new.h5', monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    callbacks_list = [checkpoint]
    history = model.fit(x_train, y_train, batch_size=32, epochs=800, validation_split=0.2, callbacks=callbacks_list, verbose=1, shuffle=True)
    print("H.histroy keys£º", history.history.keys())
    #plot
    acc = history.history['binary_accuracy']  # »ñÈ¡ÑµÁ·¼¯×¼È·ÐÔÊý¾Ý
    val_acc = history.history['val_binary_accuracy']  # »ñÈ¡ÑéÖ¤¼¯×¼È·ÐÔÊý¾Ý
    loss = history.history['loss']  # »ñÈ¡ÑµÁ·¼¯´íÎóÖµÊý¾Ý
    val_loss = history.history['val_loss']  # »ñÈ¡ÑéÖ¤¼¯´íÎóÖµÊý¾Ý
    epochs = range(1, len(acc) + 1)
    with open(acc_dir, 'a', newline='', encoding='utf-8') as f:
        for i in range(len(acc)):
            f.write('val loss='+str(val_loss[i])+','+'val accuracy='+str(val_acc[i]))
            f.write('\n')
    plt.plot(epochs, acc, 'r', label='Trainning acc')  # ÒÔepochsÎªºá×ø±ê£¬ÒÔÑµÁ·¼¯×¼È·ÐÔÎª×Ý×ø±ê
    plt.plot(epochs, val_acc, 'b', label='Vaildation acc')  # ÒÔepochsÎªºá×ø±ê£¬ÒÔÑéÖ¤¼¯×¼È·ÐÔÎª×Ý×ø±ê
    plt.legend()  # »æÖÆÍ¼Àý£¬¼´±êÃ÷Í¼ÖÐµÄÏß¶Î´ú±íºÎÖÖº¬Òå
    plt.show()
    plt.plot(epochs, loss, 'r', label='Trainning loss')  # ÒÔepochsÎªºá×ø±ê£¬ÒÔÑµÁ·¼¯×¼È·ÐÔÎª×Ý×ø±ê
    plt.plot(epochs, val_loss, 'b', label='Vaildation loss')  # ÒÔepochsÎªºá×ø±ê£¬ÒÔÑéÖ¤¼¯×¼È·ÐÔÎª×Ý×ø±ê
    plt.legend()  # »æÖÆÍ¼Àý£¬¼´±êÃ÷Í¼ÖÐµÄÏß¶Î´ú±íºÎÖÖº¬Òå
    plt.show()
