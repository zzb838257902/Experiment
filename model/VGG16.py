from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from tensorflow.keras.optimizers import SGD
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

acc_dir = r'accuracy_VGG16.txt'
csv_dir = r'Normal_all.csv'
x_train, x_test, y_train, y_test = utils.load_handle_data(csv_dir)
x_train = x_train.values.reshape(x_train.shape[0], 985, 1, 1).astype('float32')
x_test = x_test.values.reshape(x_test.shape[0], 985, 1, 1).astype('float32')
y_train = np.array(y_train)
y_test = np.array(y_test)



def vgg16():
    weight_decay = 0.0005
    nb_epoch = 100
    batch_size = 32

    # layer1
    model = Sequential()
    model.add(Conv2D(64, (3, 3), padding='same',
                     input_shape=(985, 1, 1), kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    # layer2
    model.add(Conv2D(64, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 1)))
    # layer3
    model.add(Conv2D(128, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    # layer4
    model.add(Conv2D(128, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 1)))
    # layer5
    model.add(Conv2D(256, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    # layer6
    model.add(Conv2D(256, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    # layer7
    model.add(Conv2D(256, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 1)))
    # layer8
    model.add(Conv2D(512, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    # layer9
    model.add(Conv2D(512, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    # layer10
    model.add(Conv2D(512, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 1)))
    # layer11
    model.add(Conv2D(512, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    # layer12
    model.add(Conv2D(512, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    # layer13
    model.add(Conv2D(512, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 1)))
    model.add(Dropout(0.2))
    # layer14
    model.add(Flatten())
    model.add(Dense(512, kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    # layer15
    model.add(Dense(512, kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    # layer16
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    return model

if __name__ == '__main__':

    model = vgg16()
    print(model.summary())
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['binary_accuracy'])
    checkpoint = ModelCheckpoint('vgg16.h5', monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    callbacks_list = [checkpoint]
    history = model.fit(x_train, y_train, batch_size=32, epochs=800, validation_split=0.2, callbacks=callbacks_list, verbose=1, shuffle=True)
    model.save('vgg16.h5')
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
