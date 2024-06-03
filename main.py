import keras
import tensorflow as tf
import numpy as np
import pandas as pd
import zipfile
import os

from keras import Sequential
from keras.src.layers import Dense, Dropout
from keras.src.regularizers import L2
from matplotlib import pyplot as plt


def main():
    if not os.path.exists('Final Transactions.csv'):
        with zipfile.ZipFile("dataset.zip", "r") as zip_ref:
            zip_ref.extractall()

    orgDf = pd.read_csv('Final Transactions.csv')
    df = orgDf.sample(1000000)

    data = df.get(['CUSTOMER_ID', 'TERMINAL_ID', 'TX_AMOUNT', 'TX_TIME_SECONDS', 'TX_TIME_DAYS'])
    labels = df.get('TX_FRAUD')
    one_hot_labels = pd.get_dummies(labels)

    data = np.asarray(data)
    one_hot_labels = np.asarray(one_hot_labels)

    model = Sequential()
    model.add(Dense(units=32, use_bias=True, input_shape=(5,), activation="relu"))
    model.add(Dropout(0.4))
    model.add(Dense(units=16, use_bias=True, kernel_regularizer=L2(0.01), activation="relu"))
    model.add(Dropout(0.4))
    model.add(Dense(units=16, use_bias=True, kernel_regularizer=L2(0.01), activation="relu"))
    model.add(Dropout(0.4))
    model.add(Dense(units=5, use_bias=True, activation="relu"))
    model.add(Dropout(0.4))
    model.add(Dense(units=7, use_bias=True, activation="relu"))
    model.add(Dropout(0.4))
    model.add(Dense(units=10, use_bias=True, activation="relu"))
    model.add(Dropout(0.4))
    model.add(Dense(units=1, use_bias=True, activation="relu"))
    model.add(Dropout(0.4))
    model.add(Dense(units=2, use_bias=True, activation="softmax"))
    opt = keras.optimizers.SGD(learning_rate=0.01)

    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    i = 0
    model.summary()

    ep = 50
    history = model.fit(data, one_hot_labels, batch_size=1000, epochs=ep, validation_split=0.3, shuffle=True)

    i += ep
    acc = history.history['accuracy']
    loss = history.history['loss']
    epochs = range(1, len(acc) + 1)
    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.title('Training accuracy')
    plt.legend()
    plt.figure()
    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.title('Training loss')
    plt.legend()
    plt.show()

    acc = history.history['val_accuracy']
    loss = history.history['val_loss']
    epochs = range(1, len(acc) + 1)
    plt.plot(epochs, acc, 'bo', label='val acc')
    plt.title('val accuracy')
    plt.legend()
    plt.figure()
    plt.plot(epochs, loss, 'bo', label='val loss')
    plt.title('val loss')
    plt.legend()
    plt.show()

    print(i)


if __name__ == '__main__':
    main()
