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

from sklearn.model_selection import train_test_split

def main():
    if not os.path.exists('Final Transactions.csv'):
        with zipfile.ZipFile("dataset.zip", "r") as zip_ref:
            zip_ref.extractall()

    orgDf = pd.read_csv('Final Transactions.csv')
    df = orgDf.sample(1000000)

    data = df.get(['CUSTOMER_ID', 'TERMINAL_ID', 'TX_AMOUNT'])
    labels = df.get('TX_FRAUD')

    x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2)

    model = Sequential([
        Dense(3, input_shape=(3,), use_bias=True, ),
        Dense(16, use_bias=True, activation="relu"),
        Dense(1, use_bias=True, activation="sigmoid")
    ])

    model.compile(loss='binary_crossentropy', optimizer="adam", metrics=['accuracy'])

    model.summary()

    ep = 50
    history = model.fit(x=x_train, y=y_train, batch_size=1000, epochs=ep, validation_split=0.2)

    model.evaluate(x_test, y_test)

    acc = history.history['accuracy']
    loss = history.history['loss']
    epochs = range(1, len(acc) + 1)
    plt.plot(epochs, acc, label='Training accuracy')
    acc = history.history['val_accuracy']
    plt.plot(epochs, acc, label='Validation accuracy')
    plt.title('Training accuracy')
    plt.legend()

    plt.figure()
    plt.plot(epochs, loss, label='Training loss')
    loss = history.history['val_loss']
    plt.plot(epochs, loss, label='Validation accuracy')
    plt.title('Training loss')
    plt.legend()
    plt.show()



if __name__ == '__main__':
    main()
