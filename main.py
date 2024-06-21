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
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


def convert_year_seconds(time_in_seconds):
    seconds_from_midnight = time_in_seconds % (24 * 60 * 60)
    minutes_from_midnight = seconds_from_midnight / 60
    return int(minutes_from_midnight)


def get_week_day(datetime):
    date, _ = datetime.split()
    wd = pd.Timestamp(date)
    return wd.dayofweek


def main():
    if not os.path.exists('Final Transactions.csv'):
        with zipfile.ZipFile("dataset.zip", "r") as zip_ref:
            zip_ref.extractall()

    orgDf = pd.read_csv('Final Transactions.csv')
    df = orgDf.sample(1000000)

    data = df.get(['CUSTOMER_ID', 'TERMINAL_ID', 'TX_AMOUNT'])
    data['TX_TIME_MINUTES'] = df['TX_TIME_SECONDS'].apply(convert_year_seconds)

    data['TX_WEEK_DAY'] = df['TX_DATETIME'].apply(get_week_day)
    labels = df.get('TX_FRAUD')

    x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2)

    model = Sequential([
        Dense(data.shape[1], input_shape=(data.shape[1],), use_bias=True, ),
        Dense(16, use_bias=True, activation="relu"),
        Dense(8, use_bias=True, activation="relu"),
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

    seq_predictions = model.predict(x_test)
    seq_predictions = np.transpose(seq_predictions)[0]  # transformation to get (n,)
    # Applying transformation to get binary values predictions with 0.5 as thresold
    seq_predictions = list(map(lambda x: 0 if x < 0.5 else 1, seq_predictions))

    cm = confusion_matrix(y_test, seq_predictions)
    cmd = ConfusionMatrixDisplay(cm, display_labels=['Real', 'Fraud'])
    cmd.plot()
    plt.show()

    cm = confusion_matrix(y_test, seq_predictions, normalize='all')
    cmd = ConfusionMatrixDisplay(cm, display_labels=['Real', 'Fraud'])
    cmd.plot()
    plt.show()


if __name__ == '__main__':
    main()
