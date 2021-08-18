import pandas as pd
import numpy as np
import os
import random

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

import tensorflow as tf

from tensorflow.keras import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense, Input, Bidirectional
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import load_model

from sklearn.preprocessing import MinMaxScaler

import matplotlib.pyplot as plt

INPUT_DATA_PATH = "data/processed/data_imu_polar_200_500.csv"

EPOCHS = 120
BATCH_SIZE = 64
N_FEATURES = 9
N_UNITS_LSTM = 128
N_UNITS_DENSE_1 = 128
N_UNITS_DENSE_2 = 256
LEARNING_RATE = 0.0001
N_OUTPUTS = 2
WEIGHT = 1
DATA = "200:500"
DROPOUT = 0
BIDIR = 0


random.seed(a=None, version=2)
ID_TRAIN = random.randint(0, 100000)


def build_model():
    model = Sequential()
    model.add(Input(shape=(None, N_FEATURES)))
    model.add(Dense(N_UNITS_DENSE_1))
    model.add(LSTM(N_UNITS_LSTM, return_sequences=True))
    model.add(LSTM(N_UNITS_LSTM))
    model.add(Dense(N_UNITS_DENSE_2))
    # model.add(Dropout(DROPOUT))
    model.add(Dense(N_OUTPUTS))

    return model


def train(model, optimizer, X_train, y_train, X_test, y_test, es, mc):
    model.compile(optimizer=optimizer, loss="mean_squared_error", loss_weights=[1, WEIGHT])

    history = model.fit(
        X_train,
        y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(X_test, y_test),
        callbacks=[es, mc],
    )

    return history


def build_optimizer(learning_rate):

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    return optimizer


def save_fig_history(history, path):

    plt.plot(history.history["loss"], label="loss")
    plt.plot(history.history["val_loss"], label="val_loss")
    plt.legend()
    plt.title("Train and validation loss")

    plt.savefig(path)

    return 1


def scale(X):
    scaler = MinMaxScaler()
    X_norm = []

    for x in X:
        x = scaler.fit_transform(x)
        X_norm.append(x)

    X_norm = np.array(X_norm, dtype=object)

    return X_norm


def pad(X):

    X_pad = tf.keras.preprocessing.sequence.pad_sequences(X, padding="post", dtype=float)

    return X_pad


def mean_position_error(pos, pos_pred):
    N = pos.shape[0]
    norms = np.linalg.norm(pos - pos_pred, axis=1)

    return norms.sum() / N


def dest_from_polar(start, polar):
    x_s, y_s = start[:, 0], start[:, 1]
    r, theta = polar[:, 0], polar[:, 1]

    x = r * np.cos(theta)
    y = r * np.sin(theta)

    x = x + x_s
    y = y + y_s

    return np.transpose(np.array([x, y]))


def evaluate_model(model, ind):
    starts = np.hstack(df["start"].values).reshape(-1, 2)[ind]
    dest = np.hstack(df["end"].values).reshape(-1, 2)[ind]

    preds_polar = model.predict(X_values[ind])

    dest_pred = dest_from_polar(starts, preds_polar)

    mpe = mean_position_error(dest, dest_pred)

    return mpe


if __name__ == "__main__":

    df = pd.read_pickle(INPUT_DATA_PATH)

    X_values = pad(df["IMU"].values)

    y_values = df["polar"].values
    y_values = np.asarray([elt for elt in y_values])

    n_samples = X_values.shape[0]
    print("number of samples : ", n_samples)
    indices = np.arange(n_samples)

    X_train, X_test, y_train, y_test, ind_train, ind_test = train_test_split(
        X_values, y_values, indices, test_size=0.2, random_state=0
    )

    print("train shape : ", X_train.shape)
    print("train shape : ", X_test.shape)

    model = build_model()

    print(model.summary())

    optimizer = build_optimizer(LEARNING_RATE)

    path_dir = "models/training_" + str(ID_TRAIN)

    if not os.path.isdir("models"):
        os.makedirs("models")

    os.makedirs(path_dir)
    os.chdir(path_dir)

    es = EarlyStopping(monitor="val_loss", mode="min", verbose=1, patience=10)

    mc = ModelCheckpoint(
        "best_model.h5",
        monitor="val_loss",
        mode="min",
        verbose=1,
        save_best_only=True,
    )

    history = train(model, optimizer, X_train, y_train, X_test, y_test, es, mc)
    save_fig_history(history, "loss.png")

    saved_model = load_model("best_model.h5")

    train_loss = saved_model.evaluate(X_train, y_train, verbose=0)
    test_loss = saved_model.evaluate(X_test, y_test, verbose=0)

    mpe_score_test = evaluate_model(saved_model, ind_test)
    mpe_score_train = evaluate_model(saved_model, ind_train)

    print(mpe_score_test)
    print(mpe_score_train)
