import pandas as pd
import numpy as np

def to_polar(start, end):
    x1, y1 = start[0], start[1]
    x2, y2 = end[0], end[1]

    vec = np.array([x2 - x1, y2 - y1])

    delta_r = np.sqrt(vec[0] ** 2 + vec[1] ** 2)
    delta_phi = np.arctan2(vec[1], vec[0])

    return np.array([delta_r, delta_phi])


def dest_from_polar(start, polar):
    x_s, y_s = start[:, 0], start[:, 1]
    r, theta = polar[:, 0], polar[:, 1]

    x = r * np.cos(theta)
    y = r * np.sin(theta)

    x = x + x_s
    y = y + y_s

    return np.transpose(np.array([x, y]))


def is_nan(y):
    s = y.sum()
    return np.isnan(s.sum())


def remove_nan_values(df):
    mask3 = ~df["polar"].apply(lambda x: is_nan(x))
    df = df[mask3]
    return df


def stack_and_reshape(m1, m2, m3):
    all_measures = np.stack([m1, m2, m3], axis=1)
    all_measures = all_measures.reshape(-1, 9)
    return np.array(all_measures)


def get_short_sequences(df, min_len, max_len):
    df["len"] = df["acce"].apply(lambda x: len(x))
    df = df[(df["len"] >= min_len) & (df["len"] < max_len)]
    return df