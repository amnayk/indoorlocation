import pandas as pd
import numpy as np

from preprocessing import get_short_sequences, to_polar, remove_nan_values, stack_and_reshape

MIN_LEN = 200
MAX_LEN = 500

INPUT_DATA_PATH = "data/interim/data_sites_floors_paths.csv"
OUTPUT_DATA_PATH = "data/processed/data_imu_polar_{}_{}.csv".format(MIN_LEN, MAX_LEN)

if __name__ == "__main__":
    df = pd.read_pickle(INPUT_DATA_PATH)

    df = get_short_sequences(df, MIN_LEN, MAX_LEN)

    df = df.reset_index(drop=True)

    df["polar"] = df[["start", "end"]].apply(lambda x: to_polar(x["start"], x["end"]), axis=1)

    df = remove_nan_values(df)

    df["IMU"] = df.apply(lambda x: stack_and_reshape(x["acce"], x["gyro"], x["ahrs"]), axis=1)

    df = df[["site", "floor", "path", "start", "end", "IMU", "polar"]]

    df.to_pickle(OUTPUT_DATA_PATH)
