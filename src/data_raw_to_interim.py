# transform raw data into dataframe with the paths (start,end) and the signals
# sensed during the displacement.
# output file : data_sites_floors_paths.csv

OUTPUT_PATH = "data/final/interim/data_sites_floors_paths.csv"

import os
import pandas as pd
from utils import read_data_file, split_sub_paths
import dataclasses


if __name__ == "__main__":

    ROOT = "data/train"

    sites = os.listdir(ROOT)

    df = pd.DataFrame(
        columns=["site", "floor", "path", "start", "end", "acce", "gyro", "ahrs"]
    )

    i = 0
    for site in sites:

        i = i + 1
        print(i)
        floors = os.listdir(ROOT + "/" + site)  # =B1,F1 ...

        for floor in floors:

            paths = os.listdir(ROOT + "/" + site + "/" + floor)

            for path in paths:

                absolute_path = ROOT + "/" + site + "/" + floor + "/" + path
                data = read_data_file(absolute_path)
                subpaths = split_sub_paths(data)

                for subpath in subpaths:
                    subpath_dict = dataclasses.asdict(subpath)
                    subpath_dict["site"] = site
                    subpath_dict["floor"] = floor
                    subpath_dict["path"] = path
                    df = df.append(subpath_dict, ignore_index=True)

    df.to_pickle(OUTPUT_PATH)
