import numpy as np
import Constants
from sklearn import preprocessing
import errno
import os
import sys


def main():
    np.set_printoptions(threshold=np.inf)  # prints a full matrix rather than an abbreviated matrix

    data_path = "/Users/Alliot/documents/cla-project/data/Weather_Data.csv"

    raw_data = np.genfromtxt(
        open(data_path),
        delimiter=",",
        dtype=(str, 100),
        usecols=(5, 26, 27, 28, 29, 31, 32, 38, 41, 42, 43, 44, 45, 46, 47),
        skip_header=1
    )

    data = np.empty(shape=(raw_data.shape[Constants.COLUMNS], ), dtype=(str, 100))

    # remove non-summer dates and only keep daily averages
    for i in range(0, raw_data.shape[Constants.ROWS]):
        month = int(raw_data[i, Constants.WEATHER_DATE_TIME][6])
        year = int(raw_data[i, Constants.WEATHER_DATE_TIME][2:4])

        if year != 18:  # exclude 2018 for now
            if (month == 6) or (month == 7) or (month == 8):
                if raw_data[i, raw_data.shape[Constants.COLUMNS]-1] != "":
                    data = np.vstack((data, raw_data[i, :]))

    data = np.delete(data, obj=0, axis=Constants.ROWS)

    for i in range(0, data.shape[Constants.ROWS]):
        if "" in data[i, :]:
            for j in range(0, data.shape[Constants.COLUMNS]):
                if data[i, j] == "":
                    print(j, data[i, :])


if __name__ == "__main__": main()