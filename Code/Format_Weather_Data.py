import numpy as np
import Constants
from sklearn import preprocessing
import errno
import os
import sys


def main():
    np.set_printoptions(threshold=np.inf)  # prints a full matrix rather than an abbreviated matrix

    print("\n\t##### EXECUTING FORMAT_WEATHER_DATA.PY #####\n")

    data_path = "/Users/Alliot/documents/cla-project/data/Weather_Data_edit.csv"

    raw_data = np.genfromtxt(
        open(data_path),
        delimiter=",",
        dtype=(str, 100),
        usecols=(5, 6, 26, 27, 28, 29, 31, 32, 38, 41, 42, 43, 44, 45, 46, 47),     # index 6 is the summary of day flag
        skip_header=1
    )

    data = np.empty(shape=(raw_data.shape[Constants.COLUMNS], ), dtype=(str, 100))

    # remove non-summer dates and only keep daily averages
    for i in range(0, raw_data.shape[Constants.ROWS]):
        if raw_data[i, 1] == "SOD":
            month = int(raw_data[i, Constants.WEATHER_DATE_TIME][6])
            year = int(raw_data[i, Constants.WEATHER_DATE_TIME][2:4])

            if year != 18:  # exclude 2018 for now
                if (month == 6) or (month == 7) or (month == 8):
                    # This line removes the times from the data
                    raw_data[i, Constants.WEATHER_DATE_TIME] = raw_data[i, Constants.WEATHER_DATE_TIME][0:10]
                    # convert T (Trace precipitation) to 0.005
                    if raw_data[i, Constants.WEATHER_PRECIP] == "T":
                        raw_data[i, Constants.WEATHER_PRECIP] = "0.005"

                    data = np.vstack((data, raw_data[i, :]))

    data = np.delete(data, obj=0, axis=Constants.ROWS)
    data = np.delete(data, obj=1, axis=Constants.COLUMNS)

    np.save("/Users/Alliot/Documents/CLA-Project/Data/weather_data.npy", data)

    # Count data points to make sure there is data for every day
    # summ = 0
    # for i in range(0, data.shape[0]):
    #     day = int(data[i, Constants.WEATHER_DATE_TIME][8:10])
    #     summ += day
    #
    # print(summ)

    # Check for missing elements
    # for i in range(0, data.shape[Constants.ROWS]):
    #     if "" in data[i, :]:
    #         for j in range(0, data.shape[Constants.COLUMNS]):
    #             if data[i, j] == "":
    #                 print(j, data[i, :])


if __name__ == "__main__": main()