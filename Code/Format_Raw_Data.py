import numpy as np
import pandas as pd
import errno
import sys
import os


def main():
    np.set_printoptions(threshold=np.inf)  # prints a full matrix rather than an abbreviated matrix
    # pd.options.display.max_colwidth = 100   # number of characters of lyrics shown when dataframe is printed

    print("\n\t##### EXECUTING FORMAT_RAW_DATA.PY #####\n")

    dest_path = '../Data/data/'

    # read in arguments from command line
    try:
        data_set_path = str(sys.argv[1])
    except (ValueError, IndexError):
        print('Arguments must be specified as follows:')
        print('python3 Format_Raw_Data.py <path-to-csv>')
        sys.exit(0)

    if os.path.exists(data_set_path):
        df = pd.read_csv(data_set_path, dtype=str)
    else:
        print('File not found; path does not exist')
        sys.exit(0)

    print('Extracting relevant information ...')
    if '2014' in data_set_path:
        features = [1, 3, 4, 5, 6, 7, 9, 10, 12, 15, 14, 11]
        data_set_idx = [8, 9, 10, 11]
        year = '2014'
    elif '2015' in data_set_path:
        features = [4, 5, 6, 7, 8, 10, 14, 15, 17, 20, 19, 16]
        data_set_idx = [8, 9, 10, 11]
        year = '2015'
    elif '2016' in data_set_path:
        features = [4, 5, 6, 7, 8, 10, 14, 15, 17, 20, 19, 16]
        data_set_idx = [8, 9, 10, 11]
        year = '2016'
    elif '2017' in data_set_path:
        features = [4, 5, 6, 7, 8, 10, 14, 15, 17, 20, 19, 16]
        data_set_idx = [8, 9, 10, 11]
        year = '2017'
    elif '2018' in data_set_path:
        features = [1, 2, 3, 4, 5, 6, 7, 9, 12, 11, 8]
        data_set_idx = [7, 8, 9, 10]
        year = '2018'
    else:
        print('Provided incorrect year. Must be years 2014-2018')
        sys.exit()

    data_set = df.to_numpy()[:, features]    # keep relevant features
    locations = data_set[:, 0]
    labels = data_set[:, 3]

    # get dates and times
    if year == '2018':  # special case for 2018
        times = np.empty(shape=(data_set.shape[0], ), dtype=object)
        dates = np.empty(shape=(data_set.shape[0],), dtype=object)
        for i in range(data_set.shape[0]):
            date_time = data_set[i, 6].split(' ')
            dates[i] = str(date_time[0])
            times[i] = str(date_time[1])
    else:
        dates = data_set[:, 6]
        times = data_set[:, 7]

    data_set = data_set[:, data_set_idx]  # remove locations, dates, plantDebris, algalBloomsheen, airTemp, batherLoad, algalBloom

    print('Cleaning data set ...')
    # find indices in data set to throw away
    delete_idx = []
    for i in range(data_set.shape[0]):
        row = data_set[i, :]
        if ('undefined' in row) or ('NA' in row) or (labels[i] == 'FALSE'):
            delete_idx.append(i)

    data_set = np.delete(data_set, obj=delete_idx, axis=0)
    locations = np.delete(locations, obj=delete_idx)
    times = np.delete(times, obj=delete_idx)
    dates = np.delete(dates, obj=delete_idx)
    labels = np.delete(labels, obj=delete_idx)
    data_set = data_set.astype(float)   # convert data type of data set
    labels = labels.astype(float)       # so that np.isnan(labels[i]) in the code below does not throw an error

    # remove any nans
    delete_idx = []
    for i in range(data_set.shape[0]):
        row = data_set[i, :]
        if (True in np.isnan(row)) or (np.isnan(labels[i])):
            delete_idx.append(i)
        else:
            if row[0] > 1:
                data_set[i, 0] = 1
            else:
                data_set[i, 0] = 0
            if row[1] > 1:
                data_set[i, 1] = 1
            else:
                data_set[i, 1] = 0

    data_set = np.delete(data_set, obj=delete_idx, axis=0)
    locations = np.delete(locations, obj=delete_idx)
    times = np.delete(times, obj=delete_idx)
    dates = np.delete(dates, obj=delete_idx)
    labels = np.delete(labels, obj=delete_idx)
    labels = labels.astype(int)

    times = convert_time_to_measurement(times)
    times = np.reshape(times, newshape=(len(times), 1))
    data_set = np.hstack((times, data_set))
    data_set = data_set.astype(object)
    dates = np.reshape(dates, newshape=(len(dates), 1))
    data_set = np.hstack((dates, data_set))

    print('Data set size =', data_set.shape)
    print('labels size =', labels.shape)

    print('Saving data ...')
    np.save(dest_path + 'X_' + year, data_set)
    np.save(dest_path + 'locations_' + year, locations)
    np.save(dest_path + 'y_' + year, labels)


def convert_time_to_measurement(times):
    for i in range(0, times.shape[0]):
        date_str = times[i]
        time = date_str.split(":")
        time_meas = float(time[0]) + (float(time[1]) / 60)

        times[i] = str(time_meas)

    return times.astype(float)


if __name__ == "__main__": main()