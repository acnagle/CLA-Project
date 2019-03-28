import numpy as np
# import Constants
# from sklearn import preprocessing
# import errno
# import os
import sys

def main():
    np.set_printoptions(threshold=np.inf)  # prints a full matrix rather than an abbreviated matrix

    print('\n\t##### EXECUTING FORMAT_HOURLY_WEATHER_DATA.PY #####\n')

    print('Reading in data ... ')
    # read in arguments from command line
    try:
        csv_path = str(sys.argv[1])
    except (ValueError, IndexError):
        print('Arguments must be specified as follows:')
        print('python3 one_hot_encoding.py <path-to-weather-csv>')
        sys.exit(0)

    raw_data = np.genfromtxt(
        open(csv_path),
        delimiter=",",
        dtype=(str, 100),
        usecols=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9),
        skip_header=1,
    )

    print('Formatting hourly weather data ... ')

    format_data = np.zeros(shape=raw_data.shape[1])

    for i in range(raw_data.shape[0]):
        print(i)
        line = raw_data[i, :]
        if 'FM-15' == line[1]:
            if line[9] == 'T':
                line[9] = '0.001'
            elif 's' in line[9]:
                line[9] = line[9][:-1]

            format_data = np.vstack((format_data, line))

    format_data = np.delete(format_data, obj=0, axis=0)
    np.save('hourly_weather_data.npy', format_data)


if __name__ == "__main__":main()
