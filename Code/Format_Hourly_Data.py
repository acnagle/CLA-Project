import numpy as np
import os
import sys


def main():
    np.set_printoptions(threshold=np.inf)  # prints full numpy array (no truncation)

    print('\n\t##### EXECUTING FORMAT_HOURLY_DATA.PY #####\n')

    print('Reading in data ... ')
    # read in arguments from command line
    try:
        cla_path = str(sys.argv[1])
        weather_path = str(sys.argv[2])

        cla_data = np.load(cla_path)
        weather_data = np.load(weather_path)
    except (ValueError, IndexError, FileNotFoundError):
        print('Arguments must be specified as follows:')
        print('python3 format_hourly_data.py <path-to-cla-data (.npy)> <path-to-weather-data (.npy)>')
        sys.exit(0)

    # get initial index of weather data for this data set
    year = cla_data[0, 1].split('/')[2]

    for i in range(weather_data.shape[0]):
        if weather_data[i, 0].split(' ')[0].split('/')[2] == year:
            weather_idx = i
            break

    print('Removing non-summer months ... ')
    # remove non-summer months. all data points must be in June, July, August
    summer_cla_data = np.zeros(shape=(cla_data.shape[1]))

    for i in range(cla_data.shape[0]-1, 0, -1):
        cla_month = int(cla_data[i, 1].split('/')[0])

        if (cla_month == 6) or (cla_month == 7) or (cla_month == 8):
            summer_cla_data = np.vstack((summer_cla_data, cla_data[i, :]))

    summer_cla_data = np.delete(summer_cla_data, obj=0, axis=0)

    print('Sorting CLA data ... ')
    # sort cla_data by dates
    date_to_num_arr = np.zeros(shape=(summer_cla_data.shape[0]))

    for i in range(summer_cla_data.shape[0]):
        date = summer_cla_data[i, 1].split('/')
        num = float(date[0]) + (float(date[1]) / 100)
        date_to_num_arr[i] = num

    sorted_idx = np.argsort(date_to_num_arr, kind='merge')

    date_to_num_arr = date_to_num_arr[sorted_idx]
    summer_cla_data = summer_cla_data[sorted_idx, :]

    print('Appending weather data ... ')
    # append weather data to cla data
    data = np.empty(shape=(summer_cla_data.shape[0], summer_cla_data.shape[1] + weather_data.shape[1]-6), dtype=(str, 100))
    data[:, 0] = summer_cla_data[:, 2]
    data[:, 1:5] = summer_cla_data[:, 5:]

    num_hourly_samples_per_day = 24
    cla_idx = 0

    while cla_idx != (summer_cla_data.shape[0]-1):
        weather_date = weather_data[weather_idx, 0].split(' ')[0].split('/')
        weather_date_to_num = float(weather_date[0]) + (float(weather_date[1]) / 100)

        if year != weather_date[2]:
            weather_idx += num_hourly_samples_per_day
        else:
            if date_to_num_arr[cla_idx] != weather_date_to_num:
                weather_idx += num_hourly_samples_per_day
            else:
                cla_time = summer_cla_data[cla_idx, 2].split(':')
                cla_time_to_num = float(cla_time[0]) + (float(cla_time[1]) / 100)

                weather_idx_day = weather_idx

                weather_time = weather_data[weather_idx_day, 0].split(' ')[1].split(':')
                weather_time_to_num = float(weather_time[0]) + (float(weather_time[1]) / 100)

                # print(date_to_num_arr[cla_idx], cla_time_to_num, weather_time_to_num)

                while cla_time_to_num > weather_time_to_num:
                    weather_idx_day += 1

                    weather_time = weather_data[weather_idx_day, 0].split(' ')[1].split(':')
                    weather_time_to_num = float(weather_time[0]) + (float(weather_time[1]) / 100)

                weather_time = weather_data[weather_idx_day-1, 0].split(' ')[1].split(':')
                weather_time_to_num = float(weather_time[0]) + (float(weather_time[1]) / 100)

                # print(date_to_num_arr[cla_idx], weather_date_to_num, cla_time_to_num, weather_time_to_num)

                data[cla_idx, 5:] = weather_data[weather_idx_day, 2:]

                if cla_idx < (cla_data.shape[0]-1):
                    next_cla_date = summer_cla_data[cla_idx+1, 1].split('/')
                    next_cla_date_to_num = float(next_cla_date[0]) + (float(next_cla_date[1]) / 100)

                    if next_cla_date_to_num != weather_date_to_num:
                        weather_idx += num_hourly_samples_per_day

                cla_idx += 1

    data = np.delete(data, obj=data.shape[0]-1, axis=0)

    print('Converting time to float values ... ')
    # convert time to float value
    for i in range(0, data.shape[0]):
        date_str = data[i, 0]
        time = date_str.split(":")
        time_meas = float(time[0]) + (float(time[1]) / 60)

        data[i, 0] = str(time_meas)

    print('Saving new data set ... ')
    try:
        data.astype(float)      # convert data type to float
    except ValueError:
        for i in range(0, data.shape[0], -1):
            if '' in data[i, :]:
                data = np.delete(data, obj=i, axis=0)

    filename = 'hourly_' + cla_path.split('/')[-1]

    np.save('../Data/hourly-data-sets/' + filename, data)


if __name__ == "__main__": main()
