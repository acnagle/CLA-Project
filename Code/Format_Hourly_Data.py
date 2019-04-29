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
        label_path = str(sys.argv[2])
        loc_path = str(sys.argv[3])
        weather_path = str(sys.argv[4])

        cla_data = np.load(cla_path)
        labels = np.load(label_path)
        loc = np.load(loc_path)
        weather_data = np.load(weather_path)
    except (ValueError, IndexError, FileNotFoundError):
        print('Arguments must be specified as follows:')
        print('python3 format_hourly_data.py <path-to-cla-data (.npy)> <path-to-cla-data-labels (.npy)> '
              '<path-to-locations (.npy)> <path-to-weather-data (.npy)>')
        sys.exit(0)

    # get initial index of weather data for this data set
    year = cla_data[0, 0].split('/')[2]

    for i in range(weather_data.shape[0]):
        if weather_data[i, 0].split(' ')[0].split('/')[2] == year:
            weather_idx = i
            break

    print('Removing non-summer months ... ')
    # remove non-summer months. all data points must be in June, July, August
    summer_cla_data = np.array([])
    summer_labels = np.array([])
    summer_loc = np.array([])

    for i in range(cla_data.shape[0]-1, 0, -1):
        cla_month = int(cla_data[i, 0].split('/')[0])

        if (cla_month == 6) or (cla_month == 7) or (cla_month == 8):
            if summer_cla_data.shape[0] == 0:
                summer_cla_data = cla_data[i, :]
            else:
                summer_cla_data = np.vstack((summer_cla_data, cla_data[i, :]))

            summer_labels = np.append(summer_labels, labels[i])
            summer_loc = np.append(summer_loc, loc[i])

    print('Sorting CLA data ... ')
    # sort cla_data by dates
    date_to_num_arr = np.zeros(shape=(summer_cla_data.shape[0]))

    for i in range(summer_cla_data.shape[0]):
        date = summer_cla_data[i, 0].split('/')
        num = float(date[0]) + (float(date[1]) / 100)
        date_to_num_arr[i] = num

    sorted_idx = np.argsort(date_to_num_arr, kind='merge')

    date_to_num_arr = date_to_num_arr[sorted_idx]
    summer_cla_data = summer_cla_data[sorted_idx, :]
    summer_labels = summer_labels[sorted_idx]
    summer_loc = summer_loc[sorted_idx]

    print('Appending weather data ... ')
    # append weather data to cla data
    data = np.empty(shape=(summer_cla_data.shape[0], summer_cla_data.shape[1] + weather_data.shape[1]-3), dtype=(str, 100))
    data[:, :5] = summer_cla_data[:, 1:]

    cla_idx = 0

    while cla_idx != (summer_cla_data.shape[0]-1):
        weather_date = weather_data[weather_idx, 0].split(' ')[0].split('/')
        weather_date_to_num = float(weather_date[0]) + (float(weather_date[1]) / 100)

        if year < weather_date[2]:
            weather_idx += 1
        elif year > weather_date[2]:
            break
        else:
            if date_to_num_arr[cla_idx] != weather_date_to_num:
                weather_idx += 1
            else:
                cla_time_to_num = summer_cla_data[cla_idx, 1]

                weather_idx_day = weather_idx

                weather_time = weather_data[weather_idx_day, 0].split(' ')[1].split(':')
                weather_time_to_num = float(weather_time[0]) + (float(weather_time[1]) / 60)

                while cla_time_to_num > weather_time_to_num:
                    weather_idx_day += 1

                    weather_time = weather_data[weather_idx_day, 0].split(' ')[1].split(':')
                    weather_time_to_num = float(weather_time[0]) + (float(weather_time[1]) / 60)

                weather_idx_day -= 1

                data[cla_idx, 5:] = weather_data[weather_idx_day, 2:]

                if cla_idx < (cla_data.shape[0]-1):
                    next_cla_date = summer_cla_data[cla_idx+1, 0].split('/')
                    next_cla_date_to_num = float(next_cla_date[0]) + (float(next_cla_date[1]) / 100)

                    if next_cla_date_to_num != weather_date_to_num:
                        weather_idx += 1

                cla_idx += 1

    data = np.delete(data, obj=-1, axis=0)
    summer_labels = np.delete(summer_labels, obj=-1)
    summer_loc = np.delete(summer_loc, obj=-1)

    print('Saving new data set ... ')
    try:
        data.astype(float)      # convert data type to float
    except ValueError:
        for i in range(0, data.shape[0], -1):
            if '' in data[i, :]:
                data = np.delete(data, obj=i, axis=0)

    filename_data = 'hourly_' + cla_path.split('/')[-1]
    filename_labels = 'hourly_' + label_path.split('/')[-1]
    filename_loc = 'hourly_' + loc_path.split('/')[-1]

    np.save('../Data/hourly-data-sets/' + filename_data, data)
    np.save('../Data/hourly-data-sets/' + filename_labels, summer_labels)
    np.save('../Data/hourly-data-sets/' + filename_loc, summer_loc)


if __name__ == "__main__": main()
