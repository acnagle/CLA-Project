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

    sorted_idx = np.argsort(date_to_num_arr, kind='mergesort')

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

    print('Appending additional features ... ')
    # compute sine and cosine transformation for time
    float_time = data[:, 0].astype(float)
    cos_time = np.cos((2 * np.pi * (float_time / 24)))
    sin_time = np.sin((2 * np.pi * (float_time / 24)))
    time = np.transpose(np.vstack((cos_time, sin_time)))

    # remove float encoding of time
    data = np.delete(data, obj=0, axis=1)
    date_int_encode = np.zeros(shape=(data.shape[0], 1))

    # compute integer encoding of date
    for i in range(data.shape[0]):
        # integer encoding of dates as a feature
        if date[0] == '6':
            date_int_encode[i] = int(date[1])
        elif date[0] == '7':
            date_int_encode[i] = 30 + int(date[1])      # 30 is number of days in June
        elif date[0] == '8':
            date_int_encode[i] = 61 + int(date[1])      # 61 is sum of number of days in June and July

    # add sine and cosine transformed time and integer encoding of day to data set
    data = np.hstack((time, date_int_encode, data))

    # create indicator features for whether there was rain or a bloom one day ago, or within three days or a week ago
    precip = (data[:, -1].astype(float) > 0).astype(int)   # convert precipitation to boolean values
    precip_one_day = np.roll(precip, 1)
    precip_one_day[0] = 0
    precip_three_day = np.zeros(shape=precip.shape, dtype=int)
    precip_one_week = np.zeros(shape=precip.shape, dtype=int)

    bloom = (summer_labels.astype(int) > 0).astype(int)
    bloom_one_day = np.roll(bloom, 1)
    bloom_one_day[0] = 0
    bloom_three_day = np.zeros(shape=bloom.shape, dtype=int)
    bloom_one_week = np.zeros(shape=bloom.shape, dtype=int)

    for i in range(len(precip)):
        if i == 0:
            continue

        if i < 3:
            precip_three_day[i] = np.sum(precip[0:i])
            precip_one_week[i] = np.sum(precip[0:i])

            bloom_three_day[i] = np.sum(bloom[0:i])
            bloom_one_week[i] = np.sum(bloom[0:i])
        else:
            precip_three_day[i] = np.sum(precip[i-3:i])
            bloom_three_day[i] = np.sum(bloom[i-3:i])

        if (i >= 3) and (i < 7):
            precip_one_week[i] = np.sum(precip[0:i])
            bloom_one_week[i] = np.sum(bloom[0:i])
        else:
            precip_one_week[i] = np.sum(precip[i-7:i])
            bloom_one_week[i] = np.sum(bloom[i-7:i])

    # add new features
    data = np.hstack((
        data,
        np.reshape(precip_one_day, newshape=(precip_one_day.shape[0], 1)),
        np.reshape(precip_three_day, newshape=(precip_three_day.shape[0], 1)),
        np.reshape(precip_one_week, newshape=(precip_one_week.shape[0], 1)),
        np.reshape(bloom_one_day, newshape=(bloom_one_day.shape[0], 1)),
        np.reshape(bloom_three_day, newshape=(bloom_three_day.shape[0], 1)),
        np.reshape(bloom_one_week, newshape=(bloom_one_week.shape[0], 1))
    ))

    print('Saving new data set ... ')
    try:
        data = data.astype(float)      # convert data type to float
    except ValueError:
        for i in range(0, data.shape[0], -1):
            if '' in data[i, :]:
                data = np.delete(data, obj=i, axis=0)

    filename_data = 'hourly_' + cla_path.split('/')[-1]
    filename_labels = 'hourly_' + label_path.split('/')[-1]
    filename_loc = 'hourly_' + loc_path.split('/')[-1]

    print(data[:, 3].astype(int))

    np.save('../Data/hourly-data-sets/' + filename_data, data)
    np.save('../Data/hourly-data-sets/' + filename_labels, summer_labels.astype(int))
    np.save('../Data/hourly-data-sets/' + filename_loc, summer_loc)


if __name__ == "__main__": main()
