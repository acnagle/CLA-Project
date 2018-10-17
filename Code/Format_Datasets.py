import numpy as np
import Constants
from sklearn import preprocessing
import errno
import os


def main():
    np.set_printoptions(threshold=np.inf)  # prints a full matrix rather than an abbreviated matrix

    print("\n\t##### EXECUTING FORMAT_DATASETS.PY #####\n")

    weather_data_path = "/Users/Alliot/Documents/CLA-Project/Data/weather_data.npy"
    cla_data_path = "/Users/Alliot/Documents/CLA-Project/Data/raw-data/"
    dest_path = "/Users/Alliot/Documents/CLA-Project/Data/data-sets/"

    # create dest_path if it does not exist
    if not os.path.exists(dest_path):
        try:
            os.makedirs(dest_path)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

    print("Reading in data ... ")
    weather_data = np.load(weather_data_path)

    # data_2014 = np.load(cla_data_path + "data_2014.npy")
    # data_2015 = np.load(cla_data_path + "data_2015.npy")
    # data_2016 = np.load(cla_data_path + "data_2016.npy")
    # data_2017 = np.load(cla_data_path + "data_2017.npy")

    data_2014_summer = np.load(cla_data_path + "data_2014_summer.npy")
    data_2015_summer = np.load(cla_data_path + "data_2015_summer.npy")
    data_2016_summer = np.load(cla_data_path + "data_2016_summer.npy")
    data_2017_summer = np.load(cla_data_path + "data_2017_summer.npy")

    # all_data = np.load(cla_data_path + "all_data.npy")
    all_data_summer = np.load(cla_data_path + "all_data_summer.npy")

    # data_mendota = np.load(cla_data_path + "data_mendota.npy")
    # data_monona = np.load(cla_data_path + "data_monona.npy")
    # data_kegonsa = np.load(cla_data_path + "data_kegonsa.npy")
    # data_waubesa = np.load(cla_data_path + "data_waubesa.npy")
    # data_wingra = np.load(cla_data_path + "data_wingra.npy")

    data_mendota_summer = np.load(cla_data_path + "data_mendota_summer.npy")
    data_monona_summer = np.load(cla_data_path + "data_monona_summer.npy")
    data_kegonsa_summer = np.load(cla_data_path + "data_kegonsa_summer.npy")
    data_waubesa_summer = np.load(cla_data_path + "data_waubesa_summer.npy")
    data_wingra_summer = np.load(cla_data_path + "data_wingra_summer.npy")

    print("Appending weather data ... ")
    # Create dictionary for accessing a particular index of weather_data
    weather_dict = {}   # uses the date as a key for indexing weather_data rows
    for i in range(0, weather_data.shape[Constants.ROWS]):
        month = int(weather_data[i, Constants.DATE_NO_LOC][6])
        day = int(weather_data[i, Constants.DATE_NO_LOC][8:])
        year = int(weather_data[i, Constants.DATE_NO_LOC][2:4])

        key = str(month) + "/" + str(day) + "/" + str(year)

        weather_dict[key] = i

    data_2014_summer = append_weather_data(
        cla_data=data_2014_summer,
        weather_data=weather_data,
        weather_dict=weather_dict
    )
    convert_time_to_measurement(data_2014_summer)
    data_2014_summer = np.delete(data_2014_summer, obj=Constants.DATE_NO_LOC, axis=Constants.COLUMNS)  # Remove date

    data_2015_summer = append_weather_data(
        cla_data=data_2015_summer,
        weather_data=weather_data,
        weather_dict=weather_dict
    )
    convert_time_to_measurement(data_2015_summer)
    data_2015_summer = np.delete(data_2015_summer, obj=Constants.DATE_NO_LOC, axis=Constants.COLUMNS)

    data_2016_summer = append_weather_data(
        cla_data=data_2016_summer,
        weather_data=weather_data,
        weather_dict=weather_dict
    )
    convert_time_to_measurement(data_2016_summer)
    data_2016_summer = np.delete(data_2016_summer, obj=Constants.DATE_NO_LOC, axis=Constants.COLUMNS)

    data_2017_summer = append_weather_data(
        cla_data=data_2017_summer,
        weather_data=weather_data,
        weather_dict=weather_dict
    )
    convert_time_to_measurement(data_2017_summer)
    data_2017_summer = np.delete(data_2017_summer, obj=Constants.DATE_NO_LOC, axis=Constants.COLUMNS)

    all_data_summer = append_weather_data(
        cla_data=all_data_summer,
        weather_data=weather_data,
        weather_dict=weather_dict
    )
    convert_time_to_measurement(all_data_summer)
    all_data_summer = np.delete(all_data_summer, obj=Constants.DATE_NO_LOC, axis=Constants.COLUMNS)

    data_mendota_summer = append_weather_data(
        cla_data=data_mendota_summer,
        weather_data=weather_data,
        weather_dict=weather_dict
    )
    convert_time_to_measurement(data_mendota_summer)
    data_mendota_summer = np.delete(data_mendota_summer, obj=Constants.DATE_NO_LOC, axis=Constants.COLUMNS)

    data_monona_summer = append_weather_data(
        cla_data=data_monona_summer,
        weather_data=weather_data,
        weather_dict=weather_dict
    )
    convert_time_to_measurement(data_monona_summer)
    data_monona_summer = np.delete(data_monona_summer, obj=Constants.DATE_NO_LOC, axis=Constants.COLUMNS)

    data_kegonsa_summer = append_weather_data(
        cla_data=data_kegonsa_summer,
        weather_data=weather_data,
        weather_dict=weather_dict
    )
    convert_time_to_measurement(data_kegonsa_summer)
    data_kegonsa_summer = np.delete(data_kegonsa_summer, obj=Constants.DATE_NO_LOC, axis=Constants.COLUMNS)

    data_waubesa_summer = append_weather_data(
        cla_data=data_waubesa_summer,
        weather_data=weather_data,
        weather_dict=weather_dict
    )
    convert_time_to_measurement(data_waubesa_summer)
    data_waubesa_summer = np.delete(data_waubesa_summer, obj=Constants.DATE_NO_LOC, axis=Constants.COLUMNS)

    data_wingra_summer = append_weather_data(
        cla_data=data_wingra_summer,
        weather_data=weather_data,
        weather_dict=weather_dict
    )
    convert_time_to_measurement(data_wingra_summer)
    data_wingra_summer = np.delete(data_wingra_summer, obj=Constants.DATE_NO_LOC, axis=Constants.COLUMNS)

    print("Normalizing data sets ... ")
    data_2014_summer = preprocessing.normalize(data_2014_summer)
    data_2015_summer = preprocessing.normalize(data_2015_summer)
    data_2016_summer = preprocessing.normalize(data_2016_summer)
    data_2017_summer = preprocessing.normalize(data_2017_summer)
    all_data_summer = preprocessing.normalize(all_data_summer)

    data_mendota_summer = preprocessing.normalize(data_mendota_summer)
    data_monona_summer = preprocessing.normalize(data_monona_summer)
    data_kegonsa_summer = preprocessing.normalize(data_kegonsa_summer)
    data_waubesa_summer = preprocessing.normalize(data_waubesa_summer)
    data_wingra_summer = preprocessing.normalize(data_wingra_summer)

    print("Saving data set ... ")
    np.save(dest_path + "data_2014_summer", data_2014_summer)
    np.save(dest_path + "data_2015_summer", data_2015_summer)
    np.save(dest_path + "data_2016_summer", data_2016_summer)
    np.save(dest_path + "data_2017_summer", data_2017_summer)

    np.save(dest_path + "all_data_summer", all_data_summer)

    np.save(dest_path + "data_mendota_summer", data_mendota_summer)
    np.save(dest_path + "data_monona_summer", data_monona_summer)
    np.save(dest_path + "data_kegonsa_summer", data_kegonsa_summer)
    np.save(dest_path + "data_waubesa_summer", data_waubesa_summer)
    np.save(dest_path + "data_wingra_summer", data_wingra_summer)


# This method takes in CLA data and appends weather data to each day. cla_data is the data from CLA and weather_data
# is the weather data. weather_dict is a dictionary that maps dates in string format to a row index in weather_data
def append_weather_data(cla_data, weather_data, weather_dict):
    date = cla_data[0, Constants.DATE_NO_LOC]
    weather_idx = weather_dict[date]
    new_data = np.hstack((cla_data[0, :], weather_data[weather_idx, 1:]))

    for i in range(1, cla_data.shape[Constants.ROWS]):
        date = cla_data[i, Constants.DATE_NO_LOC]
        weather_idx = weather_dict[date]
        new_sample = np.hstack((cla_data[i, :], weather_data[weather_idx, 1:]))
        new_data = np.vstack((new_data, new_sample))

    return new_data


# This method converts all date and times into a number between [0, 24). This way, we can use time as a quantitative
# measurement.
def convert_time_to_measurement(mat):
    for i in range(0, mat.shape[Constants.ROWS]):
        date_str = mat[i, Constants.TIME_NO_LOC]
        time = date_str.split(":")
        time_meas = int(time[0]) + (int(time[1]) / 60)

        mat[i, Constants.TIME_NO_LOC] = str(time_meas)


if __name__ == "__main__": main()