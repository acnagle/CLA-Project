import numpy as np
import Constants
from sklearn import preprocessing
import errno
import os


def main():
    np.set_printoptions(threshold=np.inf)  # prints a full matrix rather than an abbreviated matrix

    print("\n\t##### EXECUTING FORMAT_DATASETS.PY #####\n")

    dest_path = "/Users/Alliot/Documents/CLA-Project/Data/data-sets/"

    # create dest_path if it does not exist
    if not os.path.exists(dest_path):
        try:
            os.makedirs(dest_path)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

    print("Reading in data ... ")

    # TODO INSERT CODE TO GO THROUGH DIRECTORY AND READ IN DATA

    # TODO COMBINE WEATHER DATA AND CLA DATA

    print("Normalizing data sets ... ")
    convert_datetime_to_measurement(data_2014)
    convert_datetime_to_measurement(data_2015)
    convert_datetime_to_measurement(data_2016)
    convert_datetime_to_measurement(data_2017)
    convert_datetime_to_measurement(data_summary)

    convert_datetime_to_measurement(data_2014_summer)
    convert_datetime_to_measurement(data_2015_summer)
    convert_datetime_to_measurement(data_2016_summer)
    convert_datetime_to_measurement(data_2017_summer)
    convert_datetime_to_measurement(data_summary_summer)

    convert_datetime_to_measurement(all_data)
    convert_datetime_to_measurement(all_data_summer)

    convert_datetime_to_measurement(data_mendota)
    convert_datetime_to_measurement(data_monona)
    convert_datetime_to_measurement(data_kegonsa)
    convert_datetime_to_measurement(data_waubesa)
    convert_datetime_to_measurement(data_wingra)

    convert_datetime_to_measurement(data_mendota_summary)
    convert_datetime_to_measurement(data_monona_summary)
    convert_datetime_to_measurement(data_kegonsa_summary)
    convert_datetime_to_measurement(data_waubesa_summary)
    convert_datetime_to_measurement(data_wingra_summary)

    convert_datetime_to_measurement(data_mendota_summer)
    convert_datetime_to_measurement(data_monona_summer)
    convert_datetime_to_measurement(data_kegonsa_summer)
    convert_datetime_to_measurement(data_waubesa_summer)
    convert_datetime_to_measurement(data_wingra_summer)

    convert_datetime_to_measurement(data_mendota_summary_summer)
    convert_datetime_to_measurement(data_monona_summary_summer)
    convert_datetime_to_measurement(data_kegonsa_summary_summer)
    convert_datetime_to_measurement(data_waubesa_summary_summer)
    convert_datetime_to_measurement(data_wingra_summary_summer)

    data_2014 = preprocessing.normalize(data_2014)
    data_2015 = preprocessing.normalize(data_2015)
    data_2016 = preprocessing.normalize(data_2016)
    data_2017 = preprocessing.normalize(data_2017)
    data_summary = preprocessing.normalize(data_summary)

    data_2014_summer = preprocessing.normalize(data_2014_summer)
    data_2015_summer = preprocessing.normalize(data_2015_summer)
    data_2016_summer = preprocessing.normalize(data_2016_summer)
    data_2017_summer = preprocessing.normalize(data_2017_summer)
    data_summary_summer = preprocessing.normalize(data_summary_summer)

    all_data = preprocessing.normalize(all_data)
    all_data_summer = preprocessing.normalize(all_data_summer)

    data_mendota = preprocessing.normalize(data_mendota)
    data_monona = preprocessing.normalize(data_monona)
    data_kegonsa = preprocessing.normalize(data_kegonsa)
    data_waubesa = preprocessing.normalize(data_waubesa)
    data_wingra = preprocessing.normalize(data_wingra)

    data_mendota_summary = preprocessing.normalize(data_mendota_summary)
    data_monona_summary = preprocessing.normalize(data_monona_summary)
    data_kegonsa_summary = preprocessing.normalize(data_kegonsa_summary)
    data_waubesa_summary = preprocessing.normalize(data_waubesa_summary)
    data_wingra_summary = preprocessing.normalize(data_wingra_summary)

    data_mendota_summer = preprocessing.normalize(data_mendota_summer)
    data_monona_summer = preprocessing.normalize(data_monona_summer)
    data_kegonsa_summer = preprocessing.normalize(data_kegonsa_summer)
    data_waubesa_summer = preprocessing.normalize(data_waubesa_summer)
    data_wingra_summer = preprocessing.normalize(data_wingra_summer)

    data_mendota_summary_summer = preprocessing.normalize(data_mendota_summary_summer)
    data_monona_summary_summer = preprocessing.normalize(data_monona_summary_summer)
    data_kegonsa_summary_summer = preprocessing.normalize(data_kegonsa_summary_summer)
    data_waubesa_summary_summer = preprocessing.normalize(data_waubesa_summary_summer)
    data_wingra_summary_summer = preprocessing.normalize(data_wingra_summary_summer)

    print("Saving data set ... ")
    np.save(dest_path + "data_2014", data_2014)
    np.save(dest_path + "data_2015", data_2015)
    np.save(dest_path + "data_2016", data_2016)
    np.save(dest_path + "data_2017", data_2017)
    np.save(dest_path + "data_summary", data_summary)

    np.save(dest_path + "data_2014_summer", data_2014_summer)
    np.save(dest_path + "data_2015_summer", data_2015_summer)
    np.save(dest_path + "data_2016_summer", data_2016_summer)
    np.save(dest_path + "data_2017_summer", data_2017_summer)
    np.save(dest_path + "data_summary_summer", data_summary_summer)

    np.save(dest_path + "all_data", all_data)
    np.save(dest_path + "all_data_summer", all_data_summer)

    np.save(dest_path + "data_mendota", data_mendota)
    np.save(dest_path + "data_monona", data_monona)
    np.save(dest_path + "data_kegonsa", data_kegonsa)
    np.save(dest_path + "data_waubesa", data_waubesa)
    np.save(dest_path + "data_wingra", data_wingra)

    np.save(dest_path + "data_mendota_summary", data_mendota_summary)
    np.save(dest_path + "data_monona_summary", data_monona_summary)
    np.save(dest_path + "data_kegonsa_summary", data_kegonsa_summary)
    np.save(dest_path + "data_waubesa_summary", data_waubesa_summary)
    np.save(dest_path + "data_wingra_summary", data_wingra_summary)

    np.save(dest_path + "data_mendota_summer", data_mendota_summer)
    np.save(dest_path + "data_monona_summer", data_monona_summer)
    np.save(dest_path + "data_kegonsa_summer", data_kegonsa_summer)
    np.save(dest_path + "data_waubesa_summer", data_waubesa_summer)
    np.save(dest_path + "data_wingra_summer", data_wingra_summer)

    np.save(dest_path + "data_mendota_summary_summer", data_mendota_summary_summer)
    np.save(dest_path + "data_monona_summary_summer", data_monona_summary_summer)
    np.save(dest_path + "data_kegonsa_summary_summer", data_kegonsa_summary_summer)
    np.save(dest_path + "data_waubesa_summary_summer", data_waubesa_summary_summer)
    np.save(dest_path + "data_wingra_summary_summer", data_wingra_summary_summer)


# This method converts all date and times into a number between [0, 24). This way, we can use time as a quantitative
# measurement.
def convert_datetime_to_measurement(mat):
    for i in range(0, mat.shape[Constants.ROWS]):
        date_str = mat[i, Constants.DATE_TIME]
        date_arr = date_str.split("/")
        time = date_arr[2].split()[1].split(":")

        time_meas = int(time[0]) + (int(time[1]) / 60)

        mat[i, Constants.DATE_TIME] = str(time_meas)


if __name__ == "__main__": main()