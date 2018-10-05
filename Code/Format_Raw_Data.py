import numpy as np
import Constants
from sklearn import preprocessing
import errno
import os


def main():
    np.set_printoptions(threshold=np.inf)  # prints a full matrix rather than an abbreviated matrix

    print("\n\t##### EXECUTING FORMAT_DATA.PY #####\n")

    # data paths
    file_paths = [
        "/Users/Alliot/Documents/CLA-Project/Data/CM2014_edit.csv",
        "/Users/Alliot/Documents/CLA-Project/Data/CM2015_edit.csv",
        "/Users/Alliot/Documents/CLA-Project/Data/CM2016_edit.csv",
        "/Users/Alliot/Documents/CLA-Project/Data/CM2017_edit.csv",
        "/Users/Alliot/Documents/cla-Project/data/algal_bloom_locations_summaries.csv"
    ]

    dest_path = "/Users/Alliot/Documents/CLA-Project/Data/data-sets/"

    # create dest_path if it does not exist
    if not os.path.exists(dest_path):
        try:
            os.makedirs(dest_path)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

    print("Reading in raw data ... ")

    # read in the data
    data_2014 = np.genfromtxt(
        open(file_paths[0], "rb"),
        delimiter=",",
        dtype=(str, Constants.STR_LENGTH),
        usecols=(1, 9, 10, 4, 5, 6, 7, 12, 13, 15, 14, 11),
        filling_values="NA",
        skip_header=1
    )

    data_2015 = np.genfromtxt(
        open(file_paths[1], "rb"),
        delimiter=",",
        dtype=(str, Constants.STR_LENGTH),
        usecols=(1, 14, 15, 6, 7, 8, 10, 17, 18, 20, 19, 16),
        filling_values="NA",
        skip_header=1
    )

    data_2016 = np.genfromtxt(
        open(file_paths[2], "rb"),
        delimiter=",",
        dtype=(str, Constants.STR_LENGTH),
        usecols=(1, 14, 15, 6, 7, 8, 10, 17, 18, 20, 19, 16),
        filling_values="NA",
        skip_header=1
    )

    data_2017 = np.genfromtxt(
        open(file_paths[3], "rb"),
        delimiter=",",
        dtype=(str, Constants.STR_LENGTH),
        usecols=(1, 14, 15, 6, 7, 8, 10, 17, 18, 20, 19, 16),
        filling_values="NA",
        skip_header=1
    )

    data_summary = np.genfromtxt(
        open(file_paths[4], "rb"),
        delimiter=",",
        dtype=(str, Constants.STR_LENGTH),
        usecols=(0, 20, 12, 13, 14, 16, 24, 25, 27, 26, 23, 6, 4, 5, 8),
        skip_header=1,
        invalid_raise=False
    )

    print("Cleaning the raw data ... ")
    data_2014 = clean_data(data_2014, summary=False)
    data_2015 = clean_data(data_2015, summary=False)
    data_2016 = clean_data(data_2016, summary=False)
    data_2017 = clean_data(data_2017, summary=False)
    data_summary = clean_data(data_summary, summary=True)

    print("Forming data sets from raw data ... ")
    data_2014_summer = build_summer_months(data_2014)
    data_2015_summer = build_summer_months(data_2015)
    data_2016_summer = build_summer_months(data_2016)
    data_2017_summer = build_summer_months(data_2017)
    data_summary_summer = build_summer_months(data_summary)

    all_data = np.concatenate((data_2014, data_2015, data_2016, data_2017), axis=Constants.ROWS)
    all_data_summer = np.concatenate(
        (data_2014_summer, data_2015_summer, data_2016_summer, data_2017_summer),
        axis=Constants.ROWS
    )

    data_mendota = build_location_matrix(all_data, "Mendota")
    data_monona = build_location_matrix(all_data, "Monona")
    data_kegonsa = build_location_matrix(all_data, "Kegonsa")
    data_waubesa = build_location_matrix(all_data, "Waubesa")
    data_wingra = build_location_matrix(all_data, "Wingra")

    data_mendota_summary = build_location_matrix(data_summary, "Mendota")
    data_monona_summary = build_location_matrix(data_summary, "Monona")
    data_kegonsa_summary = build_location_matrix(data_summary, "Kegonsa")
    data_waubesa_summary = build_location_matrix(data_summary, "Waubesa")
    data_wingra_summary = build_location_matrix(data_summary, "Wingra")

    data_mendota_summer = build_location_matrix(all_data_summer, "Mendota")
    data_monona_summer = build_location_matrix(all_data_summer, "Monona")
    data_kegonsa_summer = build_location_matrix(all_data_summer, "Kegonsa")
    data_waubesa_summer = build_location_matrix(all_data_summer, "Waubesa")
    data_wingra_summer = build_location_matrix(all_data_summer, "Wingra")

    data_mendota_summary_summer = build_location_matrix(data_summary_summer, "Mendota")
    data_monona_summary_summer = build_location_matrix(data_summary_summer, "Monona")
    data_kegonsa_summary_summer = build_location_matrix(data_summary_summer, "Kegonsa")
    data_waubesa_summary_summer = build_location_matrix(data_summary_summer, "Waubesa")
    data_wingra_summary_summer = build_location_matrix(data_summary_summer, "Wingra")

    print("Creating labels for each data set ... ")
    data_2014_labels = build_labels(data_2014)
    data_2015_labels = build_labels(data_2015)
    data_2016_labels = build_labels(data_2016)
    data_2017_labels = build_labels(data_2017)
    data_summary_labels = build_labels(data_summary)

    data_2014_summer_labels = build_labels(data_2014_summer)
    data_2015_summer_labels = build_labels(data_2015_summer)
    data_2016_summer_labels = build_labels(data_2016_summer)
    data_2017_summer_labels = build_labels(data_2017_summer)
    data_summary_summer_labels = build_labels(data_summary_summer)

    all_data_labels = build_labels(all_data)
    all_data_summer_labels = build_labels(all_data_summer)

    data_mendota_labels = build_labels(data_mendota)
    data_monona_labels = build_labels(data_monona)
    data_kegonsa_labels = build_labels(data_kegonsa)
    data_waubesa_labels = build_labels(data_waubesa)
    data_wingra_labels = build_labels(data_wingra)

    data_mendota_summary_labels = build_labels(data_mendota_summary)
    data_monona_summary_labels = build_labels(data_monona_summary)
    data_kegonsa_summary_labels = build_labels(data_kegonsa_summary)
    data_waubesa_summary_labels = build_labels(data_waubesa_summary)
    data_wingra_summary_labels = build_labels(data_wingra_summary)

    data_mendota_summer_labels = build_labels(data_mendota_summer)
    data_monona_summer_labels = build_labels(data_monona_summer)
    data_kegonsa_summer_labels = build_labels(data_kegonsa_summer)
    data_waubesa_summer_labels = build_labels(data_waubesa_summer)
    data_wingra_summer_labels = build_labels(data_wingra_summer)

    data_mendota_summary_summer_labels = build_labels(data_mendota_summary_summer)
    data_monona_summary_summer_labels = build_labels(data_monona_summary_summer)
    data_kegonsa_summary_summer_labels = build_labels(data_kegonsa_summary_summer)
    data_waubesa_summary_summer_labels = build_labels(data_waubesa_summary_summer)
    data_wingra_summary_summer_labels = build_labels(data_wingra_summary_summer)

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

    data_2014 = np.delete(data_2014, obj=Constants.LOCATION, axis=Constants.COLUMNS)  # Remove location
    data_2014 = np.delete(data_2014, obj=Constants.ALGAL_BLOOM_SHEEN_NO_LOC, axis=Constants.COLUMNS)  # Remove algal_bloom_sheen
    data_2014 = np.delete(data_2014, obj=Constants.ALGAL_BLOOMS_NO_LOC, axis=Constants.COLUMNS)  # Remove algal_blooms
    data_2014 = data_2014.astype(float)
    data_2015 = np.delete(data_2015, obj=Constants.LOCATION, axis=Constants.COLUMNS)
    data_2015 = np.delete(data_2015, obj=Constants.ALGAL_BLOOM_SHEEN_NO_LOC, axis=Constants.COLUMNS)
    data_2015 = np.delete(data_2015, obj=Constants.ALGAL_BLOOMS_NO_LOC, axis=Constants.COLUMNS)
    data_2015 = data_2015.astype(float)
    data_2016 = np.delete(data_2016, obj=Constants.LOCATION, axis=Constants.COLUMNS)
    data_2016 = np.delete(data_2016, obj=Constants.ALGAL_BLOOM_SHEEN_NO_LOC, axis=Constants.COLUMNS)
    data_2016 = np.delete(data_2016, obj=Constants.ALGAL_BLOOMS_NO_LOC, axis=Constants.COLUMNS)
    data_2016 = data_2016.astype(float)
    data_2017 = np.delete(data_2017, obj=Constants.LOCATION, axis=Constants.COLUMNS)
    data_2017 = np.delete(data_2017, obj=Constants.ALGAL_BLOOM_SHEEN_NO_LOC, axis=Constants.COLUMNS)
    data_2017 = np.delete(data_2017, obj=Constants.ALGAL_BLOOMS_NO_LOC, axis=Constants.COLUMNS)
    data_2017 = data_2017.astype(float)
    data_summary = np.delete(data_summary, obj=Constants.LOCATION, axis=Constants.COLUMNS)
    data_summary = np.delete(data_summary, obj=Constants.ALGAL_BLOOM_SHEEN_NO_LOC, axis=Constants.COLUMNS)
    data_summary = np.delete(data_summary, obj=Constants.ALGAL_BLOOMS_NO_LOC, axis=Constants.COLUMNS)
    data_summary = data_summary.astype(float)

    data_2014_summer = np.delete(data_2014_summer, obj=Constants.LOCATION, axis=Constants.COLUMNS)
    data_2014_summer = np.delete(data_2014_summer, obj=Constants.ALGAL_BLOOM_SHEEN_NO_LOC, axis=Constants.COLUMNS)
    data_2014_summer = np.delete(data_2014_summer, obj=Constants.ALGAL_BLOOMS_NO_LOC, axis=Constants.COLUMNS)
    data_2014_summer = data_2014_summer.astype(float)
    data_2015_summer = np.delete(data_2015_summer, obj=Constants.LOCATION, axis=Constants.COLUMNS)
    data_2015_summer = np.delete(data_2015_summer, obj=Constants.ALGAL_BLOOM_SHEEN_NO_LOC, axis=Constants.COLUMNS)
    data_2015_summer = np.delete(data_2015_summer, obj=Constants.ALGAL_BLOOMS_NO_LOC, axis=Constants.COLUMNS)
    data_2015_summer = data_2015_summer.astype(float)
    data_2016_summer = np.delete(data_2016_summer, obj=Constants.LOCATION, axis=Constants.COLUMNS)
    data_2016_summer = np.delete(data_2016_summer, obj=Constants.ALGAL_BLOOM_SHEEN_NO_LOC, axis=Constants.COLUMNS)
    data_2016_summer = np.delete(data_2016_summer, obj=Constants.ALGAL_BLOOMS_NO_LOC, axis=Constants.COLUMNS)
    data_2016_summer = data_2016_summer.astype(float)
    data_2017_summer = np.delete(data_2017_summer, obj=Constants.LOCATION, axis=Constants.COLUMNS)
    data_2017_summer = np.delete(data_2017_summer, obj=Constants.ALGAL_BLOOM_SHEEN_NO_LOC, axis=Constants.COLUMNS)
    data_2017_summer = np.delete(data_2017_summer, obj=Constants.ALGAL_BLOOMS_NO_LOC, axis=Constants.COLUMNS)
    data_2017_summer = data_2017_summer.astype(float)
    data_summary_summer = np.delete(data_summary_summer, obj=Constants.LOCATION, axis=Constants.COLUMNS)
    data_summary_summer = np.delete(data_summary_summer, obj=Constants.ALGAL_BLOOM_SHEEN_NO_LOC, axis=Constants.COLUMNS)
    data_summary_summer = np.delete(data_summary_summer, obj=Constants.ALGAL_BLOOMS_NO_LOC, axis=Constants.COLUMNS)
    data_summary_summer = data_summary_summer.astype(float)

    all_data = np.delete(all_data, obj=Constants.LOCATION, axis=Constants.COLUMNS)
    all_data = np.delete(all_data, obj=Constants.ALGAL_BLOOM_SHEEN_NO_LOC, axis=Constants.COLUMNS)
    all_data = np.delete(all_data, obj=Constants.ALGAL_BLOOMS_NO_LOC, axis=Constants.COLUMNS)
    all_data = all_data.astype(float)
    all_data_summer = np.delete(all_data_summer, obj=Constants.LOCATION, axis=Constants.COLUMNS)
    all_data_summer = np.delete(all_data_summer, obj=Constants.ALGAL_BLOOM_SHEEN_NO_LOC, axis=Constants.COLUMNS)
    all_data_summer = np.delete(all_data_summer, obj=Constants.ALGAL_BLOOMS_NO_LOC, axis=Constants.COLUMNS)
    all_data_summer = all_data_summer.astype(float)

    data_mendota = np.delete(data_mendota, obj=Constants.LOCATION, axis=Constants.COLUMNS)
    data_mendota = np.delete(data_mendota, obj=Constants.ALGAL_BLOOM_SHEEN_NO_LOC, axis=Constants.COLUMNS)
    data_mendota = np.delete(data_mendota, obj=Constants.ALGAL_BLOOMS_NO_LOC, axis=Constants.COLUMNS)
    data_mendota = data_mendota.astype(float)
    data_monona = np.delete(data_monona, obj=Constants.LOCATION, axis=Constants.COLUMNS)
    data_monona = np.delete(data_monona, obj=Constants.ALGAL_BLOOM_SHEEN_NO_LOC, axis=Constants.COLUMNS)
    data_monona = np.delete(data_monona, obj=Constants.ALGAL_BLOOMS_NO_LOC, axis=Constants.COLUMNS)
    data_monona = data_monona.astype(float)
    data_kegonsa = np.delete(data_kegonsa, obj=Constants.LOCATION, axis=Constants.COLUMNS)
    data_kegonsa = np.delete(data_kegonsa, obj=Constants.ALGAL_BLOOM_SHEEN_NO_LOC, axis=Constants.COLUMNS)
    data_kegonsa = np.delete(data_kegonsa, obj=Constants.ALGAL_BLOOMS_NO_LOC, axis=Constants.COLUMNS)
    data_kegonsa = data_kegonsa.astype(float)
    data_waubesa = np.delete(data_waubesa, obj=Constants.LOCATION, axis=Constants.COLUMNS)
    data_waubesa = np.delete(data_waubesa, obj=Constants.ALGAL_BLOOM_SHEEN_NO_LOC, axis=Constants.COLUMNS)
    data_waubesa = np.delete(data_waubesa, obj=Constants.ALGAL_BLOOMS_NO_LOC, axis=Constants.COLUMNS)
    data_waubesa = data_waubesa.astype(float)
    data_wingra = np.delete(data_wingra, obj=Constants.LOCATION, axis=Constants.COLUMNS)
    data_wingra = np.delete(data_wingra, obj=Constants.ALGAL_BLOOM_SHEEN_NO_LOC, axis=Constants.COLUMNS)
    data_wingra = np.delete(data_wingra, obj=Constants.ALGAL_BLOOMS_NO_LOC, axis=Constants.COLUMNS)
    data_wingra = data_wingra.astype(float)

    data_mendota_summary = np.delete(data_mendota_summary, obj=Constants.LOCATION, axis=Constants.COLUMNS)
    data_mendota_summary = np.delete(data_mendota_summary, obj=Constants.ALGAL_BLOOM_SHEEN_NO_LOC, axis=Constants.COLUMNS)
    data_mendota_summary = np.delete(data_mendota_summary, obj=Constants.ALGAL_BLOOMS_NO_LOC, axis=Constants.COLUMNS)
    data_mendota_summary = data_mendota_summary.astype(float)
    data_monona_summary = np.delete(data_monona_summary, obj=Constants.LOCATION, axis=Constants.COLUMNS)
    data_monona_summary = np.delete(data_monona_summary, obj=Constants.ALGAL_BLOOM_SHEEN_NO_LOC, axis=Constants.COLUMNS)
    data_monona_summary = np.delete(data_monona_summary, obj=Constants.ALGAL_BLOOMS_NO_LOC, axis=Constants.COLUMNS)
    data_monona_summary = data_monona_summary.astype(float)
    data_kegonsa_summary = np.delete(data_kegonsa_summary, obj=Constants.LOCATION, axis=Constants.COLUMNS)
    data_kegonsa_summary = np.delete(data_kegonsa_summary, obj=Constants.ALGAL_BLOOM_SHEEN_NO_LOC, axis=Constants.COLUMNS)
    data_kegonsa_summary = np.delete(data_kegonsa_summary, obj=Constants.ALGAL_BLOOM_SHEEN_NO_LOC, axis=Constants.COLUMNS)
    data_kegonsa_summary = data_kegonsa_summary.astype(float)
    data_waubesa_summary = np.delete(data_waubesa_summary, obj=Constants.LOCATION, axis=Constants.COLUMNS)
    data_waubesa_summary = np.delete(data_waubesa_summary, obj=Constants.ALGAL_BLOOM_SHEEN_NO_LOC, axis=Constants.COLUMNS)
    data_waubesa_summary = np.delete(data_waubesa_summary, obj=Constants.ALGAL_BLOOMS_NO_LOC, axis=Constants.COLUMNS)
    data_waubesa_summary = data_waubesa_summary.astype(float)
    data_wingra_summary = np.delete(data_wingra_summary, obj=Constants.LOCATION, axis=Constants.COLUMNS)
    data_wingra_summary = np.delete(data_wingra_summary, obj=Constants.ALGAL_BLOOM_SHEEN_NO_LOC, axis=Constants.COLUMNS)
    data_wingra_summary = np.delete(data_wingra_summary, obj=Constants.ALGAL_BLOOMS_NO_LOC, axis=Constants.COLUMNS)
    data_wingra_summary = data_wingra_summary.astype(float)

    data_mendota_summer = np.delete(data_mendota_summer, obj=Constants.LOCATION, axis=Constants.COLUMNS)
    data_mendota_summer = np.delete(data_mendota_summer, obj=Constants.ALGAL_BLOOM_SHEEN_NO_LOC, axis=Constants.COLUMNS)
    data_mendota_summer = np.delete(data_mendota_summer, obj=Constants.ALGAL_BLOOMS_NO_LOC, axis=Constants.COLUMNS)
    data_mendota_summer = data_mendota_summer.astype(float)
    data_monona_summer = np.delete(data_monona_summer, obj=Constants.LOCATION, axis=Constants.COLUMNS)
    data_monona_summer = np.delete(data_monona_summer, obj=Constants.ALGAL_BLOOM_SHEEN_NO_LOC, axis=Constants.COLUMNS)
    data_monona_summer = np.delete(data_monona_summer, obj=Constants.ALGAL_BLOOMS_NO_LOC, axis=Constants.COLUMNS)
    data_monona_summer = data_monona_summer.astype(float)
    data_kegonsa_summer = np.delete(data_kegonsa_summer, obj=Constants.LOCATION, axis=Constants.COLUMNS)
    data_kegonsa_summer = np.delete(data_kegonsa_summer, obj=Constants.ALGAL_BLOOM_SHEEN_NO_LOC, axis=Constants.COLUMNS)
    data_kegonsa_summer = np.delete(data_kegonsa_summer, obj=Constants.ALGAL_BLOOMS_NO_LOC, axis=Constants.COLUMNS)
    data_kegonsa_summer = data_kegonsa_summer.astype(float)
    data_waubesa_summer = np.delete(data_waubesa_summer, obj=Constants.LOCATION, axis=Constants.COLUMNS)
    data_waubesa_summer = np.delete(data_waubesa_summer, obj=Constants.ALGAL_BLOOM_SHEEN_NO_LOC, axis=Constants.COLUMNS)
    data_waubesa_summer = np.delete(data_waubesa_summer, obj=Constants.ALGAL_BLOOMS_NO_LOC, axis=Constants.COLUMNS)
    data_waubesa_summer = data_waubesa_summer.astype(float)
    data_wingra_summer = np.delete(data_wingra_summer, obj=Constants.LOCATION, axis=Constants.COLUMNS)
    data_wingra_summer = np.delete(data_wingra_summer, obj=Constants.ALGAL_BLOOM_SHEEN_NO_LOC, axis=Constants.COLUMNS)
    data_wingra_summer = np.delete(data_wingra_summer, obj=Constants.ALGAL_BLOOMS_NO_LOC, axis=Constants.COLUMNS)
    data_wingra_summer = data_wingra_summer.astype(float)

    data_mendota_summary_summer = np.delete(data_mendota_summary_summer, obj=Constants.LOCATION, axis=Constants.COLUMNS)
    data_mendota_summary_summer = np.delete(data_mendota_summary_summer, obj=Constants.ALGAL_BLOOM_SHEEN_NO_LOC, axis=Constants.COLUMNS)
    data_mendota_summary_summer = np.delete(data_mendota_summary_summer, obj=Constants.ALGAL_BLOOMS_NO_LOC, axis=Constants.COLUMNS)
    data_mendota_summary_summer = data_mendota_summary_summer.astype(float)
    data_monona_summary_summer = np.delete(data_monona_summary_summer, obj=Constants.LOCATION, axis=Constants.COLUMNS)
    data_monona_summary_summer = np.delete(data_monona_summary_summer, obj=Constants.ALGAL_BLOOM_SHEEN_NO_LOC, axis=Constants.COLUMNS)
    data_monona_summary_summer = np.delete(data_monona_summary_summer, obj=Constants.ALGAL_BLOOMS_NO_LOC, axis=Constants.COLUMNS)
    data_monona_summary_summer = data_monona_summary_summer.astype(float)
    data_kegonsa_summary_summer = np.delete(data_kegonsa_summary_summer, obj=Constants.LOCATION, axis=Constants.COLUMNS)
    data_kegonsa_summary_summer = np.delete(data_kegonsa_summary_summer, obj=Constants.ALGAL_BLOOM_SHEEN_NO_LOC, axis=Constants.COLUMNS)
    data_kegonsa_summary_summer = np.delete(data_kegonsa_summary_summer, obj=Constants.ALGAL_BLOOMS_NO_LOC, axis=Constants.COLUMNS)
    data_kegonsa_summary_summer = data_kegonsa_summary_summer.astype(float)
    data_waubesa_summary_summer = np.delete(data_waubesa_summary_summer, obj=Constants.LOCATION, axis=Constants.COLUMNS)
    data_waubesa_summary_summer = np.delete(data_waubesa_summary_summer, obj=Constants.ALGAL_BLOOM_SHEEN_NO_LOC, axis=Constants.COLUMNS)
    data_waubesa_summary_summer = np.delete(data_waubesa_summary_summer, obj=Constants.ALGAL_BLOOMS_NO_LOC, axis=Constants.COLUMNS)
    data_waubesa_summary_summer = data_waubesa_summary_summer.astype(float)
    data_wingra_summary_summer = np.delete(data_wingra_summary_summer, obj=Constants.LOCATION, axis=Constants.COLUMNS)
    data_wingra_summary_summer = np.delete(data_wingra_summary_summer, obj=Constants.ALGAL_BLOOM_SHEEN_NO_LOC, axis=Constants.COLUMNS)
    data_wingra_summary_summer = np.delete(data_wingra_summary_summer, obj=Constants.ALGAL_BLOOMS_NO_LOC, axis=Constants.COLUMNS)
    data_wingra_summary_summer = data_wingra_summary_summer.astype(float)

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

    print("Saving data sets ... ")
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

    print("Saving labels ... ")
    np.save(dest_path + "data_2014_labels", data_2014_labels)
    np.save(dest_path + "data_2015_labels", data_2015_labels)
    np.save(dest_path + "data_2016_labels", data_2016_labels)
    np.save(dest_path + "data_2017_labels", data_2017_labels)
    np.save(dest_path + "data_summary_labels", data_summary_labels)

    np.save(dest_path + "data_2014_summer_labels", data_2014_summer_labels)
    np.save(dest_path + "data_2015_summer_labels", data_2015_summer_labels)
    np.save(dest_path + "data_2016_summer_labels", data_2016_summer_labels)
    np.save(dest_path + "data_2017_summer_labels", data_2017_summer_labels)
    np.save(dest_path + "data_summary_summer_labels", data_summary_summer_labels)

    np.save(dest_path + "all_data_labels", all_data_labels)
    np.save(dest_path + "all_data_summer_labels", all_data_summer_labels)

    np.save(dest_path + "data_mendota_labels", data_mendota_labels)
    np.save(dest_path + "data_monona_labels", data_monona_labels)
    np.save(dest_path + "data_kegonsa_labels", data_kegonsa_labels)
    np.save(dest_path + "data_waubesa_labels", data_waubesa_labels)
    np.save(dest_path + "data_wingra_labels", data_wingra_labels)

    np.save(dest_path + "data_mendota_summary_labels", data_mendota_summary_labels)
    np.save(dest_path + "data_monona_summary_labels", data_monona_summary_labels)
    np.save(dest_path + "data_kegonsa_summary_labels", data_kegonsa_summary_labels)
    np.save(dest_path + "data_waubesa_summary_labels", data_waubesa_summary_labels)
    np.save(dest_path + "data_wingra_summary_labels", data_wingra_summary_labels)

    np.save(dest_path + "data_mendota_summer_labels", data_mendota_summer_labels)
    np.save(dest_path + "data_monona_summer_labels", data_monona_summer_labels)
    np.save(dest_path + "data_kegonsa_summer_labels", data_kegonsa_summer_labels)
    np.save(dest_path + "data_waubesa_summer_labels", data_waubesa_summer_labels)
    np.save(dest_path + "data_wingra_summer_labels", data_wingra_summer_labels)

    np.save(dest_path + "data_mendota_summary_summer_labels", data_mendota_summary_summer_labels)
    np.save(dest_path + "data_monona_summary_summer_labels", data_monona_summary_summer_labels)
    np.save(dest_path + "data_kegonsa_summary_summer_labels", data_kegonsa_summary_summer_labels)
    np.save(dest_path + "data_waubesa_summary_summer_labels", data_waubesa_summary_summer_labels)
    np.save(dest_path + "data_wingra_summary_summer_labels", data_wingra_summary_summer_labels)


# This method converts all date and times into a number between [0, 24). This way, we can use time as a quantitative
# measurement.
def convert_datetime_to_measurement(mat):
    for i in range(0, mat.shape[Constants.ROWS]):
        date_str = mat[i, Constants.DATE_TIME]
        date_arr = date_str.split("/")
        time = date_arr[2].split()[1].split(":")

        time_meas = int(time[0]) + (int(time[1]) / 60)

        mat[i, Constants.DATE_TIME] = str(time_meas)


# This matrix creates and returns the numpy label vector for a particular data set
def build_labels(mat):
    labels = mat[:, Constants.ALGAL_BLOOM_SHEEN]

    return labels.astype(int)


# This method forms a matrix for a specific location across all years. mat is the matrix which will be sifted through to
# find the data points matching with the string loc. loc is the name of the location. mat_loc is matrix created for a
# specified location and is returned.
def build_location_matrix(mat, loc):
    mat_loc = [np.empty((mat.shape[Constants.COLUMNS], ))]

    for i in range(0, mat.shape[Constants.ROWS]):
        if loc in mat[i, Constants.LOCATION]:
            mat_loc = np.vstack([mat_loc, [mat[i, :]]])

    # remove the empty column used to initialize summer_mat
    mat_loc = np.delete(mat_loc, obj=0, axis=Constants.ROWS)

    return mat_loc


# This method takes a matrix and cleans it by removing any feature vectors with "NA", "FALSE", "TRUE", "undefined", or
# empty entries. new_mat is the clean version of mat, and is returned. summary is a boolean flag to indicate whether
# mat is the summary matrix
def clean_data(mat, summary):
    new_mat = np.empty(shape=(mat.shape[Constants.ROWS], mat.shape[Constants.COLUMNS]), dtype=(str, Constants.STR_LENGTH))

    idx = 0
    if not summary:
        for i in range(0, mat.shape[Constants.ROWS]):
            if ("NA" not in mat[i, :] and "" not in mat[i, :] and "FALSE" not in mat[i, :] and
                    "TRUE" not in mat[i, :] and "undefined" not in mat[i, :]):
                new_mat[idx, Constants.LOCATION] = mat[i, 0]                      # Locations
                new_mat[idx, Constants.DATE_TIME] = mat[i, 1] + " " + mat[i, 2]   # Dates and Time (24-hour)
                new_mat[idx, Constants.ALGAL_BLOOMS] = mat[i, 3]                  # algalBlooms
                new_mat[idx, Constants.ALGAL_BLOOM_SHEEN] = mat[i, 4]             # algalBloomSheen
                new_mat[idx, Constants.BATHER_LOAD] = mat[i, 5]                   # batherLoad
                new_mat[idx, Constants.PLANT_DEBRIS] = mat[i, 6]                  # plantDebris
                new_mat[idx, Constants.WATER_APPEARANCE] = mat[i, 7]              # waterAppearance
                new_mat[idx, Constants.WATER_FOWL_PRESENCE] = mat[i, 8]           # waterfowlPresence
                new_mat[idx, Constants.WAVE_INTENSITY] = mat[i, 9]                # waveIntensity
                new_mat[idx, Constants.WATER_TEMP] = mat[i, 10]                   # waterTemp
                new_mat[idx, Constants.TURBIDITY] = mat[i, 11]                    # turbidity

                # if the qualitative measurement does not conform to the convention for quantization, delete the sample
                for j in range(Constants.ALGAL_BLOOMS, Constants.WATER_TEMP):
                    if j != Constants.ALGAL_BLOOM_SHEEN:  # ignore algalBloomSheen measurement
                        if float(new_mat[idx, j]) == 0:
                            # new_mat[idx, j] = "1"
                            new_mat = np.delete(new_mat, obj=idx, axis=Constants.ROWS)
                            idx -= 1
                            break

                idx += 1

        new_mat = np.delete(new_mat, obj=mat.shape[Constants.COLUMNS]-1, axis=Constants.COLUMNS)

    else:
        for i in range(0, mat.shape[Constants.ROWS]):
            if "" not in mat[i, :] and "NA" not in mat[i, :] and "FALSE" not in mat[i, :] and "#VALUE" not in mat[i, :]:
                new_mat[idx, Constants.LOCATION] = mat[i, 0]                # Locations
                new_mat[idx, Constants.DATE_TIME] = mat[i, 1]               # Date and Time (24-hour)
                new_mat[idx, Constants.ALGAL_BLOOMS] = mat[i, 2]            # algalBlooms
                new_mat[idx, Constants.ALGAL_BLOOM_SHEEN] = mat[i, 3]       # algalBloomSheen
                new_mat[idx, Constants.BATHER_LOAD] = mat[i, 4]             # batherLoad
                new_mat[idx, Constants.PLANT_DEBRIS] = mat[i, 5]            # plantDebris
                new_mat[idx, Constants.WATER_APPEARANCE] = mat[i, 6]        # waterAppearance
                new_mat[idx, Constants.WATER_FOWL_PRESENCE] = mat[i, 7]     # waterfowlPresence
                new_mat[idx, Constants.WAVE_INTENSITY] = mat[i, 8]          # waveIntensity
                new_mat[idx, Constants.WATER_TEMP] = mat[i, 9]              # waterTemp
                new_mat[idx, Constants.TURBIDITY] = mat[i, 10]              # turbidity
                new_mat[idx, Constants.AIR_TEMP] = mat[i, 11]               # airTemp
                new_mat[idx, Constants.PRCP_24_HRS] = mat[i, 12]            # prcp_24rs
                new_mat[idx, Constants.PRCP_48_HRS] = mat[i, 13]            # prcp_48hrs
                new_mat[idx, Constants.WINDSPEED_AVG_24_HRS] = mat[i, 14]   # windspeed_avg_24hr

                # if the qualitative measurement does not conform to the convention for quantization, delete the sample
                for j in range(Constants.ALGAL_BLOOMS, Constants.WATER_TEMP):
                    if j != Constants.ALGAL_BLOOM_SHEEN:  # ignore algalBloomSheen measurement
                        if float(new_mat[idx, j]) == 0:
                            # new_mat[idx, j] = "1"
                            new_mat = np.delete(new_mat, obj=idx, axis=Constants.ROWS)
                            idx -= 1
                            break

                idx = idx + 1

    new_mat = new_mat[0:idx, :]

    return new_mat


# This method takes takes in a matrix representing a year's worth of data and turns it into a matrix with only the
# summer months (June, July, August). mat a year's worth of data points. summer_mat is the data points only for the
# summer of that year; summer_mat is returned
def build_summer_months(mat):
    summer_mat = [np.empty((mat.shape[Constants.COLUMNS], ))]

    for i in range(0, mat.shape[Constants.ROWS]):
        try:
            curr_month = int(mat[i, Constants.DATE_TIME][0:2])
        except ValueError:
            curr_month = int(mat[i, Constants.DATE_TIME][0:1])

        if (curr_month == 6) or (curr_month == 7) or (curr_month == 8):
            summer_mat = np.vstack([summer_mat, [mat[i, :]]])

    # remove the empty column used to initialize summer_mat
    summer_mat = np.delete(summer_mat, obj=0, axis=Constants.ROWS)

    return summer_mat


if __name__ == "__main__": main()
