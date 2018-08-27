import numpy as np
import os
import errno
import glob
import Constants


def main():
    np.set_printoptions(threshold=np.inf)  # prints a full matrix rather than an abbreviated matrix

    print("\n\t##### EXECUTING ALL_DATA_MATRIX_BUILD.PY #####\n")

    src_path = "/Users/Alliot/documents/cla-project/data/algal_bloom_locations_summaries.csv"
    orig_path = "/Users/Alliot/documents/cla-project/data/all-data-no-na/original/"     # path to original data folder

    # if orig_path does not exist, create it
    if not os.path.exists(orig_path):
        try:
            os.makedirs(orig_path)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

    # read in .csv file and store into mat (matrix)
    # file = open(src_path, newline="")
    # data_reader = csv.reader(file)

    print("Building All Data matrix ...")
    mat_all_data = np.genfromtxt(open(src_path, "rb"), delimiter=",", dtype=(str, Constants.STR_LENGTH),
                                 usecols=(0, 20, 12, 13, 14, 16, 24, 25, 27, 26, 23, 6, 4, 5, 8), filling_values="NA",
                                 invalid_raise=False, skip_header=1, missing_values=("NA", "", "FALSE", "#VALUE!"))

    filename_all_data = "All_Data"
    mat_all_data = build_matrix(mat_all_data)
    mat_all_data = remove_empty_entries(mat_all_data)
    matrix_to_file(mat_all_data, filename_all_data + "_matrix.csv", orig_path)

    # Build monthly matrices (March (03) through September (09))
    print("Building month matrices ...")
    mat_03 = np.transpose([np.empty((Constants.NUM_ROWS_W_IND_ALL_DATA, ))])
    mat_04 = np.transpose([np.empty((Constants.NUM_ROWS_W_IND_ALL_DATA, ))])
    mat_05 = np.transpose([np.empty((Constants.NUM_ROWS_W_IND_ALL_DATA, ))])
    mat_06 = np.transpose([np.empty((Constants.NUM_ROWS_W_IND_ALL_DATA, ))])
    mat_07 = np.transpose([np.empty((Constants.NUM_ROWS_W_IND_ALL_DATA, ))])
    mat_08 = np.transpose([np.empty((Constants.NUM_ROWS_W_IND_ALL_DATA, ))])
    mat_09 = np.transpose([np.empty((Constants.NUM_ROWS_W_IND_ALL_DATA, ))])

    for i in range(0, mat_all_data.shape[Constants.COLUMNS]-1):
        month = mat_all_data[1, i][:1]
        if month == "3":
            mat_03 = np.hstack([mat_03, np.transpose([mat_all_data[:, i]])])
        if month == "4":
            mat_04 = np.hstack([mat_04, np.transpose([mat_all_data[:, i]])])
        if month == "5":
            mat_05 = np.hstack([mat_05, np.transpose([mat_all_data[:, i]])])
        if month == "6":
            mat_06 = np.hstack([mat_06, np.transpose([mat_all_data[:, i]])])
        if month == "7":
            mat_07 = np.hstack([mat_07, np.transpose([mat_all_data[:, i]])])
        if month == "8":
            mat_08 = np.hstack([mat_08, np.transpose([mat_all_data[:, i]])])
        if month == "9":
            mat_09 = np.hstack([mat_09, np.transpose([mat_all_data[:, i]])])

    # Remove zero column in beginning of each month matrix
    mat_03 = mat_03[:, 1:]
    mat_04 = mat_04[:, 1:]
    mat_05 = mat_05[:, 1:]
    mat_06 = mat_06[:, 1:]
    mat_07 = mat_07[:, 1:]
    mat_08 = mat_08[:, 1:]
    mat_09 = mat_09[:, 1:]

    matrix_to_file(mat_03, "month_03_" + filename_all_data + "_matrix.csv", orig_path)
    matrix_to_file(mat_04, "month_04_" + filename_all_data + "_matrix.csv", orig_path)
    matrix_to_file(mat_05, "month_05_" + filename_all_data + "_matrix.csv", orig_path)
    matrix_to_file(mat_06, "month_06_" + filename_all_data + "_matrix.csv", orig_path)
    matrix_to_file(mat_07, "month_07_" + filename_all_data + "_matrix.csv", orig_path)
    matrix_to_file(mat_08, "month_08_" + filename_all_data + "_matrix.csv", orig_path)
    matrix_to_file(mat_09, "month_09_" + filename_all_data + "_matrix.csv", orig_path)

    summer_month_paths_all_data = []

    # get filenames of all files for the summer months: June (06), July (07), and August (08)
    for filename in glob.glob(os.path.join(orig_path, "*.csv")):
        if "summer" in filename:
            if "Wingra" in filename or "Mendota" in filename or "Monona" in filename or \
                    "Kegonsa" in filename or "Waubesa" in filename:
                summer_month_paths_all_data.append(filename)

    # put the filenames in chronological order
    summer_month_paths_all_data = np.sort(summer_month_paths_all_data)

    # create matrices the summer months of each year
    all_data_summer = np.empty(shape=(Constants.NUM_ROWS_W_IND_ALL_DATA, 0))

    for path in summer_month_paths_all_data:
        all_data_summer = np.hstack((all_data_summer, np.genfromtxt(open(path, "rb"), delimiter=",", dtype="str")))

    # write summer matrix to .csv file
    matrix_to_file(all_data_summer, "All_Data_summer_matrix.csv", orig_path)

    print("\n")


# Construct a matrix in a useful form using algal_bloom_locations_summaries.csv. mat is the matrix generated from
# algal_bloom_locations_summaries.csv. This method returns new_mat, which is the newly constructed matrix.
def build_matrix(mat):
    mat = np.transpose(mat)
    new_mat = np.empty(shape=(Constants.NUM_ROWS_W_IND_ALL_DATA, mat.shape[Constants.COLUMNS]),
                       dtype=(str, Constants.STR_LENGTH))

    idx = 0
    for i in range(0, mat.shape[Constants.COLUMNS]):
        if "" not in mat[:, i] and "NA" not in mat[:, i] and "FALSE" not in mat[:, i]:

            new_mat[Constants.LOCATION, idx] = mat[0, i]                # Locations
            new_mat[Constants.DATE_TIME, idx] = mat[1, i]               # Date and Time (24-hour)
            new_mat[Constants.ALGAL_BLOOMS, idx] = mat[2, i]            # algalBlooms
            new_mat[Constants.ALGAL_BLOOM_SHEEN, idx] = mat[3, i]       # algalBloomSheen
            new_mat[Constants.BATHER_LOAD, idx] = mat[4, i]             # batherLoad
            new_mat[Constants.PLANT_DEBRIS, idx] = mat[5, i]            # plantDebris
            new_mat[Constants.WATER_APPEARANCE, idx] = mat[6, i]        # waterAppearance
            new_mat[Constants.WATER_FOWL_PRESENCE, idx] = mat[7, i]     # waterfowlPresence
            new_mat[Constants.WAVE_INTENSITY, idx] = mat[8, i]          # waveIntensity
            new_mat[Constants.WATER_TEMP, idx] = mat[9, i]              # waterTemp
            new_mat[Constants.TURBIDITY, idx] = mat[10, i]              # turbidity
            new_mat[Constants.AIR_TEMP, idx] = mat[11, i]               # airTemp
            new_mat[Constants.PRCP_24_HRS, idx] = mat[12, i]            # prcp_24rs
            new_mat[Constants.PRCP_48_HRS, idx] = mat[13, i]            # prcp_48hrs
            new_mat[Constants.WINDSPEED_AVG_24_HRS, idx] = mat[14, i]   # windspeed_avg_24hr

            # adjust data entries so that qualitative measurements (except algalBloomSheen) have values [1, 3]
            for j in range(Constants.FIRST_ROW, Constants.WATER_TEMP):
                if j != Constants.ALGAL_BLOOM_SHEEN:          # ignore algalBloomSheen measurement
                    if float(new_mat[j, idx]) == 0:
                        new_mat[j, idx] = "1"

            idx = idx + 1

    return new_mat


# This method removes the empty entries (i.e. ",,,") located at the ends of the .csv files. new_mat is the matrix
# with empty entries removed.
def remove_empty_entries(mat):
    new_mat = mat

    for j in range(0, mat.shape[Constants.COLUMNS]):
        if "" in new_mat[:, j]:
            new_mat = new_mat[:, 0:j]
            break

    return new_mat


# Writes a matrix to a csv file. mat is the matrix being written to a file. filename is the name of the .csv file.
# destination_folder is the path to the destination folder where the .csv file will be stored
def matrix_to_file(mat, filename, destination_folder):
    file = open(destination_folder + filename, "w")

    for i in range(0, mat.shape[Constants.ROWS]):
        for j in range(0, mat.shape[Constants.COLUMNS]):
            if j < mat.shape[Constants.COLUMNS]-1:
                file.write(mat[i, j] + ",")
            else:
                file.write(mat[i, j] + "\n")

    file.close()


if __name__ == "__main__": main()
