import numpy as np
import csv
import os
import errno
import glob
import Constants


def main():
    np.set_printoptions(threshold=np.inf)  # prints a full matrix rather than an abbreviated matrix

    print("\n\t##### EXECUTING ALL_DATA_MATRIX_BUILD.PY #####")

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
                                 usecols=(1, 14, 15, 6, 7, 8, 10, 17, 18, 20, 19, 16), filling_values="NA")
    filename_all_data = "All_Data"
    # mat = build_matrix(data_reader)
    mat = remove_empty_entries(mat)
    matrix_to_file(mat, filename_all_data + "_matrix.csv", orig_path)

    # Build monthly matrices (March (03) through September (09))
    print("Building month matrices ...")
    mat_03 = np.transpose([np.empty((Constants.NUM_ROWS_W_IND_ALL_DATA, ))])
    mat_04 = np.transpose([np.empty((Constants.NUM_ROWS_W_IND_ALL_DATA, ))])
    mat_05 = np.transpose([np.empty((Constants.NUM_ROWS_W_IND_ALL_DATA, ))])
    mat_06 = np.transpose([np.empty((Constants.NUM_ROWS_W_IND_ALL_DATA, ))])
    mat_07 = np.transpose([np.empty((Constants.NUM_ROWS_W_IND_ALL_DATA, ))])
    mat_08 = np.transpose([np.empty((Constants.NUM_ROWS_W_IND_ALL_DATA, ))])
    mat_09 = np.transpose([np.empty((Constants.NUM_ROWS_W_IND_ALL_DATA, ))])

    for i in range(0, mat.shape[Constants.COLUMNS]-1):
        month = mat[1, i][:1]
        if month == "3":
            mat_03 = np.hstack([mat_03, np.transpose([mat[:, i]])])
        if month == "4":
            mat_04 = np.hstack([mat_04, np.transpose([mat[:, i]])])
        if month == "5":
            mat_05 = np.hstack([mat_05, np.transpose([mat[:, i]])])
        if month == "6":
            mat_06 = np.hstack([mat_06, np.transpose([mat[:, i]])])
        if month == "7":
            mat_07 = np.hstack([mat_07, np.transpose([mat[:, i]])])
        if month == "8":
            mat_08 = np.hstack([mat_08, np.transpose([mat[:, i]])])
        if month == "9":
            mat_09 = np.hstack([mat_09, np.transpose([mat[:, i]])])

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


# Build a matrix from .csv file. data_reader allow direct access to the .csv file
# for algal_bloom_locations_summaries.csv. This method returns a matrix called mat, which is the new matrix that stores
# relevant information for analysis
def build_matrix(data_reader):
    # remove unnecessary lines
    data_reader.__next__()

    mat = np.empty([Constants.NUM_ROWS_W_IND_ALL_DATA, 3857], dtype=(str, Constants.STR_LENGTH))

    col_index = 0   # used to index columns in mat

    for row in data_reader:
        if row[0] != "":
            # test_row is an array of the relevant data. I put this data in this array, so that "NA" and "FALSE" can
            # be searched for in each row. If "NA" or "FALSE" are found, that row will not be used in the data
            test_row = [row[0], row[20], row[12], row[13], row[14], row[16], row[24], row[25], row[27], row[26],
                        row[6], row[23], row[4], row[5], row[8]]
            if any("NA" in s for s in test_row):
                continue
            elif any("FALSE" in s for s in test_row):
                continue
            elif row[23] == "" or row[26] == "" or row[6] == "":     # turbidity is blank
                continue

            # # adjust column entries for poor water quality flag
            # if float(row[23]) <= 50:
            #     mat[Constants.NUM_ROWS_W_IND_ALL_DATA, col_index] = 1
            # else:
            #     mat[Constants.NUM_ROWS_W_IND_ALL_DATA, col_index] = 0

            mat[0, col_index] = row[0]          # Locations (Specific locations for each lake)
            mat[1, col_index] = row[20]         # Date and Time (24-hour)
            mat[2, col_index] = row[12]         # algalBlooms
            mat[3, col_index] = row[13]         # algalBloomSheen
            mat[4, col_index] = row[14]         # batherLoad
            mat[5, col_index] = row[16]         # plantDebris
            mat[6, col_index] = row[24]         # waterAppearance
            mat[7, col_index] = row[25]         # waterfowlPresence
            mat[8, col_index] = row[27]         # waveIntensity
            mat[9, col_index] = row[26]         # waterTemp
            mat[10, col_index] = row[23]        # turbidity
            mat[11, col_index] = row[6]         # airTemp
            mat[12, col_index] = row[4]         # prcp_24rs
            mat[13, col_index] = row[5]         # prcp_48hrs
            mat[14, col_index] = row[8]         # windspeed_avg_24hr

            # adjust data entries so that qualitative measurements (except algalBloomSheen) have values [1, 3]
            for i in range(Constants.FIRST_ROW, 9):
                if i != Constants.ALGAL_BLOOM_SHEEN:          # ignore algalBloomSheen measurement
                    if float(mat[i, col_index]) == 0:
                        mat[i, col_index] = "1"

            col_index = col_index + 1

    return mat


# This method removes the empty entries (i.e. ",,,") located at the ends of the .csv files. new_mat is the matrix
# with empty entries removed.
def remove_empty_entries(mat):
    new_mat = mat

    for j in range(0, mat.shape[Constants.COLUMNS]):
        if "" in mat[:, j]:
            new_mat = mat[:, 0:j]
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
