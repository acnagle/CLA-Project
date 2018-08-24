import numpy as np
import os
import errno
import glob
import Constants


def main():
    np.set_printoptions(threshold=np.inf)  # prints a full matrix rather than an abbreviated matrix

    print("\n\t##### EXECUTING DATA_NORM.PY #####\n")

    # source directories
    path_matrices_no_na = "/Users/Alliot/documents/cla-project/data/matrices-no-na/original/"
    path_all_data = "/Users/Alliot/documents/cla-project/data/all-data-no-na/original/"

    # destination directories for normalized data
    dest_path_matrices_no_na_normalized = "/Users/Alliot/documents/cla-project/data/matrices-no-na/normalized/"
    dest_path_matrices_all_data_normalized = "/Users/Alliot/Documents/CLA-Project/Data/all-data-no-na/normalized/"

    # get all file paths in matrices and matrices-no-na directories
    files_matrices_no_na = [filename for filename in glob.glob(os.path.join(path_matrices_no_na, "*.csv"))]
    files_matrices_all_data = [filename for filename in glob.glob(os.path.join(path_all_data, "*.csv"))]

    # if dest_path_matrices_no_na_normalized path does not exist, create it
    if not os.path.exists(dest_path_matrices_no_na_normalized):
        try:
            os.makedirs(dest_path_matrices_no_na_normalized)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

    # define a location key matrix in order to quantize locations. For each location (row 0), define an integer to
    # represent that location (row 1)
    # mat = np.genfromtxt(open(path_all_data + "All_Data_matrix.csv", "rb"), delimiter=",",
    #                     dtype=(str, Constants.STR_LENGTH))
    # loc_key = np.transpose([np.empty(shape=(2, ), dtype=(str, Constants.STR_LENGTH))])
    # int_rep = 0     # holds integer to represent next location added to loc_key
    # for j in range(0, mat.shape[Constants.COLUMNS]):
    #     if mat[Constants.LOCATION, j] not in loc_key[0, :]:
    #         loc_key = np.hstack((loc_key, np.transpose([np.array([mat[Constants.LOCATION, j], str(int_rep)])])))
    #         int_rep = int_rep + 1
    #
    # mat = np.genfromtxt(open(path_matrices_no_na + "All_year_matrix.csv", "rb"), delimiter=",",
    #                     dtype=(str, Constants.STR_LENGTH))
    # for j in range(0, mat.shape[Constants.COLUMNS]):
    #     if mat[Constants.LOCATION, j] not in loc_key[0, :]:
    #         loc_key = np.hstack((loc_key, np.transpose([np.array([mat[Constants.LOCATION, j], str(int_rep)])])))
    #         int_rep = int_rep + 1
    #
    # loc_key = np.delete(loc_key, obj=0, axis=Constants.COLUMNS)

    # print("Processing Location Key ...")
    # matrix_to_file(loc_key, "Location Key.csv", "/Users/Alliot/Documents/CLA-Project/Data/")

    # normalize and store all matrices in path_matrices_no_na
    for filename in files_matrices_no_na:
        print("Processing file " + filename[65:] + " ...")
        mat = np.genfromtxt(open(filename, "rb"), delimiter=",", dtype=(str, Constants.STR_LENGTH))
        mat = remove_empty_entries(mat)

        convert_datetime_to_measurement(mat)
        # convert_locs_to_measurement(mat=mat, loc_key=loc_key)
        mat = np.delete(mat, obj=Constants.LOCATION, axis=Constants.ROWS)   # Remove location
        mat = mat.astype(float)
        normalize_data(mat=mat, first=0, last=Constants.NUM_ROWS_NO_LOC_NO_NA)
        matrix_to_file(mat=mat, filename=filename[65:], destination_folder=dest_path_matrices_no_na_normalized)

    # if dest_path_matrices_all_data_normalized does not exist, create it
    if not os.path.exists(dest_path_matrices_all_data_normalized):
        try:
            os.makedirs(dest_path_matrices_all_data_normalized)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

    # normalize and store all matrices in path_matrices_no_na
    for filename in files_matrices_all_data:
        print("Processing file " + filename[65:] + " ...")
        mat = np.genfromtxt(open(filename, "rb"), delimiter=",", dtype=(str, Constants.STR_LENGTH))
        mat = remove_empty_entries(mat)

        convert_datetime_to_measurement(mat)
        # convert_locs_to_measurement(mat=mat, loc_key=loc_key)
        mat = np.delete(mat, obj=Constants.LOCATION, axis=Constants.ROWS)   # Remove location
        mat = mat.astype(float)
        normalize_data(mat=mat, first=0, last=Constants.NUM_ROWS_W_IND_NO_LOC_ALL_DATA)
        matrix_to_file(mat=mat, filename=filename[65:], destination_folder=dest_path_matrices_all_data_normalized)

    print("\n")


# This method removes the empty entries (i.e. ",,,") located at the ends of the .csv files in the matrices_no_na
# directory. new_mat is the matrix with empty entries removed.
def remove_empty_entries(mat):
    new_mat = mat

    for j in range(0, mat.shape[Constants.COLUMNS]):
        if "" in new_mat[:, j]:
            new_mat = new_mat[:, 0:j]
            break

    return new_mat


# This method normalizes the data in a matrix by finding the largest value in each row and dividing each element in
# row by that value. This will cause each point in the data to be between the 0 and 1. mat is the matrix whose rows will
# be normalized. first is the first row that contains data that needs to be normalized. last is the last row
# (non-inclusive) that contains data that needs to be normalized.
def normalize_data(mat, first, last):
    for i in range(first, last):
        # determine the largest value in norm_arr
        max_val = np.amax(mat[i, :])

        if max_val != 0:    # sometimes a data matrix had no algal blooms, so a row of zeros could appear
            mat[i, :] = np.divide(mat[i, :], max_val)


# Writes a matrix to a csv file. mat is the matrix being written to a file. filename is the name of the .csv file.
# destination_folder is the path to the destination folder where the .csv file will be stored
def matrix_to_file(mat, filename, destination_folder):
    mat = mat.astype(str)
    file = open(destination_folder + filename, "w")

    for i in range(0, mat.shape[Constants.ROWS]):
        for j in range(0, mat.shape[Constants.COLUMNS]):
            if j < mat.shape[Constants.COLUMNS] - 1:
                file.write(mat[i, j] + ",")
            else:
                file.write(mat[i, j] + "\n")

    file.close()


# This method converts all date and times into a number between [0, 24). This way, we can use time as a quantitative
# measurement.
def convert_datetime_to_measurement(mat):
    for j in range(0, mat.shape[Constants.COLUMNS]):
        date_str = mat[Constants.DATE_TIME, j]
        date_arr = date_str.split("/")
        time = date_arr[2].split()[1].split(":")

        time_meas = int(time[0]) + (int(time[1]) / 60)

        mat[Constants.DATE_TIME, j] = str(time_meas)


# This method converts the locations where the measurements were gathered into an integer. In this way,
# we can quantify location. mat, the matrix of data, and loc_key, the matrix of locations and their
# corresponding integers, is passed into the method.
def convert_locs_to_measurement(mat, loc_key):
    for j in range(0, mat.shape[Constants.COLUMNS]):
        idx = np.where(mat[Constants.LOCATION, j] == loc_key[0, :])
        mat[Constants.LOCATION, j] = loc_key[1, idx[0][0]]


# This method subtracts out the mean for each feature over all data points in a matrix. mat is the matrix which will
# have the mean subtracted from it. new_mat is mat after the mean is subtracted and is the returned matrix.
def subtract_mean(mat):
    new_mat = mat.astype(float)  # mat_adj is adjusted so that the mean is subtracted out of mat
    mean = np.mean(new_mat, axis=Constants.COLUMNS)  # get an array of means for each dimension
    for i in range(0, new_mat.shape[Constants.ROWS]):
        for j in range(0, new_mat.shape[Constants.COLUMNS]):
            new_mat[i, j] = new_mat[i, j] - mean[i]

    return new_mat


if __name__ == "__main__": main()
