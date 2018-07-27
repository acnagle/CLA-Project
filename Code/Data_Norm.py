import numpy as np
import os
import errno
import glob


def main():
    np.set_printoptions(threshold=np.inf)  # prints a full matrix rather than an abbreviated matrix

    print("\n\t##### EXECUTING DATA_NORM.PY #####")

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
    # represent that location (row 1). Note: there are exactly 78 locations
    num_locs = 78
    mat = np.genfromtxt(open(path_all_data + "All_Data_matrix.csv", "rb"), delimiter=",", dtype=(str, 15))
    loc_key = np.empty((2, num_locs), dtype=(str, 15))
    int_rep = 0     # holds integer to represent next location added to loc_key
    for j in range(0, mat.shape[1]):
        if mat[0, j] not in loc_key[0, :]:
            loc_key[0, int_rep] = mat[0, j]
            loc_key[1, int_rep] = str(int_rep)
            int_rep = int_rep + 1

    # add more locations
    loc_key[0, 77] = "MendotaPier18"
    loc_key[1, 77] = "77"

    print("Processing Location Key ...")
    matrix_to_file(loc_key, "Location Key.csv", "/Users/Alliot/Documents/CLA-Project/Data/")

    # normalize and store all matrices in path_matrices_no_na
    for filename in files_matrices_no_na:
        print("Processing file " + filename[65:] + " ...")
        mat = np.genfromtxt(open(filename, "rb"), delimiter=",", dtype=(str, 15))
        mat = remove_empty_entries(mat)

        # reformat data
        mat = np.delete(mat, 0, axis=0)
        mat = np.delete(mat, 1, axis=0)
        for j in range(0, mat.shape[1]):
            mat[1, j] = str(mat[1, j]) + " " + str(mat[2, j][:-3])

        mat = np.delete(mat, 2, axis=0)
        convert_datetime_to_measurement(mat)
        convert_locs_to_measurement(mat, loc_key)
        normalize_data(mat, 0, 11)
        matrix_to_file(mat, filename[65:], dest_path_matrices_no_na_normalized)

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
        mat = np.genfromtxt(open(filename, "rb"), delimiter=",", dtype=(str, 15))
        mat = remove_empty_entries(mat)
        convert_datetime_to_measurement(mat)
        convert_locs_to_measurement(mat, loc_key)
        normalize_data(mat, 0, 15)
        matrix_to_file(mat, filename[65:], dest_path_matrices_all_data_normalized)


# This method removes the empty entries (i.e. ",,,") located at the ends of the .csv files in the matrices_no_na
# directory. new_mat is the matrix with empty entries removed.
def remove_empty_entries(mat):
    new_mat = mat

    for col in range(0, mat.shape[1]):
        if mat[0, col] == "":
            new_mat = mat[:, 0:col]
            break

    return new_mat


# This method normalizes the data in a matrix by finding the largest value in each row and dividing each element in
# row by that value. This will cause each point in the data to be between the 0 and 1. mat is the matrix whose rows will
# be normalized. first is the first row that contains data that needs to be normalized. last is the last row
# (non-inclusive) that contains data that needs to be normalized
def normalize_data(mat, first, last):
    for i in range(first, last):
        # norm_arr_str contains the string representation of an entire row that is going to be normalized
        norm_arr_str = mat[i, :]

        # convert all elements in norm_arr to float
        # norm_arr is the float representation of norm_arr_str
        norm_arr = np.zeros(len(norm_arr_str), dtype=float)
        for j in range(0, len(norm_arr)):
            try:
                norm_arr[j] = float(norm_arr_str[j])
            except ValueError:
                print("The value at index " + str(j) + " could not be cast to a float.")

        # determine the largest value in norm_arr
        max_val = np.amax(norm_arr)

        if max_val != 0:
            # normalize all the elements by dividing each element by max_val
            for k in range(0, len(norm_arr)):
                norm_arr[k] = norm_arr[k] / max_val

                # store the normalize array back into its respective row in mat
                mat[i, k] = str(norm_arr[k])


# Writes a matrix to a csv file. mat is the matrix being written to a file. filename is the name of the .csv file.
# destination_folder is the path to the destination folder where the .csv file will be stored
def matrix_to_file(mat, filename, destination_folder):
    file = open(destination_folder + filename, "w")

    for i in range(0, mat.shape[0]):
        for j in range(0, mat.shape[1]):
            if j < mat.shape[1] - 1:
                file.write(mat[i, j] + ",")
            else:
                file.write(mat[i, j] + "\n")

    file.close()


# This method converts all date and times into a number between [0, 24). This way, we can use time as a quantitative
# measurement.
def convert_datetime_to_measurement(mat):
    for j in range(0, mat.shape[1]):
        date_str = mat[1, j]
        date_arr = date_str.split("/")
        time = date_arr[2].split()[1].split(":")

        time_meas = int(time[0]) + (int(time[1]) / 60)

        mat[1, j] = time_meas

        # month = int(date_arr[0])
        # day = int(date_arr[1])
        # year = int("20" + date_arr[2].split()[0])
        #
        # time = date_arr[2].split()[1].split(":")
        #
        # hour = int(time[0])
        # minute = int(time[1])
        #
        # # determine the number of seconds since 12:00 AM on the day of the measurement to the actual measurement time
        # num_seconds = (datetime(year, month, day, hour, minute) - datetime(year, month, day, 0, 0)).total_seconds()
        #
        # new_mat[1, j] = num_seconds


# This method converts the locations (there are 77 of them) where the measurements were gathered into an integer. In
# this way, we can quantify location. mat, the matrix of data, and loc_key, the matrix of locations and their
# corresponding integers, is passed into the method. new_mat is the matrix with the updated location values, and is
# returned
def convert_locs_to_measurement(mat, loc_key):
    # new_mat = mat
    for j in range(0, mat.shape[1]):
        idx = np.where(mat[0, j] == loc_key[0, :])
        mat[0, j] = loc_key[1, idx[0][0]]

    # return new_mat

if __name__ == "__main__": main()