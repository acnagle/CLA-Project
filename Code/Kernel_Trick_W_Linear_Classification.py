import numpy as np
from scipy.special import comb
import Constants
import errno
import os


def main():
    np.set_printoptions(threshold=np.inf)  # prints a full matrix rather than an abbreviated matrix

    print("\n\t##### EXECUTING KERNEL_TRICK_W_LINEAR_CLASSIFICATION.PY #####\n")

    # get source directories for normalized data matrices (summer months only!)
    # Original Data
    src_path_all_data_orig_w_ind = "/Users/Alliot/Documents/CLA-Project/Data/all-data-no-na/eigenvectors/" \
                                   "All_data_summer_matrix/All_Data_summer_matrix.csv"
    src_path_all_data_orig_no_ind = "/Users/Alliot/Documents/CLA-Project/Data/all-data-no-na/eigen-no-alg-ind/" \
                                    "All_data_summer_matrix/All_Data_summer_matrix.csv"
    src_path_mendota_orig_w_ind = "/Users/Alliot/Documents/CLA-Project/Data/all-data-no-na/eigenvectors/" \
                                  "Mendota_All_Data_summer_matrix/Mendota_All_Data_summer_matrix.csv"
    src_path_mendota_orig_no_ind = "/Users/Alliot/Documents/CLA-Project/Data/all-data-no-na/eigen-no-alg-ind/" \
                                   "Mendota_All_Data_summer_matrix/Mendota_All_Data_summer_matrix.csv"
    src_path_monona_orig_w_ind = "/Users/Alliot/Documents/CLA-Project/Data/all-data-no-na/eigenvectors/" \
                                 "Monona_All_Data_summer_matrix/Monona_All_Data_summer_matrix.csv"
    src_path_monona_orig_no_ind = "/Users/Alliot/Documents/CLA-Project/Data/all-data-no-na/eigen-no-alg-ind/" \
                                  "Monona_All_Data_summer_matrix/Monona_All_Data_summer_matrix.csv"

    dest_path = "/Users/Alliot/Documents/CLA-Project/Data/all-data-no-na/kernels/"

    # if dest_path does not exist, create it
    if not os.path.exists(dest_path):
        try:
            os.makedirs(dest_path)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

    # read in files from source directories
    mat_all_data_orig_w_ind = np.genfromtxt(open(src_path_all_data_orig_w_ind, "rb"), delimiter=",", dtype=float)
    mat_all_data_orig_no_ind = np.genfromtxt(open(src_path_all_data_orig_no_ind, "rb"), delimiter=",", dtype=float)
    mat_mendota_orig_w_ind = np.genfromtxt(open(src_path_mendota_orig_w_ind, "rb"), delimiter=",", dtype=float)
    mat_mendota_orig_no_ind = np.genfromtxt(open(src_path_mendota_orig_no_ind, "rb"), delimiter=",", dtype=float)
    mat_monona_orig_w_ind = np.genfromtxt(open(src_path_monona_orig_w_ind, "rb"), delimiter=",", dtype=float)
    mat_monona_orig_no_ind = np.genfromtxt(open(src_path_monona_orig_no_ind, "rb"), delimiter=",", dtype=float)

    mat_ker_all_data_no_ind = form_kernel(mat_all_data_orig_no_ind)
    matrix_to_file(mat_ker_all_data_no_ind, filename="All_Data_Kernel_no_ind.csv", destination_folder=dest_path)
    # np.savetext(fname=dest_path, X=mat_ker_all_data_no_ind + "All_Data_Kernel_no_ind.csv", delimiter=",")


# This method takes in a data matrix, mat, and uses the kernel trick on every vector in the matrix. In the newly formed
# vector, every feature in any column in mat is multiplied by every other feature in the column. For example, given n
# features in a vector, the transpose of one of the columns created by using the kernel trick will look as follows:
# [x1, x2, x3, ... , xn, x1^2, x2^2, x3^2, ... , x1*x2, x1*x3, x1*x4, ...]
# new_mat is the matrix where every column is adjusted by the kernel trick. new_mat is the returned matrix.
def form_kernel(mat):
    new_mat = np.zeros(shape=(Constants.NUM_ROWS_NO_IND_NO_LOC_ALL_DATA +
                              int(comb(Constants.NUM_ROWS_NO_IND_NO_LOC_ALL_DATA, 2)),
                              mat.shape[Constants.COLUMNS]), dtype=float)

    for i in range(0, Constants.NUM_ROWS_NO_IND_NO_LOC_ALL_DATA):
        # first NUM_ROWS_NO_IND_NO_LOC_ALL_DATA rows of new_mat are the same as mat
        new_mat[i, :] = mat[i, :]

    mat_row_idx = 0     # index the rows of mat
    for j in range(0, new_mat.shape[Constants.COLUMNS]):
        for k in range(Constants.NUM_ROWS_NO_IND_NO_LOC_ALL_DATA, new_mat.shape[Constants.ROWS]):
            for l in range(mat_row_idx, Constants.NUM_ROWS_NO_IND_NO_LOC_ALL_DATA - mat_row_idx):
                print(l)
                new_mat[k, i] = mat[mat_row_idx, i] * mat[l, i]

            mat_row_idx += 1

            #     new_mat[j + Constants.NUM_ROWS_NO_IND_NO_LOC_ALL_DATA, i] = mat[j, i] ** 2
            # else:
            #     print("in else", j)
            #     for k in range((Constants.NUM_ROWS_NO_IND_NO_LOC_ALL_DATA * 2), j+1):
            #         for l in range(0, Constants.NUM_ROWS_NO_IND_NO_LOC_ALL_DATA - mat_row_idx - 1):
            #             print("mat[mat_row_idx, i] = ", mat[mat_row_idx, i])
            #             print("mat[l, i] = ", mat[l, i])
            #             print()
            #             new_mat[j, i] = mat[mat_row_idx, i] * mat[l, i]
            #
            #         mat_row_idx += 1

    return new_mat


# Writes a matrix to a csv file. mat is the matrix being written to a file. filename is the name of the .csv file.
# destination_folder is the path to the destination folder where the .csv file will be stored
def matrix_to_file(mat, filename, destination_folder):
    file = open(destination_folder + filename, "w")

    for i in range(0, mat.shape[Constants.ROWS]):
        for j in range(0, mat.shape[Constants.COLUMNS]):
            if j < mat.shape[Constants.COLUMNS]-1:
                file.write(str(mat[i, j]) + ",")
            else:
                file.write(str(mat[i, j]) + "\n")

    file.close()


if __name__ == "__main__": main()
