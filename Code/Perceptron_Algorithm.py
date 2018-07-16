import numpy as np
import os
import errno

num_rows_no_ind = 12   # number of measurements per data point for data with no indicator
num_rows_3d_proj = 3    # number of rows in a 3D projection matrix

def main():
    np.set_printoptions(threshold=np.inf)  # prints a full matrix rather than an abbreviated matrix

    print("\n\t##### EXECUTING PERCEPTRON_ALGORITHM.PY #####\n")

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

    # Projected Data (3D)
    src_path_all_data_proj_no_ind = "/Users/Alliot/Documents/CLA-Project/Data/all-data-no-na/projections/" \
                                   "All_Data_summer_matrix/All_Data_summer_matrix_proj_no-alg-ind_3d.csv"
    src_path_mendota_proj_no_ind = "/Users/Alliot/Documents/CLA-Project/Data/all-data-no-na/projections/" \
                                  "Mendota_All_Data_summer_matrix/Mendota_All_Data_summer_matrix_proj_no-alg-ind_3d.csv"
    src_path_monona_proj_no_ind = "/Users/Alliot/Documents/CLA-Project/Data/all-data-no-na/projections/" \
                                 "Monona_All_Data_summer_matrix/Monona_All_Data_summer_matrix_proj_no-alg-ind_3d.csv"

    # dest_path = "/Users/Alliot/Documents/CLA-Project/Perceptron/"
    #
    # # if dest_path does not exist, create it
    # if not os.path.exists(dest_path):
    #     try:
    #         os.makedirs(dest_path)
    #     except OSError as e:
    #         if e.errno != errno.EEXIST:
    #             raise

    # read in files from source directories
    mat_all_data_orig_w_ind = np.genfromtxt(open(src_path_all_data_orig_w_ind, "rb"), delimiter=",", dtype=float)
    mat_all_data_orig_no_ind = np.genfromtxt(open(src_path_all_data_orig_no_ind, "rb"), delimiter=",", dtype=float)
    mat_mendota_orig_w_ind = np.genfromtxt(open(src_path_mendota_orig_w_ind, "rb"), delimiter=",", dtype=float)
    mat_mendota_orig_no_ind = np.genfromtxt(open(src_path_mendota_orig_no_ind, "rb"), delimiter=",", dtype=float)
    mat_monona_orig_w_ind = np.genfromtxt(open(src_path_monona_orig_w_ind, "rb"), delimiter=",", dtype=float)
    mat_monona_orig_no_ind = np.genfromtxt(open(src_path_monona_orig_no_ind, "rb"), delimiter=",", dtype=float)

    mat_all_data_proj_no_ind = np.genfromtxt(open(src_path_all_data_proj_no_ind, "rb"), delimiter=",", dtype=float)
    mat_mendota_proj_no_ind = np.genfromtxt(open(src_path_mendota_proj_no_ind, "rb"), delimiter=",", dtype=float)
    mat_monona_proj_no_ind = np.genfromtxt(open(src_path_monona_proj_no_ind, "rb"), delimiter=",", dtype=float)

    mat_all_data_orig_w_ind_update = update_labels(mat_all_data_orig_w_ind)
    print(mat_all_data_orig_w_ind_update)

# This method adjusts the labels in the "w_ind" (with algae indicator) matrices to be more useful for the perceptron
# algorithm. Namely, this method will change the data points with indication of an algae bloom (blue-green and green)
# to have a label of 1. The data points that have an indication of no algae bloom will be updated to have a label of
# -1. mat is the matrix containing the data points and an algae indicator label. new_mat contains the updated labels
# and is returned at the end of this method.
def update_labels(mat):
    new_mat = mat
    for j in range(0, mat.shape[1]):
        if mat[1, j] >= 0.5:
            new_mat[1, j] = 1
        elif mat[1, j] == 0:
            new_mat[1, j] = -1
        else:
            print("Unexpected algae label at index", j)

    return new_mat


if __name__ == "__main__": main()
