import numpy as np
import sys
import matplotlib.pyplot as plt
import os
import errno


num_rows_no_ind = 12   # number of measurements per data point for data with no indicator
num_rows_w_ind = 13   # number of measurements per data point for data with indicator
num_rows_3d_proj = 3    # number of rows in a 3D projection matrix


def main():
    np.set_printoptions(threshold=np.inf)  # prints a full matrix rather than an abbreviated matrix

    print("\n\t##### EXECUTING COMPUTE_K_NEAREST_NEIGHBOR.PY #####\n")

    # get source directories for normalized data matrices (summer months only!)
    # Original Data
    src_path_all_data_orig_no_ind = "/Users/Alliot/Documents/CLA-Project/Data/all-data-no-na/eigen-no-alg-ind/" \
                                    "All_data_summer_matrix/All_Data_summer_matrix.csv"
    src_path_mendota_orig_no_ind = "/Users/Alliot/Documents/CLA-Project/Data/all-data-no-na/eigen-no-alg-ind/" \
                                   "Mendota_All_Data_summer_matrix/Mendota_All_Data_summer_matrix.csv"
    src_path_monona_orig_no_ind = "/Users/Alliot/Documents/CLA-Project/Data/all-data-no-na/eigen-no-alg-ind/" \
                                  "Monona_All_Data_summer_matrix/Monona_All_Data_summer_matrix.csv"

    # Projected Data (3D)
    src_path_all_data_proj_no_ind = "/Users/Alliot/Documents/CLA-Project/Data/all-data-no-na/projections/" \
                                    "All_Data_summer_matrix/All_Data_summer_matrix_proj_no-alg-ind_3d.csv"
    src_path_mendota_proj_no_ind = "/Users/Alliot/Documents/CLA-Project/Data/all-data-no-na/projections/" \
                                   "Mendota_All_Data_summer_matrix/Mendota_All_Data_summer_matrix_proj_no-alg-ind_3d.csv"
    src_path_monona_proj_no_ind = "/Users/Alliot/Documents/CLA-Project/Data/all-data-no-na/projections/" \
                                  "Monona_All_Data_summer_matrix/Monona_All_Data_summer_matrix_proj_no-alg-ind_3d.csv"

    dest_path = "/Users/Alliot/Documents/CLA-Project/Data/all-data-no-na/PCA/"

    # if dest_path does not exist, create it
    if not os.path.exists(dest_path):
        try:
            os.makedirs(dest_path)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

    # read in files from source directories
    mat_all_data_orig_no_ind = np.genfromtxt(open(src_path_all_data_orig_no_ind, "rb"), delimiter=",", dtype=float)
    mat_mendota_orig_no_ind = np.genfromtxt(open(src_path_mendota_orig_no_ind, "rb"), delimiter=",", dtype=float)
    mat_monona_orig_no_ind = np.genfromtxt(open(src_path_monona_orig_no_ind, "rb"), delimiter=",", dtype=float)

    mat_all_data_proj_no_ind = np.genfromtxt(open(src_path_all_data_proj_no_ind, "rb"), delimiter=",", dtype=float)
    mat_mendota_proj_no_ind = np.genfromtxt(open(src_path_mendota_proj_no_ind, "rb"), delimiter=",", dtype=float)
    mat_monona_proj_no_ind = np.genfromtxt(open(src_path_monona_proj_no_ind, "rb"), delimiter=",", dtype=float)


# This method computes the steps required for the PCA. mat is the matrix for which PCA will be performed.
# It returns #TODO FIGURE OUT WHAT THIS METHOD RETURNS
def pca(mat):
    # 1. Subtract the mean from each data point along each dimension
    adj_mat = mat   # adj_mat is adjusted so that the mean is subtracted out of mat
    for i in range(0, mat.shape[0]):
        mean = np.mean(mat[i], axis=1)



if __name__ == "__main__": main()