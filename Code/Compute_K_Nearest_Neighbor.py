import numpy as np
import glob
import os


num_rows = 13   # number of measurements per data point.

def main():
    np.set_printoptions(threshold=np.inf)  # prints a full matrix rather than an abbreviated matrix

    print("\n\t##### EXECUTING COMPUTE_K_NEAREST_NEIGHBOR.PY #####")

    # get source directories for normalized data matrices (summer months only!)
    src_path_all_data = "/Users/Alliot/Documents/CLA-Project/Data/all-data-no-na/eigenvectors/" \
                                    "All_data_summer_matrix/All_Data_summer_matrix.csv"
    src_path_mendota = "/Users/Alliot/Documents/CLA-Project/Data/all-data-no-na/eigenvectors/" \
                                            "Mendota_All_Data_summer_matrix/Mendota_All_Data_summer_matrix.csv"
    src_path_monona = "/Users/Alliot/Documents/CLA-Project/Data/all-data-no-na/eigenvectors/" \
                                           "Monona_All_Data_summer_matrix/Monona_All_Data_summer_matrix.csv"

    # read in files from source directories
    mat_all_data = np.genfromtxt(open(src_path_all_data, "rb"), delimiter=",", dtype=float)
    mat_mendota = np.genfromtxt(open(src_path_mendota, "rb"), delimiter=",", dtype=float)
    mat_monona = np.genfromtxt(open(src_path_monona, "rb"), delimiter=",", dtype=float)

    # create validation set for each matrix
    mat_all_data_val = create_val_set(mat_all_data)
    mat_mendota_val = create_val_set(mat_mendota)
    mat_monona_val = create_val_set(mat_monona)


# This method creates and returns the validation set for a matrix, mat. The validation set consists of 20% of the data
# points in mat. In order to determine which data points are in the set, this method will calculate 20% of the width
# of mat and choose data points that are linearly spaced throughout mat. Thus, the width of the returned validation set
# matrix will be 20% the width of mat. mat is the matrix for which the validation set will be calculated. mat_val is the
# matrix containing the validation set from mat.
def create_val_set(mat):
    mat_width = mat.shape[1]    # width of mat
    num_val_points = np.floor(mat_width * 0.2)   # aka the width of mat_val

    # find the indices from mat which will choose the data points for mat_val
    val_idx = np.linspace(0, mat_width-1, num=num_val_points, dtype=int)
    mat_val = np.transpose([np.empty((num_rows, ))])

    for i in val_idx:
        mat_val = np.hstack([mat_val, np.transpose([mat[:, i]])])

    return mat_val


if __name__ == "__main__": main()