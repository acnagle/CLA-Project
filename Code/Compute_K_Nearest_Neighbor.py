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
    print(mat_all_data.shape)

    # create validation set for each matrix
    mat_all_data_val, all_data_val_idx = create_val_set(mat_all_data)
    mat_mendota_val, mendota_val_idx = create_val_set(mat_mendota)
    mat_monona_val, monona_val_idx = create_val_set(mat_monona)
    print(mat_all_data_val.shape)

    # create training set
    mat_all_data_train = create_training_set(mat_all_data, all_data_val_idx)
    mat_mendota_train = create_training_set(mat_mendota, mendota_val_idx)
    mat_monona_train = create_training_set(mat_monona, monona_val_idx)
    print(mat_all_data_train.shape)

    # calculate k-nearest neighbor for each dataset
    mat_all_data_labels = calculate_k_nn(mat_all_data_train, mat_all_data_val, 5)

# This method creates and returns the validation set for a matrix, mat. The validation set consists of 20% of the data
# points in mat. In order to determine which data points are in the set, this method will calculate 20% of the width
# of mat and choose data points that are linearly spaced throughout mat. Thus, the width of the returned validation set
# matrix will be 20% the width of mat. mat is the matrix for which the validation set will be calculated. mat_val is the
# matrix containing the validation set from mat and is returned. val_idx is the indices within mat that the points in
# mat_val are chosen. val_idx is a returned value.
def create_val_set(mat):
    mat_width = mat.shape[1]    # width of mat
    num_val_points = np.floor(mat_width * 0.2)   # aka the width of mat_val

    # find the indices from mat which will choose the data points for mat_val
    val_idx = np.linspace(0, mat_width-1, num=num_val_points, dtype=int)
    mat_val = np.transpose([np.empty((num_rows, ))])
    print(val_idx.shape)
    for i in val_idx:
        mat_val = np.hstack([mat_val, np.transpose([mat[:, i]])])

    return mat_val, val_idx


# This method creates and returns the training set for a matrix, mat. The training set consists of 80% of the data
# points in mat, and are chosen so that they do not include any of the points in the validation set. This method returns
# mat_train, which is the training set.
def create_training_set(mat, val_idx):
    mat_train = mat
    for i in np.flip(val_idx, 0):
        mat_train = np.delete(mat_train, i, 1)

    return mat_train


# This method determines the labels for each point in the validation set, mat_val, used k-nearest neighbors in the
# training dataset, mat_train. k is the number of neighbors checked for in mat_train. labels is returned from this
# method. Ecah label has an associated value: 0 for no algal bloom, 0.5 for blue-green algal bloom, and 1 for green
# algal bloom.
def calculate_k_nn(mat_train, mat_val, k):


    return labels


if __name__ == "__main__": main()