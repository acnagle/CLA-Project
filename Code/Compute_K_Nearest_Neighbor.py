import numpy as np


num_rows_no_ind = 12   # number of measurements per data point for data with no indicator
num_rows_w_ind = 12   # number of measurements per data point for data with indicator


def main():
    np.set_printoptions(threshold=np.inf)  # prints a full matrix rather than an abbreviated matrix

    print("\n\t##### EXECUTING COMPUTE_K_NEAREST_NEIGHBOR.PY #####")

    # get source directories for normalized data matrices (summer months only!)
    src_path_all_data_w_ind = "/Users/Alliot/Documents/CLA-Project/Data/all-data-no-na/eigenvectors/" \
                                    "All_data_summer_matrix/All_Data_summer_matrix.csv"
    src_path_all_data_no_ind = "/Users/Alliot/Documents/CLA-Project/Data/all-data-no-na/eigen-no-alg-ind/" \
                              "All_data_summer_matrix/All_Data_summer_matrix.csv"
    src_path_mendota_w_ind = "/Users/Alliot/Documents/CLA-Project/Data/all-data-no-na/eigenvectors/" \
                                            "Mendota_All_Data_summer_matrix/Mendota_All_Data_summer_matrix.csv"
    src_path_mendota_no_ind = "/Users/Alliot/Documents/CLA-Project/Data/all-data-no-na/eigen-no-alg-ind/" \
                             "Mendota_All_Data_summer_matrix/Mendota_All_Data_summer_matrix.csv"
    src_path_monona_w_ind = "/Users/Alliot/Documents/CLA-Project/Data/all-data-no-na/eigenvectors/" \
                                           "Monona_All_Data_summer_matrix/Monona_All_Data_summer_matrix.csv"
    src_path_monona_no_ind = "/Users/Alliot/Documents/CLA-Project/Data/all-data-no-na/eigen-no-alg-ind/" \
                            "Monona_All_Data_summer_matrix/Monona_All_Data_summer_matrix.csv"

    # read in files from source directories
    mat_all_data_w_ind = np.genfromtxt(open(src_path_all_data_w_ind, "rb"), delimiter=",", dtype=float)
    mat_all_data_no_ind = np.genfromtxt(open(src_path_all_data_no_ind, "rb"), delimiter=",", dtype=float)
    mat_mendota_w_ind = np.genfromtxt(open(src_path_mendota_w_ind, "rb"), delimiter=",", dtype=float)
    mat_mendota_no_ind = np.genfromtxt(open(src_path_mendota_no_ind, "rb"), delimiter=",", dtype=float)
    mat_monona_w_ind = np.genfromtxt(open(src_path_monona_w_ind, "rb"), delimiter=",", dtype=float)
    mat_monona_no_ind = np.genfromtxt(open(src_path_monona_no_ind, "rb"), delimiter=",", dtype=float)
    # print(mat_all_data_no_ind.shape)

    # create validation set for each matrix which also doesn't include the algae indicators
    mat_all_data_val_no_ind, all_data_val_idx = create_val_set(mat_all_data_no_ind)
    mat_mendota_val_no_ind, mendota_val_idx = create_val_set(mat_mendota_no_ind)
    mat_monona_val_no_ind, monona_val_idx = create_val_set(mat_monona_no_ind)
    # print(mat_all_data_val_no_ind.shape)

    # create training set for each matrix which also doesn;t include the algae indicators
    mat_all_data_train_no_ind = create_training_set(mat_all_data_no_ind, all_data_val_idx)
    mat_mendota_train_no_ind = create_training_set(mat_mendota_no_ind, mendota_val_idx)
    mat_monona_train_no_ind = create_training_set(mat_monona_no_ind, monona_val_idx)
    # print(mat_all_data_train_no_ind.shape)

    # calculate k-nearest neighbor for each dataset
    # mat_all_data_labels, mat_all_data_error = \
    calculate_k_nn(mat_all_data_train_no_ind, mat_all_data_val_no_ind, mat_all_data_w_ind, 5)

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
    mat_val = np.transpose([np.empty((num_rows_no_ind, ))])
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
# algal bloom. mat_train and mat_val do not have the algae indicators included, but mat_w_ind does. mat_w_ind has the
# same exact vectors as mat_train and mat_val when concatenated, but mat_w_ind simply includes the algae indicator.
# error is the amount of error in determining the labels
def calculate_k_nn(mat_train, mat_val, mat_w_ind, k):
    # 1. Create a matrix of L2-norms. For each column (x_val) in mat_val, calculate || x_val - x_train ||^2, where
    # x_train is every column in mat_train. A row in l2_mat corresponds to a column in mat_val, and a columnn in l2_mat
    # corresponds to a column in mat_train. For example, the l2_mat[2, 5] = || x_val_2 - x_train ||^2
    l2_mat = np.empty((mat_val.shape[1], mat_train.shape[1]), dtype=float)
    for i in range(0, mat_val.shape[1]):
        for j in range(0, mat_train.shape[1]):
            l2_mat[i, j] = (np.linalg.norm(mat_val[:, i] - mat_train[:, j])) ** 2

    # 2. Determine k-nn for each row in l2_mat. For each nearest neighbor we will need the column index from mat_train
    # and the label of the column from mat_train.
    # a. Get column indices from mat_train
    k_nn_idx = np.empty((mat_val.shape[1], k), dtype=int)
    for i in range(0, mat_val.shape[1]):
        for j in range(0, k):
            k_nn_idx[i, j] = np.argmin(l2_mat[i, :])
            l2_mat[i, k_nn_idx[i, j]] = float("inf")


    print(k_nn_idx)
    # TODO MAKE SURE THAT K_NN_IDX IS ALWAYS GIVING THE SAME VALUES!!! MAY NEED TO DO SOME ROUNDING...

    # return labels, error


if __name__ == "__main__": main()