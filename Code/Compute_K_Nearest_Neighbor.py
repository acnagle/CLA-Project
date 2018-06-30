import numpy as np


num_rows_no_ind = 12   # number of measurements per data point for data with no indicator
num_rows_w_ind = 13   # number of measurements per data point for data with indicator


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

    # create validation set for each matrix which also doesn't include the algae indicators
    mat_all_data_val_w_ind, all_data_val_idx = create_val_set(mat_all_data_w_ind)
    mat_all_data_val_no_ind, all_data_val_idx = create_val_set(mat_all_data_no_ind)
    mat_mendota_val_w_ind, mendota_val_idx = create_val_set(mat_mendota_w_ind)
    mat_mendota_val_no_ind, mendota_val_idx = create_val_set(mat_mendota_no_ind)
    mat_monona_val_w_ind, monona_val_idx = create_val_set(mat_monona_w_ind)
    mat_monona_val_no_ind, monona_val_idx = create_val_set(mat_monona_no_ind)

    # create training set for each matrix which also doesn;t include the algae indicators
    mat_all_data_train_w_ind = create_training_set(mat_all_data_w_ind, all_data_val_idx)
    mat_all_data_train_no_ind = create_training_set(mat_all_data_no_ind, all_data_val_idx)
    mat_mendota_train_w_ind = create_training_set(mat_mendota_w_ind, mendota_val_idx)
    mat_mendota_train_no_ind = create_training_set(mat_mendota_no_ind, mendota_val_idx)
    mat_monona_train_w_ind = create_training_set(mat_monona_w_ind, monona_val_idx)
    mat_monona_train_no_ind = create_training_set(mat_monona_no_ind, monona_val_idx)

    # calculate k-nearest neighbor for each dataset
    # mat_all_data_labels, mat_all_data_error = \
    calculate_k_nn(mat_all_data_train_w_ind, mat_all_data_train_no_ind,
                   mat_all_data_val_w_ind, mat_all_data_val_no_ind, 5)


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

    if mat.shape[0] == num_rows_no_ind:
        mat_val = np.transpose([np.empty((num_rows_no_ind, ))])
    elif mat.shape[0] == num_rows_w_ind:
        mat_val = np.transpose([np.empty((num_rows_w_ind,))])

    for i in val_idx:
        mat_val = np.hstack([mat_val, np.transpose([mat[:, i]])])

    # SPECIAL NOTE: for some reason I can't explain, the above code in this method appends an extra column to the front
    # of mat_val with values that are extremely tiny and large (order of 10^-250 to 10^314 or so). This code deletes
    # that column
    mat_val = np.delete(mat_val, 0, 1)

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
# method. Each label has an associated value: 0 for no algal bloom, 0.5 for blue-green algal bloom, and 1 for green
# algal bloom. mat_train_w_ind and mat_w_ind have the algae indicator included, whereas mat_val_no_ind and
# mat_train_no_ind do not. error is the amount of error in determining the labels
def calculate_k_nn(mat_train_w_ind, mat_train_no_ind, mat_val_w_indxs, mat_val_no_ind, k):
    # 1. Create a matrix of L2-norms. For each column (x_val) in mat_val_no_ind, calculate || x_val - x_train ||^2,
    # where x_train is every column in mat_train_no_ind. A row in l2_mat corresponds to a column in mat_val_no_ind,
    # and a column in l2_mat corresponds to a column in mat_train_no_ind. For example,
    # l2_mat[2, 5] = || x_val_2 - x_train ||^2
    l2_mat = np.empty((mat_val_no_ind.shape[1], mat_train_no_ind.shape[1]), dtype=float)
    for i in range(0, mat_val_no_ind.shape[1]):
        for j in range(0, mat_train_no_ind.shape[1]):
            l2_mat[i, j] = (np.linalg.norm(mat_val_no_ind[:, i] - mat_train_no_ind[:, j])) ** 2

    # 2. Determine k-nn for each row in l2_mat. For each nearest neighbor we will need the column index from
    # mat_train_no_ind and the label of the column from mat_train_no_ind.
    # a. Get column indices from mat_train
    k_nn_idx = np.empty((mat_val_no_ind.shape[1], k), dtype=int)
    for i in range(0, mat_val_no_ind.shape[1]):
        for j in range(0, k):
            k_nn_idx[i, j] = np.argmin(l2_mat[i, :])
            l2_mat[i, k_nn_idx[i, j]] = float("inf")

    # b. Create a 3D matrix containing the k-nearest neighbors for each vector in mat_val. The rows correspond to
    # the data measurements (num_rows_no_ind in total). The columns represent each of the k-nearest neighbors. Each
    # slice is the set of k-nearest neighbors for each vector in mat_val. the k_nn 3D matrix contains the data points
    # from mat_train_w_ind, so determining labels for the data points in mat_val_no_ind will be easier later.
    k_nn = np.empty((num_rows_w_ind, k, mat_val_no_ind.shape[1]))
    for i in range(0, mat_val_no_ind.shape[1]):
        for j in range(0, k):
            k_nn[:, j, i] = mat_train_w_ind[:, k_nn_idx[i, j]]

    # 3. Determine the label of each point in the validation set. In mat_train_w_ind, the algae bloom indicator is
    # located in row index 1. The single most abundant label in the nearest neighbor of a data point will be stored in
    # row index 1 of mat_val_label. This matrix will be an exact copy of mat_val_no_ind, except most of the rows will
    # be shifted down to make room for the new row of labels. If there are ever the same number of labels among the
    # different labels in the neighbors, the data point will get the label of whichever label is closet to the point.
    # Additionally, the label chosen must be in the set of labels that has the same number of different labels.
    num_no_alg = 0  # tally of number of data points in k-nn with no indication of algal bloom
    num_bg_alg = 0  # tally of number of data points in k-nn with blue-green algal bloom
    num_gr_alg = 0  # tally of number of data points in k-nn with green algal bloom
    label = 0       # label for data point
    mat_val_label = np.insert(mat_val_no_ind, 1, 0, axis=0)
    for i in range(0, mat_val_no_ind.shape[1]):
        for j in range(0, k):
            if k_nn[1, j, i] == 0:
                num_no_alg = num_no_alg + 1
            elif k_nn[1, j, i] == 0.5:
                num_bg_alg = num_bg_alg + 1
            elif k_nn[1, j, i] == 1:
                num_gr_alg = num_gr_alg + 1
            else:
                print("Invalid algae indicator value for labeling!")

        if (num_no_alg > num_bg_alg) and (num_no_alg > num_gr_alg):
            label = 0
        elif (num_bg_alg > num_no_alg) and (num_bg_alg > num_gr_alg):
            label = 0.5
        elif (num_gr_alg > num_no_alg) and (num_gr_alg > num_bg_alg):
            label = 1
        elif num_no_alg == num_bg_alg:
            for l in range(0, k):
                if (k_nn[1, l, i] == 0) or (k_nn[1, l, i] == 0.5):
                    label = k_nn[1, l, i]
                    break
        elif num_no_alg == num_gr_alg:
            for l in range(0, k):
                if (k_nn[1, l, i] == 0) or (k_nn[1, l, i] == 1):
                    label = k_nn[1, l, i]
                    break
        elif num_bg_alg == num_gr_alg:
            for l in range(0, k):
                if (k_nn[1, l, i] == 0.5) or (k_nn[1, l, i] == 1):
                    label = k_nn[1, l, i]
                    break

        mat_val_label[1, i] = label

        # reset tallies
        num_no_alg = 0
        num_bg_alg = 0
        num_gr_alg = 0

    print(mat_val_label)

    # TODO THOUROUGHLY TEST THE CODE FOR STEP 3 BEFORE MOVING ON

    # return labels, error


if __name__ == "__main__": main()