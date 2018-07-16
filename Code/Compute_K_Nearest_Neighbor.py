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

    dest_path = "/Users/Alliot/Documents/CLA-Project/Data/all-data-no-na/K-NN/"

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

    mat_all_data_proj_no_ind = np.genfromtxt(open(src_path_all_data_proj_no_ind, "rb"), delimiter=",", dtype=float)
    mat_mendota_proj_no_ind = np.genfromtxt(open(src_path_mendota_proj_no_ind, "rb"), delimiter=",", dtype=float)
    mat_monona_proj_no_ind = np.genfromtxt(open(src_path_monona_proj_no_ind, "rb"), delimiter=",", dtype=float)

    # create validation set for each matrix which also doesn't include the algae indicators
    mat_all_data_orig_val_w_ind, all_data_orig_val_idx = create_val_set(mat_all_data_orig_w_ind)
    mat_all_data_orig_val_no_ind, all_data_orig_val_idx = create_val_set(mat_all_data_orig_no_ind)
    mat_mendota_orig_val_w_ind, mendota_orig_val_idx = create_val_set(mat_mendota_orig_w_ind)
    mat_mendota_orig_val_no_ind, mendota_orig_val_idx = create_val_set(mat_mendota_orig_no_ind)
    mat_monona_orig_val_w_ind, monona_orig_val_idx = create_val_set(mat_monona_orig_w_ind)
    mat_monona_orig_val_no_ind, monona_orig_val_idx = create_val_set(mat_monona_orig_no_ind)

    mat_all_data_proj_val_no_ind, all_data_proj_val_idx = create_val_set(mat_all_data_proj_no_ind)
    mat_mendota_proj_val_no_ind, mendota_proj_val_idx = create_val_set(mat_mendota_proj_no_ind)
    mat_monona_proj_val_no_ind, monona_proj_val_idx = create_val_set(mat_monona_proj_no_ind)

    # create training set for each matrix which also doesn't include the algae indicators
    mat_all_data_orig_train_w_ind = create_training_set(mat_all_data_orig_w_ind, all_data_orig_val_idx)
    mat_all_data_orig_train_no_ind = create_training_set(mat_all_data_orig_no_ind, all_data_orig_val_idx)
    mat_mendota_orig_train_w_ind = create_training_set(mat_mendota_orig_w_ind, mendota_orig_val_idx)
    mat_mendota_orig_train_no_ind = create_training_set(mat_mendota_orig_no_ind, mendota_orig_val_idx)
    mat_monona_orig_train_w_ind = create_training_set(mat_monona_orig_w_ind, monona_orig_val_idx)
    mat_monona_orig_train_no_ind = create_training_set(mat_monona_orig_no_ind, monona_orig_val_idx)

    mat_all_data_proj_train_no_ind = create_training_set(mat_all_data_proj_no_ind, all_data_proj_val_idx)
    mat_mendota_proj_train_no_ind = create_training_set(mat_mendota_proj_no_ind, mendota_proj_val_idx)
    mat_monona_proj_train_no_ind = create_training_set(mat_monona_proj_no_ind, monona_proj_val_idx)

    # calculate k-nearest neighbor for each dataset and compute errors
    k_arr = np.linspace(1, 20, num=20, dtype=int)    # create a vector of k values to perform k-nn with

    # create vectors to store various balanced error rates (BER) for original data
    all_data_orig_ber = np.empty((k_arr.shape[0]))
    mendota_orig_ber = np.empty((k_arr.shape[0]))
    monona_orig_ber = np.empty((k_arr.shape[0]))
    all_data_orig_ber_bin = np.empty((k_arr.shape[0]))
    mendota_orig_ber_bin = np.empty((k_arr.shape[0]))
    monona_orig_ber_bin = np.empty((k_arr.shape[0]))

    # create vectors to store various balanced error rates (BER) for 3D projections
    all_data_proj_3d_ber = np.empty((k_arr.shape[0]))
    mendota_proj_3d_ber = np.empty((k_arr.shape[0]))
    monona_proj_3d_ber = np.empty((k_arr.shape[0]))
    all_data_proj_3d_ber_bin = np.empty((k_arr.shape[0]))
    mendota_proj_3d_ber_bin = np.empty((k_arr.shape[0]))
    monona_proj_3d_ber_bin = np.empty((k_arr.shape[0]))

    # create vectors to store accuracies for original data
    all_data_orig_acc = np.empty((k_arr.shape[0]))
    mendota_orig_acc = np.empty((k_arr.shape[0]))
    monona_orig_acc = np.empty((k_arr.shape[0]))
    all_data_orig_acc_bin = np.empty((k_arr.shape[0]))
    mendota_orig_acc_bin = np.empty((k_arr.shape[0]))
    monona_orig_acc_bin = np.empty((k_arr.shape[0]))

    # create vectors to store accuracies for 3D projections
    all_data_proj_3d_acc = np.empty((k_arr.shape[0]))
    mendota_proj_3d_acc = np.empty((k_arr.shape[0]))
    monona_proj_3d_acc = np.empty((k_arr.shape[0]))
    all_data_proj_3d_acc_bin = np.empty((k_arr.shape[0]))
    mendota_proj_3d_acc_bin = np.empty((k_arr.shape[0]))
    monona_proj_3d_acc_bin = np.empty((k_arr.shape[0]))

    for k in k_arr:
        print("\n\t%%%%% COMPUTING K-NN FOR K =", k, "%%%%%\n\n")
        print("k-nearest neighbors for summer, all data:")
        print("None vs. Blue-Green vs. Green\n")
        print("Original Data:\n")
        all_data_orig_ber[k - 1], all_data_orig_acc[k-1] = \
            calculate_k_nn(mat_all_data_orig_train_w_ind, mat_all_data_orig_train_no_ind,
                           mat_all_data_orig_val_w_ind, mat_all_data_orig_val_no_ind, k)
        print("Projected Data (3D):\n")
        all_data_proj_3d_ber[k-1], all_data_proj_3d_acc[k-1] = \
            calculate_k_nn(mat_all_data_orig_train_w_ind, mat_all_data_proj_train_no_ind,
                           mat_all_data_orig_val_w_ind, mat_all_data_proj_val_no_ind, k)
        print("None vs. Algae\n")
        all_data_orig_ber_bin[k-1], all_data_orig_acc_bin[k-1] = \
            calculate_k_nn_binary(mat_all_data_orig_train_w_ind, mat_all_data_orig_train_no_ind,
                           mat_all_data_orig_val_w_ind, mat_all_data_orig_val_no_ind, k)
        print("Projected Data (3D):\n")
        all_data_proj_3d_ber_bin[k-1], all_data_proj_3d_acc_bin[k-1] = \
            calculate_k_nn_binary(mat_all_data_orig_train_w_ind, mat_all_data_proj_train_no_ind,
                           mat_all_data_orig_val_w_ind, mat_all_data_proj_val_no_ind, k)
        print("===========================================================\n")
        print("k-nearest neighbors for summer, Mendota:\n")
        print("None vs. Blue-Green vs. Green\n")
        print("Original Data:\n")
        mendota_orig_ber[k-1], mendota_orig_acc[k-1] = \
            calculate_k_nn(mat_mendota_orig_train_w_ind, mat_mendota_orig_train_no_ind,
                           mat_mendota_orig_val_w_ind, mat_mendota_orig_val_no_ind, k)
        print("Projected Data (3D):\n")
        mendota_proj_3d_ber[k-1], mendota_proj_3d_acc[k-1] = \
            calculate_k_nn(mat_mendota_orig_train_w_ind, mat_mendota_proj_train_no_ind,
                           mat_mendota_orig_val_w_ind, mat_mendota_proj_val_no_ind, k)
        print("None vs. Algae\n")
        mendota_orig_ber_bin[k-1], mendota_orig_acc_bin[k-1] = \
            calculate_k_nn_binary(mat_mendota_orig_train_w_ind, mat_mendota_orig_train_no_ind,
                           mat_mendota_orig_val_w_ind, mat_mendota_orig_val_no_ind, k)
        print("Projected Data (3D):\n")
        mendota_proj_3d_ber_bin[k-1], mendota_proj_3d_acc_bin[k-1] = \
            calculate_k_nn_binary(mat_mendota_orig_train_w_ind, mat_mendota_proj_train_no_ind,
                           mat_mendota_orig_val_w_ind, mat_mendota_proj_val_no_ind, k)
        print("===========================================================\n")
        print("k-nearest neighbors for summer, Monona:\n")
        print("None vs. Blue-Green vs. Green\n")
        print("Original Data:\n")
        monona_orig_ber[k-1], monona_orig_acc[k-1] = \
            calculate_k_nn(mat_monona_orig_train_w_ind, mat_monona_orig_train_no_ind,
                           mat_monona_orig_val_w_ind, mat_monona_orig_val_no_ind, k)
        print("Projected Data (3D):\n")
        monona_proj_3d_ber[k-1], monona_proj_3d_acc[k-1] = \
            calculate_k_nn(mat_monona_orig_train_w_ind, mat_monona_proj_train_no_ind,
                           mat_monona_orig_val_w_ind, mat_monona_proj_val_no_ind, k)
        print("None vs. Algae\n")
        monona_orig_ber_bin[k-1], monona_orig_acc_bin[k-1] = \
            calculate_k_nn_binary(mat_monona_orig_train_w_ind, mat_monona_orig_train_no_ind,
                           mat_monona_orig_val_w_ind, mat_monona_orig_val_no_ind, k)
        print("Projected Data (3D):\n")
        monona_proj_3d_ber_bin[k-1], monona_proj_3d_acc_bin[k-1] = \
            calculate_k_nn_binary(mat_monona_orig_train_w_ind, mat_monona_proj_train_no_ind,
                           mat_monona_orig_val_w_ind, mat_monona_proj_val_no_ind, k)
        print("===========================================================\n")

    plt.figure(1)
    plt.plot(k_arr, all_data_orig_ber, "r", k_arr, mendota_orig_ber, "b", k_arr, monona_orig_ber, "g")
    plt.legend(("All Data", "Mendota", "Monona"))
    plt.ylabel("Balanced Error Rate")
    plt.xlabel("k")
    plt.xticks(np.arange(2, k_arr.shape[0]+1, 2))
    plt.title("K-NN Balanced Error Rate (BER) vs. k, Original Data (Summer only)")
    plt.savefig(os.path.join(dest_path, "Original Data BER.png"))

    plt.figure(2)
    plt.plot(k_arr, all_data_proj_3d_ber, "r", k_arr, mendota_proj_3d_ber, "b", k_arr, monona_proj_3d_ber, "g")
    plt.legend(("All Data", "Mendota", "Monona"))
    plt.ylabel("Balanced Error Rate")
    plt.xlabel("k")
    plt.xticks(np.arange(2, k_arr.shape[0] + 1, 2))
    plt.title("K-NN Balanced Error Rate (BER) vs. k, 3D Projection (Summer only)")
    plt.savefig(os.path.join(dest_path, "3D Projection BER.png"))

    plt.figure(3)
    plt.plot(k_arr, all_data_orig_acc, "r", k_arr, mendota_orig_acc, "b", k_arr, monona_orig_acc, "g")
    plt.legend(("All Data", "Mendota", "Monona"))
    plt.ylabel("Accuracy")
    plt.xlabel("k")
    plt.xticks(np.arange(2, k_arr.shape[0] + 1, 2))
    plt.title("K-NN Accuracy vs. k, Original Data (Summer only)")
    plt.savefig(os.path.join(dest_path, "Original Data Accuracy.png"))

    plt.figure(4)
    plt.plot(k_arr, all_data_proj_3d_acc, "r", k_arr, mendota_proj_3d_acc, "b", k_arr, monona_proj_3d_acc, "g")
    plt.legend(("All Data", "Mendota", "Monona"))
    plt.ylabel("Accuracy")
    plt.xlabel("k")
    plt.xticks(np.arange(2, k_arr.shape[0] + 1, 2))
    plt.title("K-NN Accuracy vs. k, 3D Projection (Summer only)")
    plt.savefig(os.path.join(dest_path, "3D Projection Accuracy.png"))

    plt.figure(5)
    plt.plot(k_arr, all_data_orig_ber_bin, "r", k_arr, mendota_orig_ber_bin, "b", k_arr, monona_orig_ber_bin, "g")
    plt.legend(("All Data", "Mendota", "Monona"))
    plt.ylabel("Balanced Error Rate")
    plt.xlabel("k")
    plt.xticks(np.arange(2, k_arr.shape[0] + 1, 2))
    plt.title("K-NN Balanced Error Rate (BER) vs. k, Original Data (Summer only) [Binary]")
    plt.savefig(os.path.join(dest_path, "Original Data BER (Binary).png"))

    plt.figure(6)
    plt.plot(k_arr, all_data_proj_3d_ber_bin, "r", k_arr, mendota_proj_3d_ber_bin, "b", k_arr, 
             monona_proj_3d_ber_bin, "g")
    plt.legend(("All Data", "Mendota", "Monona"))
    plt.ylabel("Balanced Error Rate")
    plt.xlabel("k")
    plt.xticks(np.arange(2, k_arr.shape[0] + 1, 2))
    plt.title("K-NN Balanced Error Rate (BER) vs. k, 3D Projection (Summer only) [Binary]")
    plt.savefig(os.path.join(dest_path, "3D Projection BER (Binary).png"))

    plt.figure(7)
    plt.plot(k_arr, all_data_orig_acc_bin, "r", k_arr, mendota_orig_acc_bin, "b", k_arr, monona_orig_acc_bin, "g")
    plt.legend(("All Data", "Mendota", "Monona"))
    plt.ylabel("Accuracy")
    plt.xlabel("k")
    plt.xticks(np.arange(2, k_arr.shape[0] + 1, 2))
    plt.title("K-NN Accuracy vs. k, Original Data (Summer only) [Binary]")
    plt.savefig(os.path.join(dest_path, "Original Data Accuracy (Binary).png"))

    plt.figure(8)
    plt.plot(k_arr, all_data_proj_3d_acc_bin, "r", k_arr, mendota_proj_3d_acc_bin, "b", k_arr,
             monona_proj_3d_acc_bin, "g")
    plt.legend(("All Data", "Mendota", "Monona"))
    plt.ylabel("Accuracy")
    plt.xlabel("k")
    plt.xticks(np.arange(2, k_arr.shape[0] + 1, 2))
    plt.title("K-NN Accuracy vs. k, 3D Projection (Summer only) [Binary]")
    plt.savefig(os.path.join(dest_path, "3D Projection Accuracy (Binary).png"))


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
        mat_val = np.transpose([np.empty((num_rows_w_ind, ))])
    elif mat.shape[0] == num_rows_3d_proj:
        mat_val = np.transpose([np.empty((num_rows_3d_proj, ))])
    else:
        print("Data matrix does not have the expected number of rows.")
        sys.exit()

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


# This method predicts the labels for each point in the validation set, mat_val, used k-nearest neighbors in the
# training dataset, mat_train. k is the number of neighbors checked for in mat_train. mat_train_w_ind and mat_w_ind
# have the algae indicator included, whereas mat_val_no_ind and mat_train_no_ind do not. ber is the Balanced Error
# Rate, and is returned from this method. accuracy is the accuracy of label predication, and is returned
# from this method.
def calculate_k_nn(mat_train_w_ind, mat_train_no_ind, mat_val_w_ind, mat_val_no_ind, k):
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

    # b. Create a matrix containing the k-nearest neighbor label for each vector in mat_val. The rows correspond to
    # the labels of each data point in mat_train_w_ind. The columns represent each of the k-nearest neighbors.
    k_nn_labels = np.empty((mat_val_no_ind.shape[1], k))
    for i in range(0, mat_val_no_ind.shape[1]):
        for j in range(0, k):
            k_nn_labels[i, j] = mat_train_w_ind[1, k_nn_idx[i, j]]

    # 3. predict the label of each point in the validation set. In mat_train_w_ind, the algae bloom indicator is
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
            if k_nn_labels[i, j] == 0:
                num_no_alg = num_no_alg + 1
            elif k_nn_labels[i, j] == 0.5:
                num_bg_alg = num_bg_alg + 1
            elif k_nn_labels[i, j] == 1:
                num_gr_alg = num_gr_alg + 1
            else:
                print("Error: Invalid algae indicator value for labeling!")

        if (num_no_alg > num_bg_alg) and (num_no_alg > num_gr_alg):
            label = 0
        elif (num_bg_alg > num_no_alg) and (num_bg_alg > num_gr_alg):
            label = 0.5
        elif (num_gr_alg > num_no_alg) and (num_gr_alg > num_bg_alg):
            label = 1
        elif num_no_alg == num_bg_alg:
            for l in range(0, k):
                if (k_nn_labels[i, l] == 0) or (k_nn_labels[i, l] == 0.5):
                    label = k_nn_labels[i, l]
                    break
        elif num_no_alg == num_gr_alg:
            for l in range(0, k):
                if (k_nn_labels[i, l] == 0) or (k_nn_labels[i, l] == 1):
                    label = k_nn_labels[i, l]
                    break
        elif num_bg_alg == num_gr_alg:
            for l in range(0, k):
                if (k_nn_labels[i, l] == 0.5) or (k_nn_labels[i, l] == 1):
                    label = k_nn_labels[i, l]
                    break

        mat_val_label[1, i] = label

        # reset tallies
        num_no_alg = 0
        num_bg_alg = 0
        num_gr_alg = 0

    # 4. Construct a confusion matrix, mat_conf. A confusion matrix consists of the true labels for the data points
    # along its rows, and the predicted labels from k-nearest neighbors along its columns. The confusion matrix will
    # necessary to calculate the balanced error rate (BER), accuracy, and other relevant errors for evaluation of the
    # k-nn method. mat_conf is a 3x3 matrix because we only have three labels: no algae, blue-green algae, and green
    # algae. Each entry in mat_conf is the sum of occurrences of each predicted label for each true label.
    mat_conf = np.zeros((3, 3), dtype=int)

    # This for loop will populate mat_conf with the true labels and the predicted labels simultaneously.
    for i in range(0, mat_val_w_ind.shape[1]):
        if mat_val_w_ind[1, i] == 0:
            if mat_val_label[1, i] == 0:
                mat_conf[0, 0] = mat_conf[0, 0] + 1
            elif mat_val_label[1, i] == 0.5:
                mat_conf[0, 1] = mat_conf[0, 1] + 1
            elif mat_val_label[1, i] == 1:
                mat_conf[0, 2] = mat_conf[0, 2] + 1
        elif mat_val_w_ind[1, i] == 0.5:
            if mat_val_label[1, i] == 0:
                mat_conf[1, 0] = mat_conf[1, 0] + 1
            elif mat_val_label[1, i] == 0.5:
                mat_conf[1, 1] = mat_conf[1, 1] + 1
            elif mat_val_label[1, i] == 1:
                mat_conf[1, 2] = mat_conf[1, 2] + 1
        elif mat_val_w_ind[1, i] == 1:
            if mat_val_label[1, i] == 0:
                mat_conf[2, 0] = mat_conf[2, 0] + 1
            elif mat_val_label[1, i] == 0.5:
                mat_conf[2, 1] = mat_conf[2, 1] + 1
            elif mat_val_label[1, i] == 1:
                mat_conf[2, 2] = mat_conf[2, 2] + 1

    print("Confusion matrix:")
    print(mat_conf)

    # 5. Calculate relevant errors and accuracies
    # Given a confusion matrix as follows:
    # [ a b c ]
    # [ d e f ]
    # [ g h i ]
    # We can define the following equations:
    # Balanced Error Rate (BER) = [(b + c) / (a + b + c) + (d + f) / (d + e + f) + (g + h) / (g + h + i)] / 3
    # accuracy = (a + e + i) / (sum(C[i, j])), where C[i, j] is the [i, j] entry of the confusion matrix
    # error per label = each of the terms in the numerator of BER. ex: (b + c) / (a + b + c)

    ber = ((mat_conf[0, 1] + mat_conf[0, 2]) / (mat_conf[0, 0] + mat_conf[0, 1] + mat_conf[0, 2]) +
           (mat_conf[1, 0] + mat_conf[1, 2]) / (mat_conf[1, 0] + mat_conf[1, 1] + mat_conf[1, 2]) +
           (mat_conf[2, 0] + mat_conf[2, 1]) / (mat_conf[2, 0] + mat_conf[2, 1] + mat_conf[2, 2])) / 3

    no_alg_error = (mat_conf[0, 1] + mat_conf[0, 2]) / (mat_conf[0, 0] + mat_conf[0, 1] + mat_conf[0, 2])
    bg_alg_error = (mat_conf[1, 0] + mat_conf[1, 2]) / (mat_conf[1, 0] + mat_conf[1, 1] + mat_conf[1, 2])
    gr_alg_error = (mat_conf[2, 0] + mat_conf[2, 1]) / (mat_conf[2, 0] + mat_conf[2, 1] + mat_conf[2, 2])

    accuracy = (mat_conf[0, 0] + mat_conf[1, 1] + mat_conf[2, 2]) / (mat_conf.sum())

    print("\nBalanced Error Rate (BER):\t    ", ber)
    print("No Algae Label Error Rate:\t    ", no_alg_error)
    print("Blue-Green Algae Label Error Rate:  ", bg_alg_error)
    print("Green Label Error Rate:\t\t    ", gr_alg_error)
    print("Accuracy:\t\t\t    ", accuracy, "\n")

    return ber, accuracy


# NOTE: This method is nearly the same as calculate_k_nn. The only difference is that the two distinct algae bloom
# types (blue-green and green) are grouped into one. In this way we have two options: no algae bloom or algae bloom.
# Hence the use of the word "binary" in the method name.
# This method predicts the labels for each point in the validation set, mat_val, used k-nearest neighbors in the
# training dataset, mat_train. k is the number of neighbors checked for in mat_train. mat_train_w_ind and mat_w_ind
# have the algae indicator included, whereas mat_val_no_ind and mat_train_no_ind do not. ber is the Balanced Error
# Rate, and is returned from this method. accuracy is the accuracy of label predication, and is returned
# from this method.
def calculate_k_nn_binary(mat_train_w_ind, mat_train_no_ind, mat_val_w_ind, mat_val_no_ind, k):
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

    # b. Create a matrix containing the k-nearest neighbor label for each vector in mat_val. The rows correspond to
    # the labels of each data point in mat_train_w_ind. The columns represent each of the k-nearest neighbors.
    k_nn_labels = np.empty((mat_val_no_ind.shape[1], k))
    for i in range(0, mat_val_no_ind.shape[1]):
        for j in range(0, k):
            k_nn_labels[i, j] = mat_train_w_ind[1, k_nn_idx[i, j]]

    # 3. predict the label of each point in the validation set. In mat_train_w_ind, the algae bloom indicator is
    # located in row index 1. The single most abundant label in the nearest neighbor of a data point will be stored in
    # row index 1 of mat_val_label. This matrix will be an exact copy of mat_val_no_ind, except most of the rows will
    # be shifted down to make room for the new row of labels. If there are ever the same number of labels among the
    # different labels in the neighbors, the data point will get the label of whichever label is closet to the point.
    # Additionally, the label chosen must be in the set of labels that has the same number of different labels.
    num_no_alg = 0  # tally of number of data points in k-nn with no indication of algal bloom
    num_w_alg = 0   # tally of number of data points in k-nn with algal bloom
    label = 0       # label for data point
    mat_val_label = np.insert(mat_val_no_ind, 1, 0, axis=0)
    for i in range(0, mat_val_no_ind.shape[1]):
        for j in range(0, k):
            if k_nn_labels[i, j] == 0:
                num_no_alg = num_no_alg + 1
            elif k_nn_labels[i, j] == 0.5:
                num_w_alg = num_w_alg + 1
            elif k_nn_labels[i, j] == 1:
                num_w_alg = num_w_alg + 1
            else:
                print("Error: Invalid algae indicator value for labeling!")

        if num_no_alg > num_w_alg:
            label = 0
        elif num_w_alg > num_no_alg:
            label = 1
        elif num_no_alg == num_w_alg:
            for l in range(0, k):
                if k_nn_labels[i, l] == 0:
                    label = 0
                    break
                elif (k_nn_labels[i, l] == 0.5) or (k_nn_labels[i, l] == 1):
                    label = 1
                    break

        mat_val_label[1, i] = label

        # reset tallies
        num_no_alg = 0
        num_w_alg = 0

    # 4. Construct a confusion matrix, mat_conf. A confusion matrix consists of the true labels for the data points
    # along its rows, and the predicted labels from k-nearest neighbors along its columns. The confusion matrix will
    # necessary to calculate the balanced error rate (BER), accuracy, and other relevant errors for evaluation of the
    # k-nn method. mat_conf is a 3x3 matrix because we only have three labels: no algae, blue-green algae, and green
    # algae. Each entry in mat_conf is the sum of occurrences of each predicted label for each true label.
    mat_conf = np.zeros((2, 2), dtype=int)

    # This for loop will populate mat_conf with the true labels and the predicted labels simultaneously.
    for i in range(0, mat_val_w_ind.shape[1]):
        if mat_val_w_ind[1, i] == 0:
            if mat_val_label[1, i] == 0:
                mat_conf[0, 0] = mat_conf[0, 0] + 1
            elif mat_val_label[1, i] == 1:
                mat_conf[0, 1] = mat_conf[0, 1] + 1
        elif mat_val_w_ind[1, i] == 1:
            if mat_val_label[1, i] == 0:
                mat_conf[1, 0] = mat_conf[1, 0] + 1
            elif mat_val_label[1, i] == 1:
                mat_conf[1, 1] = mat_conf[1, 1] + 1

    print("Confusion matrix:")
    print(mat_conf)

    # 5. Calculate relevant errors and accuracies
    # Given a confusion matrix as follows:
    # [ a b ]
    # [ c d ]
    # We can define the following equations:
    # Balanced Error Rate (BER) = (b / (a + b) + c / (c + d)) / 2
    # accuracy = (a + d) / (sum(C[i, j])), where C[i, j] is the [i, j] entry of the confusion matrix
    # error per label = each of the terms in the numerator of BER. ex: b / (a + b)

    ber = (mat_conf[0, 1] / (mat_conf[0, 0] + mat_conf[0, 1]) + mat_conf[1, 0] / (mat_conf[1, 1] + mat_conf[1, 0])) / 2

    no_alg_error = mat_conf[0, 1] / (mat_conf[0, 0] + mat_conf[0, 1])
    w_alg_error = mat_conf[1, 0] / (mat_conf[1, 1] + mat_conf[1, 0])

    accuracy = (mat_conf[0, 0] + mat_conf[1, 1]) / (mat_conf.sum())

    print("\nBalanced Error Rate (BER):\t    ", ber)
    print("No Algae Label Error Rate:\t    ", no_alg_error)
    print("With Algae Label Error Rate:\t    ", w_alg_error)
    print("Accuracy:\t\t\t    ", accuracy, "\n")

    return ber, accuracy

# Writes a matrix to a .csv file. mat is the matrix being written to a file. filename is the name
# of the .csv file. destination_folder is the path to the destination folder where the .csv file will be stored
def matrix_to_file(mat, filename, destination_folder):
    file = open(destination_folder + filename, "w")

    for i in range(0, mat.shape[0]):
        for j in range(0, mat.shape[1]):
            if j < mat.shape[1] - 1:
                file.write(str(mat[i, j]) + ",")
            else:
                file.write(str(mat[i, j]) + "\n")

    file.close()


if __name__ == "__main__": main()