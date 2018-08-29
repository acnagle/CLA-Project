import numpy as np
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import errno
import os
import Constants
import sys


def main():
    np.set_printoptions(threshold=np.inf)  # prints a full matrix rather than an abbreviated matrix

    print("\n\t##### EXECUTING KERNEL_TRICK_W_LINEAR_CLASSIFICATION.PY #####\n")

    # source directories for normalized data matrices with algae indicator (summer months only!)
    src_path_all_data_norm_w_ind = "/Users/Alliot/Documents/CLA-Project/Data/all-data-no-na/eigenvectors/" \
                                   "All_data_summer_matrix/All_Data_summer_matrix.csv"
    src_path_mendota_norm_w_ind = "/Users/Alliot/Documents/CLA-Project/Data/all-data-no-na/eigenvectors/" \
                                  "Mendota_All_Data_summer_matrix/Mendota_All_Data_summer_matrix.csv"
    src_path_monona_norm_w_ind = "/Users/Alliot/Documents/CLA-Project/Data/all-data-no-na/eigenvectors/" \
                                 "Monona_All_Data_summer_matrix/Monona_All_Data_summer_matrix.csv"

    # source directories for matrices using the kernel trick
    src_path_all_data_ker_no_ind = "/Users/Alliot/Documents/CLA-Project/Data/all-data-no-na/kernels/" \
                                   "All_Data_Kernel_no_ind.csv"
    src_path_mendota_ker_no_ind = "/Users/Alliot/Documents/CLA-Project/Data/all-data-no-na/kernels/" \
                                  "Mendota_Kernel_no_ind.csv"
    src_path_monona_ker_no_ind = "/Users/Alliot/Documents/CLA-Project/Data/all-data-no-na/kernels/" \
                                 "Monona_Kernel_no_ind.csv"

    dest_path = "/Users/Alliot/Documents/CLA-Project/Data/all-data-no-na/kernels/"

    # if dest_path does not exist, create it
    if not os.path.exists(dest_path):
        try:
            os.makedirs(dest_path)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

    # read in files from source directories
    mat_all_data_norm_w_ind = np.genfromtxt(open(src_path_all_data_norm_w_ind, "rb"), delimiter=",", dtype=float)
    mat_all_data_ker_no_ind = np.genfromtxt(open(src_path_all_data_ker_no_ind, "rb"), delimiter=",", dtype=float)
    mat_mendota_norm_w_ind = np.genfromtxt(open(src_path_mendota_norm_w_ind, "rb"), delimiter=",", dtype=float)
    mat_mendota_ker_no_ind = np.genfromtxt(open(src_path_mendota_ker_no_ind, "rb"), delimiter=",", dtype=float)
    mat_monona_norm_w_ind = np.genfromtxt(open(src_path_monona_norm_w_ind, "rb"), delimiter=",", dtype=float)
    mat_monona_ker_no_ind = np.genfromtxt(open(src_path_monona_ker_no_ind, "rb"), delimiter=",", dtype=float)

    # get the labels for each norm matrix. THE ONLY PURPOSE OF THE NORM MATRICES IS TO RETRIEVE THE LABELS!
    mat_all_data_labels = mat_all_data_norm_w_ind[Constants.ALGAL_BLOOM_SHEEN_NO_LOC, :]
    mat_mendota_labels = mat_mendota_norm_w_ind[Constants.ALGAL_BLOOM_SHEEN_NO_LOC, :]
    mat_monona_labels = mat_monona_norm_w_ind[Constants.ALGAL_BLOOM_SHEEN_NO_LOC, :]

    # modify the label vectors to be binary. Label 0 indicates no algal bloom, label 1 indicates an algal bloom
    for i in range(0, len(mat_all_data_labels)):
        if mat_all_data_labels[i] == 0.5:
            mat_all_data_labels[i] = 1

    for i in range(0, len(mat_mendota_labels)):
        if mat_mendota_labels[i] == 0.5:
            mat_mendota_labels[i] = 1

    for i in range(0, len(mat_monona_labels)):
        if mat_monona_labels[i] == 0.5:
            mat_monona_labels[i] = 1

    # create validation sets and validation label arrays
    # mat_all_data_ker_val, mat_all_data_ker_val_idx = create_val_set(mat=mat_all_data_ker_no_ind)
    # mat_mendota_ker_val, mat_mendota_ker_val_idx = create_val_set(mat=mat_mendota_ker_no_ind)
    # mat_monona_ker_val, mat_monona_ker_val_idx = create_val_set(mat=mat_monona_ker_no_ind)
    #
    # mat_all_data_target_labels_val = [mat_all_data_labels[i] for i in mat_all_data_ker_val_idx]
    # mat_mendota_target_labels_val = [mat_mendota_labels[i] for i in mat_mendota_ker_val_idx]
    # mat_monona_target_labels_val = [mat_monona_labels[i] for i in mat_monona_ker_val_idx]
    #
    # # create training sets and training label arrays
    # mat_all_data_ker_train, mat_all_data_target_labels_train = create_training_set(
    #     mat=mat_all_data_ker_no_ind,
    #     val_idx=mat_all_data_ker_val_idx,
    #     labels=mat_all_data_labels
    # )
    #
    # mat_mendota_ker_train, mat_mendota_target_labels_train = create_training_set(
    #     mat=mat_mendota_ker_no_ind,
    #     val_idx=mat_mendota_ker_val_idx,
    #     labels=mat_mendota_labels
    # )
    #
    # mat_monona_ker_train, mat_monona_target_labels_train = create_training_set(
    #     mat=mat_monona_ker_no_ind,
    #     val_idx=mat_monona_ker_val_idx,
    #     labels=mat_monona_labels
    # )
    #
    # # transpose validation and training sets so that they are in the correct format (n_samples, m_features)
    # mat_all_data_ker_val = mat_all_data_ker_val.T
    # mat_all_data_ker_train = mat_all_data_ker_train.T
    # mat_mendota_ker_val = mat_mendota_ker_val.T
    # mat_mendota_ker_train = mat_mendota_ker_train.T
    # mat_monona_ker_val = mat_monona_ker_val.T
    # mat_monona_ker_train = mat_monona_ker_train.T

    # transpose validation and training sets so that they are in the correct format (n_samples, m_features)
    mat_all_data_ker_no_ind = mat_all_data_ker_no_ind.T
    mat_mendota_ker_no_ind = mat_mendota_ker_no_ind.T
    mat_monona_ker_no_ind = mat_monona_ker_no_ind.T

    # create training and validation sets for the data matrices (x) and label arrays (y)
    # mat_all_data_x_train, mat_all_data_x_val, mat_all_data_y_train, mat_all_data_y_val = train_test_split(
    #     mat_all_data_ker_no_ind,
    #     mat_all_data_labels,
    #     test_size=0.2
    # )
    #
    # mat_mendota_x_train, mat_mendota_x_val, mat_mendota_y_train, mat_mendota_y_val = train_test_split(
    #     mat_mendota_ker_no_ind,
    #     mat_mendota_labels,
    #     test_size=0.2
    # )
    #
    # mat_monona_x_train, mat_monona_x_val, mat_monona_y_train, mat_monona_y_val = train_test_split(
    #     mat_monona_ker_no_ind,
    #     mat_monona_labels,
    #     test_size=0.2
    # )

    # use linear classification to predict the labels for the validation data sets
    # mat_all_data_coef, mat_all_data_intercept, mat_all_data_pred_labels_val = linear_classification(
    #     data_train=mat_all_data_ker_train,
    #     data_val=mat_all_data_ker_val,
    #     labels_train=mat_all_data_target_labels_train
    # )
    #
    # mat_mendota_coef, mat_mendota_intercept, mat_mendota_pred_labels_val = linear_classification(
    #     data_train=mat_mendota_ker_train,
    #     data_val=mat_mendota_ker_val,
    #     labels_train=mat_mendota_target_labels_train
    # )
    #
    # mat_monona_coef, mat_monona_intercept, mat_monona_pred_labels_val = linear_classification(
    #     data_train=mat_monona_ker_train,
    #     data_val=mat_monona_ker_val,
    #     labels_train=mat_monona_target_labels_train
    # )
    # # print(mat_all_data_pred_labels_val)
    # print(mat_all_data_target_labels_val)
    # # calculate errors
    # mat_all_data_ber, mat_all_data_no_alg_error, mat_all_data_alg_error, mat_all_data_conf =  calculate_error(
    #     pred_labels=mat_all_data_pred_labels_val,
    #     target_labels=mat_all_data_target_labels_val
    # )

    # use linear classification to predict the labels for the validation data sets
    mat_all_data_coef, mat_all_data_intercept, mat_all_data_pred_labels_val, mat_all_data_target_labels_val = \
        linear_classification(data_matrix=mat_all_data_ker_no_ind, labels=mat_all_data_labels)

    print("MSE:", mean_squared_error(mat_all_data_target_labels_val, mat_all_data_pred_labels_val))
    print("Variance score:", r2_score(mat_all_data_target_labels_val, mat_all_data_pred_labels_val))

    # calculate errors
    mat_all_data_ber, mat_all_data_no_alg_error, mat_all_data_alg_error, mat_all_data_conf = calculate_error(
        pred_labels=mat_all_data_pred_labels_val,
        target_labels=mat_all_data_target_labels_val
    )

    # print(mat_all_data_conf)
    # print("BER:", mat_all_data_ber)
    # print("Algae Prediction Error:", mat_all_data_alg_error)
    # print("No Algae Prediction Error:", mat_all_data_no_alg_error)


# This method calculates the Balanced Error Rate (BER), and the error rates for no algae and algae prediction. This
# methd accepts an array of predicted labels, pred_labels, and an array of target labels, target_labels. This method
# returns ber (the balanced error rate), no_alg_error (error rate for no algae prediction), and alg_error (error
# rate for algae prediction). The confusion matrix, mat_conf, is returned as well (see first comment in method for a
# description of a confusion matrix).
def calculate_error(pred_labels, target_labels):
    # Construct a confusion matrix, mat_conf. A confusion matrix consists of the true labels for the data points
    # along its rows, and the predicted labels from k-nearest neighbors along its columns. The confusion matrix will
    # be necessary to calculate BER and other relevant errors for evaluation of the kernel trick with linear
    # classification. mat_conf is a 2x2 matrix because we only have two labels: no algae and algae. Each entry in
    # mat_conf is the sum of occurrences of each predicted label for each true label.
    mat_conf = np.zeros(shape=(2, 2), dtype=int)

    if len(pred_labels) != len(target_labels):
        print("Predicted and target label arrays are not the same length!")
        sys.exit()

    # This for loop will populate mat_conf with the true labels and the predicted labels simultaneously.
    for i in range(0, len(pred_labels)):
        if (pred_labels[i] == 0) and (target_labels[i] == 0):
            mat_conf[0, 0] += 1
        elif (pred_labels[i] == 0) and (target_labels[i] == 1):
            mat_conf[0, 1] += 1
        elif (pred_labels[i] == 1) and (target_labels[i] == 0):
            mat_conf[1, 0] += 1
        elif (pred_labels[i] == 1) and (target_labels[i] == 1):
            mat_conf[1, 1] += 1

    # Calculate relevant errors and accuracies
    # Given a confusion matrix as follows:
    # [ a b ]
    # [ c d ]
    # We can define the following equations:
    # Balanced Error Rate (BER) = (b / (a + b) + c / (c + d)) / 2
    # error per label = each of the terms in the numerator of BER. ex: b / (a + b)

    ber = (mat_conf[0, 1] / (mat_conf[0, 0] + mat_conf[0, 1]) + mat_conf[1, 0] / (mat_conf[1, 1] + mat_conf[1, 0])) / 2

    no_alg_error = mat_conf[0, 1] / (mat_conf[0, 0] + mat_conf[0, 1])
    alg_error = mat_conf[1, 0] / (mat_conf[1, 1] + mat_conf[1, 0])

    return ber, no_alg_error, alg_error, mat_conf


# This method linearly classifies the data into two categories: no algae (label 0) and algae (label 1). data_train is
# the matrix (n_samples, m_features) which contains the training data. data_val is the matrix (n_samples, m_features)
# which contains the validation set. labels_train is the array containing the  algae bloom labels (0 or 1) for each
# feature vector in data_train. coef, the array of coefficients for the linear regression problem, and intercept, the
# independent term in the linear model, are returned. The predicated labels for the validation set, pred_labels_val, are
# returned. # TODO update this method header
def linear_classification(data_matrix, labels):
    x_train, x_val, y_train, y_val = train_test_split(
        data_matrix,
        labels,
        test_size=0.2
    )

    reg = linear_model.LinearRegression()
    reg.fit(x_train, y_train)
    pred_labels_val = reg.predict(x_val)

    return reg.coef_, reg.intercept_, pred_labels_val, y_val


# This method creates and returns the validation set for a matrix, mat. The validation set consists of 20% of the data
# points in mat. In order to determine which data points are in the set, this method will calculate 20% of the width
# of mat and choose data points that are linearly spaced throughout mat. Thus, the width of the returned validation set
# matrix will be 20% the width of mat. mat is the matrix for which the validation set will be calculated. mat_val is the
# matrix containing the validation set from mat and is returned. val_idx is the indices within mat that the points in
# mat_val are chosen. val_idx is a returned value.
def create_val_set(mat):
    mat_width = mat.shape[Constants.COLUMNS]    # width of mat
    num_val_points = np.floor(mat_width * 0.2)   # aka the width of mat_val

    # find the indices from mat which will choose the data points for mat_val
    val_idx = np.linspace(0, mat_width-1, num=num_val_points, dtype=int)

    mat_val = np.transpose([np.empty((mat.shape[Constants.ROWS], ))])

    for i in val_idx:
        mat_val = np.hstack([mat_val, np.transpose([mat[:, i]])])

    # SPECIAL NOTE: for some reason I can't explain, the above code in this method appends an extra column to the front
    # of mat_val with values that are extremely tiny and large (order of 10^-250 to 10^314 or so). This code deletes
    # that column
    mat_val = np.delete(mat_val, obj=0, axis=Constants.COLUMNS)

    return mat_val, val_idx


# This method creates and returns the training set for a matrix, mat, and the labels associated with the training set.
# The training set consists of 80% of the data points in mat, and are chosen so that they do not include any of the
# points in the validation set. This method returns mat_train, which is the training set and labels_train, which is the
# set of labels associated with mat_train
def create_training_set(mat, val_idx, labels):
    mat_train = mat
    labels_train = labels
    for i in np.flip(val_idx, axis=0):
        mat_train = np.delete(mat_train, obj=i, axis=Constants.COLUMNS)
        labels_train = np.delete(labels_train, obj=i)

    return mat_train, labels_train


if __name__ == "__main__": main()
