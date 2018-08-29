import numpy as np
from sklearn import linear_model
from sklearn.model_selection import train_test_split
import errno
import os
import Constants
import sys


def main():
    np.set_printoptions(threshold=np.inf)  # prints a full matrix rather than an abbreviated matrix

    print("\n\t##### EXECUTING KERNEL_TRICK_W_LINEAR_CLASSIFICATION.PY #####\n")

    # source directories for normalized data matrices with algae indicator (summer months only!)
    src_path_all_data_summer_norm_w_ind = "/Users/Alliot/Documents/CLA-Project/Data/all-data-no-na/eigenvectors/" \
                                          "All_data_summer_matrix/All_Data_summer_matrix.csv"
    src_path_mendota_norm_w_ind = "/Users/Alliot/Documents/CLA-Project/Data/all-data-no-na/eigenvectors/" \
                                  "Mendota_All_Data_summer_matrix/Mendota_All_Data_summer_matrix.csv"
    src_path_monona_norm_w_ind = "/Users/Alliot/Documents/CLA-Project/Data/all-data-no-na/eigenvectors/" \
                                 "Monona_All_Data_summer_matrix/Monona_All_Data_summer_matrix.csv"
    src_path_all_data_norm_w_ind = "/Users/Alliot/Documents/CLA-Project/Data/all-data-no-na/eigenvectors/" \
                                   "All_Data_matrix/All_Data_matrix.csv"

    # source directories for matrices using the kernel trick
    src_path_all_data_summer_ker_no_ind = "/Users/Alliot/Documents/CLA-Project/Data/all-data-no-na/kernels/" \
                                          "All_Data_Summer_Kernel_no_ind.csv"
    src_path_mendota_ker_no_ind = "/Users/Alliot/Documents/CLA-Project/Data/all-data-no-na/kernels/" \
                                  "Mendota_Kernel_no_ind.csv"
    src_path_monona_ker_no_ind = "/Users/Alliot/Documents/CLA-Project/Data/all-data-no-na/kernels/" \
                                 "Monona_Kernel_no_ind.csv"
    src_path_all_data_ker_no_ind = "/Users/Alliot/Documents/CLA-Project/Data/all-data-no-na/kernels/" \
                                   "All_Data_Kernel_no_ind.csv"

    dest_path = "/Users/Alliot/Documents/CLA-Project/Data/all-data-no-na/kernels/"

    # if dest_path does not exist, create it
    if not os.path.exists(dest_path):
        try:
            os.makedirs(dest_path)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

    # read in files from source directories
    mat_all_data_summer_norm_w_ind = np.genfromtxt(
        open(src_path_all_data_summer_norm_w_ind, "rb"),
        delimiter=",",
        dtype=float
    )
    mat_all_data_summer_ker_no_ind = np.genfromtxt(
        open(src_path_all_data_summer_ker_no_ind, "rb"),
        delimiter=",",
        dtype=float
    )
    mat_mendota_norm_w_ind = np.genfromtxt(open(src_path_mendota_norm_w_ind, "rb"), delimiter=",", dtype=float)
    mat_mendota_ker_no_ind = np.genfromtxt(open(src_path_mendota_ker_no_ind, "rb"), delimiter=",", dtype=float)
    mat_monona_norm_w_ind = np.genfromtxt(open(src_path_monona_norm_w_ind, "rb"), delimiter=",", dtype=float)
    mat_monona_ker_no_ind = np.genfromtxt(open(src_path_monona_ker_no_ind, "rb"), delimiter=",", dtype=float)
    mat_all_data_norm_w_ind = np.genfromtxt(open(src_path_all_data_norm_w_ind, "rb"), delimiter=",", dtype=float)
    mat_all_data_ker_no_ind = np.genfromtxt(open(src_path_all_data_ker_no_ind, "rb"), delimiter=",", dtype=float)

    # get the labels for each norm matrix. THE ONLY PURPOSE OF THE NORM MATRICES IS TO RETRIEVE THE LABELS!
    mat_all_data_summer_labels = mat_all_data_summer_norm_w_ind[Constants.ALGAL_BLOOM_SHEEN_NO_LOC, :]
    mat_mendota_labels = mat_mendota_norm_w_ind[Constants.ALGAL_BLOOM_SHEEN_NO_LOC, :]
    mat_monona_labels = mat_monona_norm_w_ind[Constants.ALGAL_BLOOM_SHEEN_NO_LOC, :]
    mat_all_data_labels = mat_all_data_norm_w_ind[Constants.ALGAL_BLOOM_SHEEN_NO_LOC, :]

    # modify the label vectors to be binary. Label 0 indicates no algal bloom, label 1 indicates an algal bloom
    for i in range(0, len(mat_all_data_summer_labels)):
        if mat_all_data_summer_labels[i] == 0.5:
            mat_all_data_summer_labels[i] = 1

    for i in range(0, len(mat_mendota_labels)):
        if mat_mendota_labels[i] == 0.5:
            mat_mendota_labels[i] = 1

    for i in range(0, len(mat_monona_labels)):
        if mat_monona_labels[i] == 0.5:
            mat_monona_labels[i] = 1

    for i in range(0, len(mat_all_data_labels)):
        if mat_all_data_labels[i] == 0.5:
            mat_all_data_labels[i] = 1

    # transpose validation and training sets so that they are in the correct format (n_samples, m_features)
    mat_all_data_summer_ker_no_ind = mat_all_data_summer_ker_no_ind.T
    mat_mendota_ker_no_ind = mat_mendota_ker_no_ind.T
    mat_monona_ker_no_ind = mat_monona_ker_no_ind.T
    mat_all_data_ker_no_ind = mat_all_data_ker_no_ind.T

    # use linear classification to predict the labels for the validation data sets
    mat_all_data_summer_coef, mat_all_data_summer_intercept, mat_all_data_summer_pred_labels_val, mat_all_data_summer_target_labels_val = \
        linear_classification(data_matrix=mat_all_data_summer_ker_no_ind, labels=mat_all_data_summer_labels)

    mat_mendota_coef, mat_mendota_intercept, mat_mendota_pred_labels_val, mat_mendota_target_labels_val = \
        linear_classification(data_matrix=mat_mendota_ker_no_ind, labels=mat_mendota_labels)

    mat_monona_coef, mat_monona_intercept, mat_monona_pred_labels_val, mat_monona_target_labels_val = \
        linear_classification(data_matrix=mat_monona_ker_no_ind, labels=mat_monona_labels)

    mat_all_data_coef, mat_all_data_intercept, mat_all_data_pred_labels_val, mat_all_data_target_labels_val = \
        linear_classification(data_matrix=mat_all_data_ker_no_ind, labels=mat_all_data_labels)

    # calculate errors
    mat_all_data_summer_ber, mat_all_data_summer_no_alg_error, mat_all_data_summer_alg_error, mat_all_data_summer_conf = calculate_error(
        pred_labels=mat_all_data_pred_labels_val,
        target_labels=mat_all_data_target_labels_val
    )

    mat_mendota_ber, mat_mendota_no_alg_error, mat_mendota_alg_error, mat_mendota_conf = calculate_error(
        pred_labels=mat_mendota_pred_labels_val,
        target_labels=mat_mendota_target_labels_val
    )

    mat_monona_ber, mat_monona_no_alg_error, mat_monona_alg_error, mat_monona_conf = calculate_error(
        pred_labels=mat_monona_pred_labels_val,
        target_labels=mat_monona_target_labels_val
    )

    mat_all_data_ber, mat_all_data_no_alg_error, mat_all_data_alg_error, mat_all_data_conf = calculate_error(
        pred_labels=mat_all_data_pred_labels_val,
        target_labels=mat_all_data_target_labels_val
    )

    print("Results for all lakes, all months")
    print("Confusion matrix:")
    print(mat_all_data_conf)
    print("\nBER:", mat_all_data_ber)
    print("No Algae Prediction Error:", mat_all_data_no_alg_error)
    print("Algae Prediction Error:", mat_all_data_alg_error)
    print("---------------------------------------------------------------------------\n")

    print("Results for all lakes, summer months (June through August) only")
    print("Confusion matrix:")
    print(mat_all_data_summer_conf)
    print("\nBER:", mat_all_data_summer_ber)
    print("No Algae Prediction Error:", mat_all_data_summer_no_alg_error)
    print("Algae Prediction Error:", mat_all_data_summer_alg_error)
    print("---------------------------------------------------------------------------\n")

    print("Results for lake Mendota, summer months only")
    print("Confusion matrix:")
    print(mat_mendota_conf)
    print("\nBER:", mat_mendota_ber)
    print("No Algae Prediction Error:", mat_mendota_no_alg_error)
    print("Algae Prediction Error:", mat_mendota_alg_error)
    print("---------------------------------------------------------------------------\n")

    print("Results for lake Monona, summer months only")
    print("\nConfusion matrix:")
    print(mat_monona_conf)
    print("BER:", mat_monona_ber)
    print("No Algae Prediction Error:", mat_monona_no_alg_error)
    print("Algae Prediction Error:", mat_monona_alg_error)
    print("---------------------------------------------------------------------------\n")


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
        test_size=0.1
    )
    
    clf = linear_model.SGDClassifier(
        loss="perceptron",
        penalty="none",
        alpha=0.0001,
        fit_intercept=True,
        max_iter=200,
        tol=None,
        shuffle=True,
        verbose=0,
        n_jobs=1,
        random_state=None,
        learning_rate="optimal",
        class_weight=None,
        warm_start=False,        # Explore this parameters too
        average=True,
    )

    clf.fit(x_train, y_train)
    pred_labels_val = clf.predict(x_val)

    return clf.coef_, clf.intercept_, pred_labels_val, y_val


if __name__ == "__main__": main()
