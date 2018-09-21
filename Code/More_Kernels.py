import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import svm
import matplotlib.pyplot as plt
from textwrap import wrap
import errno
import os
import Constants
import sys


def main():
    np.set_printoptions(threshold=np.inf)  # prints a full matrix rather than an abbreviated matrix

    print("\n\t##### EXECUTING MORE_KERNELS.PY #####\n")

    # source directories for normalized data matrices with algae indicator (summer months only!)
    src_path_all_data_summer_norm_w_ind = "/Users/Alliot/Documents/CLA-Project/Data/all-data-no-na/eigenvectors/" \
                                          "All_data_summer_matrix/All_Data_summer_matrix.csv"
    src_path_mendota_norm_w_ind = "/Users/Alliot/Documents/CLA-Project/Data/all-data-no-na/eigenvectors/" \
                                  "Mendota_All_Data_summer_matrix/Mendota_All_Data_summer_matrix.csv"
    src_path_monona_norm_w_ind = "/Users/Alliot/Documents/CLA-Project/Data/all-data-no-na/eigenvectors/" \
                                 "Monona_All_Data_summer_matrix/Monona_All_Data_summer_matrix.csv"
    src_path_all_data_norm_w_ind = "/Users/Alliot/Documents/CLA-Project/Data/all-data-no-na/eigenvectors/" \
                                   "All_Data_matrix/All_Data_matrix.csv"

    # source directories for standard data set (No kernel trick, no algae indicator)
    src_path_all_data_summer_no_ind = "/Users/Alliot/Documents/CLA-Project/Data/all-data-no-na/eigen-no-alg-ind/" \
                                      "All_Data_summer_matrix/All_Data_summer_matrix.csv"
    src_path_mendota_no_ind = "/Users/Alliot/Documents/CLA-Project/Data/all-data-no-na/eigen-no-alg-ind/" \
                              "Mendota_All_Data_summer_matrix/Mendota_All_Data_summer_matrix.csv"
    src_path_monona_no_ind = "/Users/Alliot/Documents/CLA-Project/Data/all-data-no-na/eigen-no-alg-ind/" \
                             "Monona_All_Data_summer_matrix/Monona_All_Data_summer_matrix.csv"
    src_path_all_data_no_ind = "/Users/Alliot/Documents/CLA-Project/Data/all-data-no-na/eigen-no-alg-ind/" \
                               "All_Data_matrix/All_Data_matrix.csv"

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
    mat_all_data_summer_no_ind = np.genfromtxt(
        open(src_path_all_data_summer_no_ind, "rb"),
        delimiter=",",
        dtype=float
    )

    mat_mendota_norm_w_ind = np.genfromtxt(open(src_path_mendota_norm_w_ind, "rb"), delimiter=",", dtype=float)
    mat_mendota_no_ind = np.genfromtxt(open(src_path_mendota_no_ind, "rb"), delimiter=",", dtype=float)

    mat_monona_norm_w_ind = np.genfromtxt(open(src_path_monona_norm_w_ind, "rb"), delimiter=",", dtype=float)
    mat_monona_no_ind = np.genfromtxt(open(src_path_monona_no_ind, "rb"), delimiter=",", dtype=float)

    mat_all_data_norm_w_ind = np.genfromtxt(open(src_path_all_data_norm_w_ind, "rb"), delimiter=",", dtype=float)
    mat_all_data_no_ind = np.genfromtxt(open(src_path_all_data_no_ind, "rb"), delimiter=",", dtype=float)

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

    # transpose test and training sets so that they are in the correct format (n_samples, m_features)
    mat_all_data_summer_no_ind = mat_all_data_summer_no_ind.T
    mat_mendota_no_ind = mat_mendota_no_ind.T
    mat_monona_no_ind = mat_monona_no_ind.T
    mat_all_data_no_ind = mat_all_data_no_ind.T

    # vector of data matrices to try the kernel trick with
    data_vec = np.array([mat_all_data_summer_no_ind, mat_mendota_no_ind, mat_monona_no_ind, mat_all_data_no_ind])

    # description of each data matrix in data_vec
    data_desc = ["all lakes, summer months only (June through August)", "Mendota, summer months only",
                 "Monona, summer months only", "all lakes, all months"]

    # array of label vectors
    labels_vec = np.array([mat_all_data_summer_labels, mat_mendota_labels, mat_monona_labels, mat_all_data_labels])

    kernels = ["poly", "rbf", "sigmoid"]       # a matrix of kernel names to be used with sklearn

    num_iterations = 1

    c = np.linspace(start=1000, stop=5000, num=17)

    for i in range(0, len(data_vec)):
        for kern in kernels:
            # create vectors for plotting error rates for each kernel
            y_ber = np.zeros(c.shape[0])
            y_no_alg = np.zeros(c.shape[0])
            y_alg = np.zeros(c.shape[0])

            for k in range(0, c.shape[0]):
                svc = svm.SVC(
                    C=c[k],
                    kernel=kern,
                    degree=3,
                    gamma="auto",
                    coef0=0,
                    probability=False,
                    shrinking=True,
                    tol=0.0001,
                    verbose=False,
                    max_iter=-1,
                    decision_function_shape="ovo"
                )

                cumulative_ber = 0
                cumulative_no_alg_error = 0
                cumulative_alg_error = 0

                for j in range(0, num_iterations):
                    x_train, x_test, y_train, y_test = train_test_split(
                        data_vec[i],
                        labels_vec[i],
                        test_size=0.33,
                        # random_state=543,
                        shuffle=True
                    )

                    svc.fit(x_train, y_train)
                    pred_labels_test = svc.predict(x_test)

                    ber, no_alg_error, alg_error, _ = calculate_error(pred_labels_test, y_test)

                    cumulative_ber += ber
                    cumulative_no_alg_error += no_alg_error
                    cumulative_alg_error += alg_error

                total_ber = cumulative_ber / num_iterations
                total_no_alg_error = cumulative_no_alg_error / num_iterations
                total_alg_error = cumulative_alg_error / num_iterations

                y_ber[k] = total_ber
                y_no_alg[k] = total_no_alg_error
                y_alg[k] = total_alg_error

                print_results(
                    title="Results for " + data_desc[i] + " (Kernel type: " + kern + ", C = " + str(c[k]) + ")",
                    ber=total_ber,
                    no_alg_error=total_no_alg_error,
                    alg_error=total_alg_error
                )

            plt.figure()
            plt.plot(c, y_ber, "b", c, y_no_alg, "g", c, y_alg, "r")
            plt.ylabel("Error Rate")
            plt.xlabel("C")
            plt.legend(("BER", "No Algae", "Algae"))
            plt.title("\n".join(wrap("Error Rates vs. C for " + data_desc[i] + " (Kernel type: " + kern + ")", 60)))
            plt.savefig(os.path.join(dest_path, "Error Rates vs. C for " + data_desc[i] + " (Kernel type " + kern + ").png"))


# This method calculates the Balanced Error Rate (BER), and the error rates for no algae and algae prediction. This
# method accepts an array of predicted labels, pred_labels, and an array of target labels, target_labels. This method
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
        elif (pred_labels[i] == 1) and (target_labels[i] == 0):
            mat_conf[0, 1] += 1
        elif (pred_labels[i] == 0) and (target_labels[i] == 1):
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


# This method prints the results of the linear classification
def print_results(title, ber, no_alg_error, alg_error):
    print(title)
    print("BER:", ber)
    print("No Algae Prediction Error:", no_alg_error)
    print("Algae Prediction Error:", alg_error)
    print("---------------------------------------------------------------------------\n")


if __name__ == "__main__": main()
