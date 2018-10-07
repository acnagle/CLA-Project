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

    print("\n\t##### EXECUTING POLYNOMIAL_KERNEL.PY #####\n")

    print("Reading in data set ...")
    dest_path = "/Users/Alliot/Documents/CLA-Project/Data/all-data-no-na/kernels/"
    data_path = "/Users/Alliot/Documents/CLA-Project/Data/data-sets/"

    # if dest_path does not exist, create it
    if not os.path.exists(dest_path):
        try:
            os.makedirs(dest_path)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

    x = np.load(data_path + "data_summary_summer.npy")
    y = np.load(data_path + "data_summary_summer_labels.npy")

    print("Creating training and testing sets ... ")
    num_alg = 0  # count the number of algae instances
    num_no_alg = 0  # count the number of no algae instances

    # Convert labels to binary: -1 for no algae and 1 for algae
    for i in range(0, len(y)):
        if y[i] == 0:
            y[i] = -1
            num_no_alg += 1
        if y[i] == 1 or y[i] == 2:
            y[i] = 1
            num_alg += 1

    # shrink the data set by randomly removing occurences of no algae until the number of no algae samples equals the
    # number of algae samples
    idx = 0  # index for the data set
    sample_bias = 0  # adjust the difference in the number of the two types of samples (no_alg and alg)
    while num_no_alg != (num_alg - sample_bias):
        # circle through the data sets until the difference of num_no_alg and num_alg equals
        # the value specified by sample_bias
        if idx == (len(y) - 1):
            idx = 0

        if y[idx] == -1:
            if np.random.rand() >= 0.5:  # remove this sample with some probability
                y = np.delete(y, obj=idx)
                x = np.delete(x, obj=idx, axis=Constants.ROWS)
                num_no_alg -= 1
            else:
                idx += 1
        else:
            idx += 1

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.33,
        # random_state=123,
        shuffle=True
    )

    svc = svm.SVC(
        C=1000,
        kernel="rbf",
        gamma="auto",
        coef0=0,
        probability=False,
        shrinking=True,
        tol=0.0001,
        verbose=False,
        max_iter=-1,
        decision_function_shape="ovo"
    )

    svc.fit(x_train, y_train)
    y_pred = svc.predict(x_test)

    ber, no_alg_error, alg_error, mat_conf = calculate_error(y_pred, y_test)

    print("\n~~~~~~~~~~~~~~~~~~~~~~ Results ~~~~~~~~~~~~~~~~~~~~~~\n")
    print("BER:", ber)
    print("No Algae Error Rate:", no_alg_error)
    print("Algae Error Rate:", alg_error)
    print("Confusion Matrix:")
    print(mat_conf)
    print()

    # plt.figure()
    # plt.plot(c, y_ber, "b", c, y_no_alg, "g", c, y_alg, "r")
    # plt.ylabel("Error Rate")
    # plt.xlabel("C")
    # plt.legend(("BER", "No Algae", "Algae"))
    # plt.title("\n".join(wrap("Error Rates vs. C for " + data_desc[i] + " (Kernel type: Gaussian RBF)", 60)))
    # plt.savefig(os.path.join(dest_path, "Error Rates vs. C for " + data_desc[i] + " (Gaussian RBF Kernel).png"))


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

    # no_alg = 0   # number of 0s (no algae) in target_labels
    # no_alg_error = 0 # number of incorrect predictions on no algae (expected 0 but pred_labels[i] gave 1)
    # alg = 0   # number of 1s (algae) in target_labels
    # alg_error = 0 # number of incorrect predictions on algae (expected 1 but pred_labels[i] gave 0)
    #
    # for i in range(0, len(pred_labels)):
    #     if target_labels[i] == 0:
    #         no_alg += 1
    #         if pred_labels[i] == 1:
    #             no_alg_error += 1
    #     elif target_labels[i] == 1:
    #         alg += 1
    #         if pred_labels[i] == 0:
    #             alg_error += 1
    #     else:
    #         print("Unexpected target label: ", target_labels[i])
    #         sys.exit()
    #
    # no_alg_error = no_alg_error / no_alg
    # alg_error = alg_error / alg
    #
    # return no_alg_error, alg_error

    # This for loop will populate mat_conf with the true labels and the predicted labels simultaneously.
    for i in range(0, len(pred_labels)):
        if (pred_labels[i] == -1) and (target_labels[i] == -1):
            mat_conf[0, 0] += 1
        elif (pred_labels[i] == 1) and (target_labels[i] == -1):
            mat_conf[0, 1] += 1
        elif (pred_labels[i] == -1) and (target_labels[i] == 1):
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

    ber = float("%0.4f" % ber)
    no_alg_error = float("%0.4f" % no_alg_error)
    alg_error = float("%0.4f" % alg_error)

    return ber, no_alg_error, alg_error, mat_conf


# This method prints the results of the linear classification
def print_results(title, no_alg_error, alg_error):
    print(title)
    print("No Algae Prediction Error:", no_alg_error)
    print("Algae Prediction Error:", alg_error)
    print("---------------------------------------------------------------------------\n")


if __name__ == "__main__": main()
