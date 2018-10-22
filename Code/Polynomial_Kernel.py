import numpy as np
from sklearn import model_selection
from sklearn import svm
import matplotlib.pyplot as plt
from textwrap import wrap
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
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

    X = np.load(data_path + "all_data_summer.npy")
    y = np.load(data_path + "all_data_summer_labels.npy")

    print("Creating training and testing sets ... ")
    num_alg = 0  # count the number of algae instances
    num_no_alg = 0  # count the number of no algae instances

    # Convert labels to binary: -1 for no algae and 1 for algae
    for i in range(0, len(y)):
        if y[i] == 0:
            y[i] = -1
            num_no_alg += 1
        elif y[i] == 1 or y[i] == 2:
            y[i] = 1
            num_alg += 1

    # shrink the data set by randomly removing occurences of no algae until the number of no algae samples equals the
    # number of algae samples
    idx = 0  # index for the data set
    sample_bias = 10  # adjust the difference in the number of the two types of samples (no_alg and alg)
    while num_no_alg != (num_alg - sample_bias):
        # circle through the data sets until the difference of num_no_alg and num_alg equals
        # the value specified by sample_bias
        if idx == (len(y) - 1):
            idx = 0

        if y[idx] == -1:
            if np.random.rand() >= 0.5:  # remove this sample with some probability
                y = np.delete(y, obj=idx)
                X = np.delete(X, obj=idx, axis=Constants.ROWS)
                num_no_alg -= 1
            else:
                idx += 1
        else:
            idx += 1

    print("Performing classification with polynomial kernel ... ")
    cumulative_ber = 0
    cumulative_no_alg_error = 0
    cumulative_alg_error = 0

    num_points = 4      # size of c and coef parameter arrays
    starting_point = 1

    avg_ber = np.zeros(shape=(num_points, num_points))      # avg BER evaluated for each C and coef
    idx = 0     # indexes the results matrix

    # construct parameter arrays
    c = np.linspace(start=starting_point, stop=1000, num=num_points, endpoint=True)
    coef = np.linspace(start=starting_point, stop=1000, num=num_points, endpoint=True)

    degree = 3
    num_splits = 100

    # define constants for plotting
    vmin = 1    # min BER
    vmax = 0    # max BER

    sss = model_selection.StratifiedShuffleSplit(n_splits=num_splits, test_size=0.2)

    for i in range(0, len(c)):
        for j in range(0, len(coef)):
            for train_idx, test_idx in sss.split(X, y):
                X_train, X_test = X[train_idx], X[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]

                svc = svm.SVC(
                    C=c[i],
                    kernel="poly",
                    degree=degree,
                    gamma="auto",
                    coef0=coef[j],
                    probability=False,
                    shrinking=True,
                    tol=0.0001,
                    verbose=False,
                    max_iter=-1,
                    decision_function_shape="ovo"
                )

                svc.fit(X_train, y_train)
                y_pred = svc.predict(X_test)

                ber, no_alg_error, alg_error, _ = calculate_error(y_pred, y_test)

                cumulative_ber += ber
                cumulative_no_alg_error += no_alg_error
                cumulative_alg_error += alg_error

            print("\n~~~ Results for polynomial of degree = " + str(degree) + ", C = " + str(float("%0.4f" % c[i])) +
                  ", coef0 = " + str(float("%0.4f" % coef[j])) + " ~~~\n")

            print("Averages from " + str(num_splits) + " iterations of stratified shuffle split\n")

            print_results(
                ber=float("%0.4f" % (cumulative_ber / num_splits)),
                no_alg_error=float("%0.4f" % (cumulative_no_alg_error / num_splits)),
                alg_error=float("%0.4f" % (cumulative_alg_error / num_splits))
            )

            avg_ber[i, j] = float("%0.4f" % (cumulative_ber / num_splits))

            idx += 1

            cumulative_ber = 0
            cumulative_no_alg_error = 0
            cumulative_alg_error = 0

            if (cumulative_ber / num_splits) < vmin:
                vmin = float("%0.4f" % (cumulative_ber / num_splits))
            if (cumulative_ber / num_splits) > vmax:
                vmax = float("%0.4f" % (cumulative_ber / num_splits))

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(c, coef, avg_ber, cmap=cm.coolwarm, vmin=vmin, vmax=vmax)
    # ax.scatter(results[0, :], results[1, :], results[2, :])
    plt.xlabel("C")
    plt.ylabel("coef")
    plt.title("\n".join(wrap("BER as a Function of C and coef for Polynomial Kernel", 60)))
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.savefig(os.path.join(dest_path, "BER as a Function of C and coef for Polynomial Kernel.png"))

    avg_ber = np.asarray(avg_ber)
    np.savetxt(os.path.join(dest_path, "poly-kernel-c.csv"), c, delimiter=",")
    np.savetxt(os.path.join(dest_path, "poly-kernel-coef.csv"), coef, delimiter=",")
    np.savetxt(os.path.join(dest_path, "poly-kernel-avg-ber.csv"), avg_ber, delimiter=",")


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

    # NOTE: I have define the rows to be the true labels and the columns to be the predicted labels

    ber = (mat_conf[0, 1] / (mat_conf[0, 0] + mat_conf[0, 1]) + mat_conf[1, 0] / (mat_conf[1, 1] + mat_conf[1, 0])) / 2

    no_alg_error = mat_conf[0, 1] / (mat_conf[0, 0] + mat_conf[0, 1])
    alg_error = mat_conf[1, 0] / (mat_conf[1, 1] + mat_conf[1, 0])

    ber = float("%0.4f" % ber)
    no_alg_error = float("%0.4f" % no_alg_error)
    alg_error = float("%0.4f" % alg_error)

    return ber, no_alg_error, alg_error, mat_conf


# This method prints the results of the linear classification
def print_results(ber, no_alg_error, alg_error):
    print("BER:", ber)
    print("No Algae Prediction Error:", no_alg_error)
    print("Algae Prediction Error:", alg_error)
    print()


if __name__ == "__main__": main()