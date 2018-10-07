import numpy as np
from sklearn import model_selection
import sys
import Constants


def main():
    np.set_printoptions(threshold=np.inf)  # prints a full matrix rather than an abbreviated matrix

    print("\n\t##### EXECUTING LS_CLASSIFICATION.PY #####\n")

    print("Reading in data set ... ")

    data_path = "/Users/Alliot/Documents/CLA-Project/Data/data-sets/"
    x = np.load(data_path + "data_summary_summer.npy")
    y = np.load(data_path + "data_summary_summer_labels.npy")

    x = np.concatenate((x, np.ones(shape=(x.shape[0], 1))), axis=Constants.COLUMNS)

    print("Creating training and testing sets ... ")
    num_alg = 0     # count the number of algae instances
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
    idx = 0     # index for the data set
    sample_bias = 100   # adjust the difference in the number of the two types of samples (no_alg and alg)
    while num_no_alg != (num_alg - sample_bias):
        # circle through the data sets until the difference of num_no_alg and num_alg equals
        # the value specified by sample_bias
        if idx == (len(y) - 1):
            idx = 0

        if y[idx] == -1:
            if np.random.rand() >= 0.5:     # remove this sample with some probability
                y = np.delete(y, obj=idx)
                x = np.delete(x, obj=idx, axis=Constants.ROWS)
                num_no_alg -= 1
            else:
                idx += 1
        else:
            idx += 1

    cumulative_ber = 0
    cumulative_no_alg_error = 0
    cumulative_alg_error = 0
    cumulative_w = np.zeros(shape=x.shape[Constants.COLUMNS])

    # save ber, no_alg_error, and alg_error (and weight vector [see below]) for lowest/highest BER and alg_error
    best_ber = [1, 0, 0]
    worst_ber = [0, 0, 0]
    best_alg_error = [0, 0, 1]
    worst_alg_error = [0, 0, 0]

    num_iterations = 100000

    print("Computing LS averages ... ")
    for i in range(0, num_iterations):
        x_train, x_test, y_train, y_test = model_selection.train_test_split(
            x,
            y,
            test_size=0.33,
            # random_state=123,
            shuffle=True
        )

        # print("Computing least squares ...")
        # these first four steps are intermediary before computing the weight vector w for LS
        a = np.transpose(x_train)
        b = np.matmul(a, x_train)
        c = np.linalg.inv(b)
        d = np.matmul(c, a)

        # compute the weight vector
        w = np.matmul(d, y_train)

        # print("Evaluating performance of least squares model ... ")
        # predict labels
        y_pred = np.matmul(x_test, w)
        y_pred = np.sign(y_pred)

        ber, no_alg_error, alg_error, _ = calculate_error(y_pred, y_test)

        cumulative_ber += ber
        cumulative_no_alg_error += no_alg_error
        cumulative_alg_error += alg_error
        cumulative_w = np.add(cumulative_w, w)

        if ber < best_ber[0]:
            best_ber = [ber, no_alg_error, alg_error]
            best_ber_w = w
        if ber > worst_ber[0]:
            worst_ber = [ber, no_alg_error, alg_error]
            worst_ber_w = w
        if alg_error < best_alg_error[2]:
            best_alg_error = [ber, no_alg_error, alg_error]
            best_alg_error_w = w
        if alg_error > worst_alg_error[2]:
            worst_alg_error = [ber, no_alg_error, alg_error]
            worst_alg_error_w = w

    print("\n~~~~~~~~~~~~~~~~~~~~~~ Results ~~~~~~~~~~~~~~~~~~~~~~\n")

    print_results(
        "Averages (of " + str(num_iterations) + ")",
        cumulative_ber / num_iterations,
        cumulative_no_alg_error / num_iterations,
        cumulative_alg_error / num_iterations,
        np.true_divide(cumulative_w, num_iterations)
    )

    print_results(
        "Lowest BER",
        best_ber[0],
        best_ber[1],
        best_ber[2],
        best_ber_w
    )

    print_results(
        "Highest BER",
        worst_ber[0],
        worst_ber[1],
        worst_ber[2],
        worst_ber_w
    )

    print_results(
        "Lowest Algae Error",
        best_alg_error[0],
        best_alg_error[1],
        best_alg_error[2],
        best_alg_error_w
    )

    print_results(
        "Highest Algae Error",
        worst_alg_error[0],
        worst_alg_error[1],
        worst_alg_error[2],
        worst_alg_error_w
    )

    print("Magnitude of the difference of weight vectors for highest and lowest BER:")
    print(np.abs(np.add(best_ber_w, -1 * worst_ber_w)))
    print("\n\n")

    print("Magnitude of the difference of weight vectors for highest and lowest algae error:")
    print(np.abs(np.add(best_alg_error_w, -1 * worst_alg_error_w)))

    print("\n\n")

    # print("\n~~~~~~~~~~~~~~~~~~~~~~ Results ~~~~~~~~~~~~~~~~~~~~~~\n")
    # print("BER:", ber)
    # print("No Algae Error Rate:", no_alg_error)
    # print("Algae Error Rate:", alg_error)
    # print("Confusion Matrix:")
    # print(mat_conf)
    # print("Weight vector:")
    # print(w)
    # print()


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
    #     if target_labels[i] == -1:
    #         no_alg += 1
    #         if pred_labels[i] == 1:
    #             no_alg_error += 1
    #     elif target_labels[i] == 1:
    #         alg += 1
    #         if pred_labels[i] == -1:
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
        if (pred_labels[i] == 1) and (target_labels[i] == -1):
            mat_conf[0, 1] += 1
        if (pred_labels[i] == -1) and (target_labels[i] == 1):
            mat_conf[1, 0] += 1
        if (pred_labels[i] == 1) and (target_labels[i] == 1):
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


def print_results(message, ber, no_alg_error, alg_error, w):
    print(message)
    print("BER:", ber)
    print("No Algae Error Rate:", no_alg_error)
    print("Algae Error Rate:", alg_error)
    print("Weight vector:")
    print(w)
    print("\n\n")


if __name__ == "__main__": main()