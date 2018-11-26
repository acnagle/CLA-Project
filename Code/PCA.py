import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys
import Constants


def main():
    np.set_printoptions(threshold=np.inf)  # prints a full matrix rather than an abbreviated matrix

    print("\n\t##### EXECUTING PCA.PY #####\n")

    print("Reading in data set ... ")
    data_sets_path = "/Users/Alliot/Documents/CLA-Project/Data/data-sets/"
    X = np.load(data_sets_path + "data_2017_summer.npy")
    y = np.load(data_sets_path + "data_2017_summer_labels.npy")

    X = np.concatenate((X, np.ones(shape=(X.shape[0], 1))), axis=Constants.COLUMNS)

    print("Removing no algae points ... ")
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
    sample_bias = 0   # adjust the difference in the number of the two types of samples (no_alg and alg)
    while num_no_alg != (num_alg - sample_bias):
        # circle through the data sets until the difference of num_no_alg and num_alg equals
        # the value specified by sample_bias
        if idx == (len(y) - 1):
            idx = 0

        if y[idx] == -1:
            if np.random.rand() >= 0.5:     # remove this sample with some probability
                y = np.delete(y, obj=idx)
                X = np.delete(X, obj=idx, axis=Constants.ROWS)
                num_no_alg -= 1
            else:
                idx += 1
        else:
            idx += 1

    X = preprocessing.scale(X, axis=1)

    S = (1/(X.shape[0])) * np.matmul(np.transpose(X), X)
    W, V = np.linalg.eig(S)
    Y = np.matmul(X, V[:, 0:3])

    # U, S, VH = np.linalg.svd(X)
    # Pu = np.matmul(U, np.transpose(U))
    # Y = np.matmul(Pu, X)
    # Y = Y[:, 0:3]
    # a = np.linspace(start=-1, stop=1, num=10)
    # a1 = VH[0, 0] * a
    # b1 = VH[0, 1] * a
    # v1 = VH[0, 2] * a
    # # v2 = VH[1, :] * a
    # # v3 = VH[2, :] * a

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for i in range(0, Y.shape[Constants.ROWS]):
        if y[i] == -1:
            c = 'g'
        elif y[i] == 1:
            c = 'r'
        else:
            c = 'b'
        ax.scatter(Y[i, 0], Y[i, 1], Y[i, 2], c=c)

    # plt.plot(a, a, v1)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    plt.show()


if __name__ == "__main__": main()