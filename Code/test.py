import numpy as np
import os


def main():
    np.set_printoptions(threshold=np.inf)  # prints a full matrix rather than an abbreviated matrix

    # test predictions
    dataset = [[2.7810836, 2.550537003, 0],
               [1.465489372, 2.362125076, 0],
               [3.396561688, 4.400293529, 0],
               [1.38807019, 1.850220317, 0],
               [3.06407232, 3.005305973, 0],
               [7.627531214, 2.759262235, 1],
               [5.332441248, 2.088626775, 1],
               [6.922596716, 1.77106367, 1],
               [8.675418651, -0.242068655, 1],
               [7.673756466, 3.508563011, 1]]

    weights = [-0.1, 0.20653640140000007, -0.23418117710000003]

    for row in dataset:
        prediction = predict(row, weights)
        print("Expected=%d, Predicted=%d" % (row[-1], prediction))


    # A = np.array([
    #     [1, 1, 1, 1, 1],
    #     [0, 0, 0, 0, 0],
    #     [1, 1, 1, 1, 1],
    #     [1, 1, 1, 1, 1],
    #     [1, 1, 1, 1, 1],
    #     [1, 1, 1, 1, 1],
    #     [1, 1, 1, 1, 1],
    #     [0.9213, 0.8494, 0.9123, 0.9820, 1],
    #     [0.675, 0.9125, 0.775, 0.7875, 1],
    #     [0, 0, 0, 0, 0]
    # ])
    #
    # print(A)
    #
    # B = A.dot(A.T)
    #
    # print(B)
    #
    # w, v = np.linalg.eig(B)
    #
    # w = np.around(w, decimals=6)
    # v = np.around(v, decimals=6)
    #
    # idx = w.argsort()[::-1]
    # w = w[idx]
    # v = v[:, idx]
    # print(w)
    # print(v)


    # A_trans = A.transpose()
    # C = np.zeros((A.shape[0], A.shape[0]), dtype=float)
    #
    # for i in range(0, A.shape[0]):
    #     C[:, i] = np.outer(A[:, i], A_trans[:, i])

    # C = np.outer(A, A.T)

    # w, v = np.linalg.eig(B)

    # print(w)

    # print(v)

    # A = np.array([[1, 2, 3, 0, 9, 7, 4], [5, 6, 7, 1, 3, 2, 6], [9, 8, 3, 6, 5, 4, 3], [2, 4, 3, 5, 0, 1, 2]])
    # # B = A.dot(A.T)
    # # w, v = np.linalg.eig(B)
    #
    # print(A)
    # print(np.delete(A, 2, 0))

    # for dirpath, dirnames, filenames in os.walk('/Users/Alliot/documents/cla project/data/matrices-no-na/eigenvectors/'):
    #     for filename in [f for f in filenames if f.endswith('matrix.csv')]:
    #         print(os.path.join(dirpath, filename))

    # a = [[1, 2], [5, 6], [9, 10]]
    #
    # b = np.pad(a, ((0, 0), (0, 1)), 'constant', constant_values=0)
    #
    # # c,d = np.linalg.eig(b)
    #
    # print(b)
    # print(c)
    # print(d)

    # mat1 = np.zeros([7, 7], dtype=int)
    # # mat2 = np.empty([5, 5], dtype=int)
    # # mat3 = np.empty([5, 5], dtype=int)
    #
    # for i in range(0, 7):
    #     for j in range(0, 5):
    #         mat1[i, j] = j + 1
    #         # mat2[i, j] = 2
    #         # mat3[i, j] = 3
    #
    # new_col = np.zeros([7, 1], dtype=int)
    #
    # print(mat1)
    # shift_column(2, 5, mat1, new_col[0])
    # print()
    # print(mat1)
    #
    # print()
    # print('end first method call')
    #
    # shift_column(3, 6, mat1, new_col[0])
    # print()
    # print(mat1)
    #
    # shift_column(0, 6, mat1, new_col[0])
    # print()
    # print(mat1)
    # # print(mat1)
    # # print(mat2)
    # # print(mat3)


# This method shifts a column in the matrix to make room for the insertion of another column. index is the index
# of the column to be shifted to the right by one. TotalCol is the total number of non-empty columns in mat.
# mat is the matrix whose columns are being shifted.
def shift_column(index, total_col, mat, new_col):
    i = total_col

    if total_col < mat.shape[1]:
        while i > index:
            mat[:, i] = mat[:, i - 1]
            i = i - 1

        # insert the last column (newest data read in from file) into the open column
        mat[:, index] = new_col


# Make a prediction with weights
def predict(row, weights):
    activation = weights[0]
    for i in range(len(row)-1):
        activation += weights[i + 1] * row[i]
    return 1.0 if activation >= 0.0 else 0.0


if __name__ == "__main__": main()
