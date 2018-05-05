import numpy as np
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def main():
    np.set_printoptions(threshold=np.inf)  # prints a full matrix rather than an abbreviated matrix

    print('\n\t##### EXECUTING DATA_VISUALIZATION.PY #####')

    # source directories
    path_matrices_no_na_proj = '/Users/Alliot/documents/cla project/data/matrices-no-na/projections/'

    # get all subdirectories within path_matrices_no_na_projections
    dirnames = [x[0] for x in os.walk(path_matrices_no_na_proj)]

    # this string is a substring of all yearly matrices
    yearly_substr = '_year_matrix'

    filename_w_directory = np.empty(3, dtype=(str, 105))

    # get the filenames w/ directory of the 3 matrices for 2015, 2016, and 2017
    j = 0   # j is used to index filename_w_directory
    for i in range(0, len(dirnames)):
        if yearly_substr in dirnames[i]:
            if 'all' not in dirnames[i]:
                filename_w_directory[j] = dirnames[i] + '/' + dirnames[i][68:] + '.csv'
                j = j + 1

    proj_mat1 = np.genfromtxt(open(filename_w_directory[0], 'rb'), delimiter=',', dtype=float)
    proj_mat2 = np.genfromtxt(open(filename_w_directory[1], 'rb'), delimiter=',', dtype=float)
    proj_mat3 = np.genfromtxt(open(filename_w_directory[2], 'rb'), delimiter=',', dtype=float)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(proj_mat1[1, :], proj_mat1[2, :], proj_mat1[3, :], alpha=0.25)
    plt.title('Year 2015')
    plt.show()

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(proj_mat2[1, :], proj_mat2[2, :], proj_mat2[3, :], alpha=0.25)
    plt.title('Year 2016')
    plt.show()

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(proj_mat3[1, :], proj_mat3[2, :], proj_mat3[3, :], alpha=0.25)
    plt.title('Year 2017')
    plt.show()

if __name__ == '__main__': main()