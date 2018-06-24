import numpy as np
import math
import os
import glob
import errno

# the height of the matrices. ie the number of measurements per sample. IMPORTANT NOTE: The .csv files being read in
# by this code has 15 rows. The last row is a "poor water quality flag" (binary) that is 1 if the turbidity is below
# 50 and 0 otherwise. By choosing num_rows = 14, I'm eliminating this row.
num_rows = 14


def main():
    np.set_printoptions(threshold=np.inf)  # prints a full matrix rather than an abbreviated matrix

    print("\n\t##### EXECUTING EIGEN_NO_ALGAE_INDICATOR.PY #####")

    # source directories
    path_matrices_no_na = "/Users/Alliot/documents/cla-project/data/matrices-no-na/normalized"
    path_all_data = "/Users/Alliot/documents/cla-project/data/all-data-no-na/normalized/"

    # destination directories for eigenvectors
    dest_path_matrices_no_na = "/Users/Alliot/documents/cla-project/data/matrices-no-na/eigen-no-alg-ind/"
    dest_path_all_data = "/Users/Alliot/documents/cla-project/data/all-data-no-na/eigen-no-alg-ind/"

    # get all file paths in matrices-no-na directory
    files_matrices_no_na = [filename for filename in glob.glob(os.path.join(path_matrices_no_na, "*.csv"))]

    # compute eigenvectors, eigenvalues, and singular values of all the matricies in path_matrices_no_na directory
    for filename_w_directory in files_matrices_no_na:
        mat = np.genfromtxt(open(filename_w_directory, "rb"), delimiter=",", dtype="str")
        mat = matrix_str_to_float(mat, 5, num_rows)

        # Remove algae indicator (row 2 (index 1) for this set of matrices)
        mat = np.delete(mat, 1, 0)  # delete algal indicator (algalBloomSheen)

        # get filename of mat and compute eigenvectors, eigenvalues, and svd values
        filename = filename_w_directory[67:]
        print("Processing file " + filename + " ...")
        eigv1, eigv2, eigv3, eigvals, svdvals = get_eigenvectors(mat)

        # final_directory is the final location of mat and its eigenvectors
        final_directory = dest_path_matrices_no_na + filename[:-4] + "/"
        eigv1_filename = filename[:-4] + "_eigv1.csv"
        eigv2_filename = filename[:-4] + "_eigv2.csv"
        eigv3_filename = filename[:-4] + "_eigv3.csv"
        eigvals_filename = filename[:-4] + "_eigenvalues.csv"
        svdvals_filename = filename[:-4] + "_singularvalues.csv"

        # if final_directory does not exist, create it
        if not os.path.exists(final_directory):
            try:
                os.makedirs(final_directory)
            except OSError as e:
                if e.errno != errno.EEXIST:
                    raise

        matrix_to_file(mat, filename, final_directory)
        vector_to_file(eigv1, eigv1_filename, final_directory)
        vector_to_file(eigv2, eigv2_filename, final_directory)
        vector_to_file(eigv3, eigv3_filename, final_directory)
        vector_to_file(eigvals, eigvals_filename, final_directory)
        vector_to_file(svdvals, svdvals_filename, final_directory)

    # compute eigenvectors, eigenvalues, and singular values of all the matricies in path_all_data directory
    # get all file paths in all-data-no-na directory
    files_matrices_all_data = [filename for filename in glob.glob(os.path.join(path_all_data, "*.csv"))]
    for filename_w_directory in files_matrices_all_data:
        mat = np.genfromtxt(open(filename_w_directory, "rb"), delimiter=",", dtype="str")
        mat = matrix_str_to_float(mat, 2, 15)

        # Remove algae indicator (row 2 (index 1) for this set of matrices)
        mat = np.delete(mat, 1, 0)  # delete algal indicator (algalBloomSheen)

        # get filename of mat and compute eigenvectors, eigenvalues, and svd values
        filename = filename_w_directory[67:]
        print("Processing file " + filename + " ...")
        eigv1, eigv2, eigv3, eigvals, svdvals = get_eigenvectors(mat)

        # final_directory is the final location of mat and its eigenvectors
        final_directory = dest_path_all_data + filename[:-4] + "/"
        eigv1_filename = filename[:-4] + "_eigv1.csv"
        eigv2_filename = filename[:-4] + "_eigv2.csv"
        eigv3_filename = filename[:-4] + "_eigv3.csv"
        eigvals_filename = filename[:-4] + "_eigenvalues.csv"
        svdvals_filename = filename[:-4] + "_singularvalues.csv"

        # if final_directory does not exist, create it
        if not os.path.exists(final_directory):
            try:
                os.makedirs(final_directory)
            except OSError as e:
                if e.errno != errno.EEXIST:
                    raise

        matrix_to_file(mat, filename, final_directory)
        vector_to_file(eigv1, eigv1_filename, final_directory)
        vector_to_file(eigv2, eigv2_filename, final_directory)
        vector_to_file(eigv3, eigv3_filename, final_directory)
        vector_to_file(eigvals, eigvals_filename, final_directory)
        vector_to_file(svdvals, svdvals_filename, final_directory)


# This method takes a matrix mat, which is a matrix of string, and converts it into a matrix of float so the data
# can be analyze numerically. It also trims any non-numerical entries, such as location, ID, etc. new_mat is returned,
# which is the trimmed version of mat with float elements. first is the first row index of numerical entires, and
# last is the last row index
def matrix_str_to_float(mat, first, last):
    new_mat = np.zeros((last - first, mat.shape[1]), dtype=float)
    for i in range(first, last):
        for j in range(0, mat.shape[1]):
            try:
                new_mat[i-first, j] = float(mat[i, j])
            except ValueError:
                print("The value at index " + str(j) + " could not be cast to a float.")

    return new_mat


# This method takes a matrix mat and returns the three eigenvectors corresponding to the largest 3
# eigenvalues. The vectors are returned in three separate vectors: eigv1 for the vector corresponding the largest
# eigenvalue, eigv2 for the second largest, and eigv3 for the third largest. This method also returns eigvals,
# which is the vector of sorted (from greatest to least) eigenvalues, and svdvals, which is the vector
# of singular values
def get_eigenvectors(mat):
    # compute the eigenvalues and store in w, and compute vectors and store them in v.
    a = mat.dot(mat.T)
    w, v = np.linalg.eig(a)

    w = np.around(w, decimals=10)
    v = np.around(v, decimals=10)

    w = np.real(w)
    v = np.real(v)

    idx = w.argsort()[::-1]
    w = w[idx]
    v = v[:, idx]

    eigv1 = v[0]
    eigv2 = v[1]
    eigv3 = v[2]
    eigvals = w

    svdvals = np.sqrt(w)

    return eigv1, eigv2, eigv3, eigvals, svdvals


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


# This method writes a vector to a .csv file. vec is the vector being written to a file. filename is the name
# # of the .csv file. destination_folder is the path to the destination folder where the .csv file will be stored
def vector_to_file(vec, filename, destination_folder):
    file = open(destination_folder + filename, "w")

    for i in range(0, vec.shape[0]):
            file.write(str(vec[i]) + "\n")

    file.close()


if __name__ == "__main__": main()