import numpy as np
import os
import glob
import errno
import Constants


def main():
    np.set_printoptions(threshold=np.inf)  # prints a full matrix rather than an abbreviated matrix

    print("\n\t##### EXECUTING COMPUTE_EIGENVECTORS.PY #####\n")

    # source directories
    path_matrices_no_na = "/Users/Alliot/documents/cla-project/data/matrices-no-na/normalized"
    path_all_data = "/Users/Alliot/documents/cla-project/data/all-data-no-na/normalized/"

    # destination directories for eigenvectors
    dest_path_matrices_no_na_eigen = "/Users/Alliot/documents/cla-project/data/matrices-no-na/eigenvectors/"
    dest_path_all_data_eigen = "/Users/Alliot/documents/cla-project/data/all-data-no-na/eigenvectors/"

    # get all file paths in matrices-no-na directory
    files_matrices_no_na = [filename for filename in glob.glob(os.path.join(path_matrices_no_na, "*.csv"))]

    # compute eigenvectors, eigenvalues, and singular values of all the matrices in path_matrices_no_na directory
    for filename_w_directory in files_matrices_no_na:
        mat = np.genfromtxt(open(filename_w_directory, "rb"), delimiter=",", dtype="str")
        mat = mat.astype(dtype=float)
        filename = filename_w_directory[67:]
        print("Processing file " + filename + " ...")
        eigv1, eigv2, eigv3, eigvals, svdvals = get_eigenvectors(mat)

        # final_directory is the final location of mat and its eigenvectors
        final_directory = dest_path_matrices_no_na_eigen + filename[:-4] + "/"
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

        matrix_to_file(mat, filename=filename, destination_folder=final_directory)
        vector_to_file(eigv1, filename=eigv1_filename, destination_folder=final_directory)
        vector_to_file(eigv2, filename=eigv2_filename, destination_folder=final_directory)
        vector_to_file(eigv3, filename=eigv3_filename, destination_folder=final_directory)
        vector_to_file(eigvals, filename=eigvals_filename, destination_folder=final_directory)
        vector_to_file(svdvals, filename=svdvals_filename, destination_folder=final_directory)

    # compute eigenvectors, eigenvalues, and singular values of all the matricies in path_all_data directory
    # get all file paths in all-data-no-na directory
    files_matrices_all_data = [filename for filename in glob.glob(os.path.join(path_all_data, "*.csv"))]
    for filename_w_directory in files_matrices_all_data:
        mat = np.genfromtxt(open(filename_w_directory, "rb"), delimiter=",", dtype="str")
        mat = mat.astype(dtype=float)
        filename = filename_w_directory[67:]
        print("Processing file " + filename + " ...")
        eigv1, eigv2, eigv3, eigvals, svdvals = get_eigenvectors(mat)

        # final_directory is the final location of mat and its eigenvectors
        final_directory = dest_path_all_data_eigen + filename[:-4] + "/"
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

        matrix_to_file(mat, filename=filename, destination_folder=final_directory)
        vector_to_file(eigv1, filename=eigv1_filename, destination_folder=final_directory)
        vector_to_file(eigv2, filename=eigv2_filename, destination_folder=final_directory)
        vector_to_file(eigv3, filename=eigv3_filename, destination_folder=final_directory)
        vector_to_file(eigvals, filename=eigvals_filename, destination_folder=final_directory)
        vector_to_file(svdvals, filename=svdvals_filename, destination_folder=final_directory)

    print("\n")


# This method takes a matrix mat, which is a matrix of string, and converts it into a matrix of float so the data
# can be analyze numerically. It also trims any non-numerical entries, such as location, ID, etc. new_mat is returned,
# which is the trimmed version of mat with float elements. first is the first row index of numerical entires, and
# last is the last row index
def matrix_str_to_float(mat, first, last):
    new_mat = np.zeros(shape=(last - first, mat.shape[Constants.COLUMNS]), dtype=float)
    for i in range(first, last):
        for j in range(0, mat.shape[Constants.COLUMNS]):
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
    # a = mat.dot(mat.T)
    # w, v = np.linalg.eig(a)
    #
    # w = np.around(w, decimals=10)
    # v = np.around(v, decimals=10)
    #
    # w = np.real(w)
    # v = np.real(v)
    #
    # idx = w.argsort()[::-1]
    # w = w[idx]
    # v = v[:, idx]
    #
    # eigv1 = v[:, 0]
    # eigv2 = v[:, 1]
    # eigv3 = v[:, 2]
    # eigvals = w
    #
    # svdvals = np.diag(np.sqrt(w))

    cov_mat = np.cov(mat)  # calculate covariance matrix
    u, s, v = np.linalg.svd(cov_mat, full_matrices=True)

    eigv1 = u[:, 0]
    eigv2 = u[:, 1]
    eigv3 = u[:, 2]

    svdvals = s

    eigvals = np.diag(np.sqrt(svdvals))

    return eigv1, eigv2, eigv3, eigvals, svdvals


# Writes a matrix to a .csv file. mat is the matrix being written to a file. filename is the name
# of the .csv file. destination_folder is the path to the destination folder where the .csv file will be stored
def matrix_to_file(mat, filename, destination_folder):
    file = open(destination_folder + filename, "w")

    for i in range(0, mat.shape[Constants.ROWS]):
        for j in range(0, mat.shape[Constants.COLUMNS]):
            if j < mat.shape[Constants.COLUMNS] - 1:
                file.write(str(mat[i, j]) + ",")
            else:
                file.write(str(mat[i, j]) + "\n")

    file.close()


# This method writes a vector to a .csv file. vec is the vector being written to a file. filename is the name
# # of the .csv file. destination_folder is the path to the destination folder where the .csv file will be stored
def vector_to_file(vec, filename, destination_folder):
    file = open(destination_folder + filename, "w")

    for i in range(0, vec.shape[Constants.ROWS]):
            file.write(str(vec[i]) + "\n")

    file.close()


if __name__ == "__main__": main()