import numpy as np
import math
import os
import glob
import errno


def main():
    np.set_printoptions(threshold=np.inf)  # prints a full matrix rather than an abbreviated matrix

    print('\n\t##### EXECUTING COMPUTE_EIGENVECTORS.PY #####')

    # source directories
    path_matrices_no_na = '/Users/Alliot/documents/cla project/data/matrices-no-na/normalized'
    path_all_data = \
        '/Users/Alliot/documents/cla project/data/all-data-no-na/normalized/algal_bloom_locations_summaries_norm.csv'

    # destination directories for eigenvectors
    dest_path_matrices_no_na_eigen = '/Users/Alliot/documents/cla project/data/matrices-no-na/eigenvectors/'
    dest_path_all_data_eigen = '/Users/Alliot/documents/cla project/data/all-data-no-na/eigenvectors/'

    # get all file paths in matrices-no-na directory
    files_matrices_no_na = [filename for filename in glob.glob(os.path.join(path_matrices_no_na, '*.csv'))]

    # compute eigenvectors, eigenvalues, and singular values of all the matricies in path_matrices_no_na directory
    for filename_w_directory in files_matrices_no_na:
        mat = np.genfromtxt(open(filename_w_directory, 'rb'), delimiter=',', dtype='str')
        mat = matrix_str_to_float(mat, 5, 15)
        filename = filename_w_directory[67:]
        print('Processing file ' + filename + ' ...')
        eigv1, eigv2, eigv3, eigvals, svdvals = get_eigenvectors(mat)

        # final_directory is the final location of mat and its eigenvectors
        final_directory = dest_path_matrices_no_na_eigen + filename[:-4] + '/'
        eigv1_filename = filename[:-4] + '_eigv1.csv'
        eigv2_filename = filename[:-4] + '_eigv2.csv'
        eigv3_filename = filename[:-4] + '_eigv3.csv'
        eigvals_filename = filename[:-4] + '_eigenvalues.csv'
        svdvals_filename = filename[:-4] + '_singularvalues.csv'

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

    # compute eigenvectors, eigenvalues, and singular values of all the matricies in path_matrices_no_na directory
    mat = np.genfromtxt(open(path_all_data, 'rb'), delimiter=',', dtype='str')
    mat = matrix_str_to_float(mat, 2, 16)
    filename = path_all_data[67:]
    print('Processing file ' + filename + ' ...')
    eigv1, eigv2, eigv3, eigvals, svdvals = get_eigenvectors(mat)

    # final_directory is the final location of mat and its eigenvectors
    final_directory = dest_path_matrices_no_na_eigen + filename[:-4] + '/'
    eigv1_filename = filename[:-8] + 'eigv1.csv'
    eigv2_filename = filename[:-8] + 'eigv2.csv'
    eigv3_filename = filename[:-8] + 'eigv3.csv'
    eigvals_filename = filename[:-8] + 'eigenvalues.csv'
    svdvals_filename = filename[:-8] + 'singularvalues.csv'

    # if final_directory does not exist, create it
    if not os.path.exists(dest_path_all_data_eigen):
        try:
            os.makedirs(dest_path_all_data_eigen)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

    matrix_to_file(mat, filename, dest_path_all_data_eigen)
    vector_to_file(eigv1, eigv1_filename, dest_path_all_data_eigen)
    vector_to_file(eigv2, eigv2_filename, dest_path_all_data_eigen)
    vector_to_file(eigv3, eigv3_filename, dest_path_all_data_eigen)
    vector_to_file(eigvals, eigvals_filename, dest_path_all_data_eigen)
    vector_to_file(svdvals, svdvals_filename, dest_path_all_data_eigen)

# This method takes a matrix mat, which is a matrix of string, and converts it into a matrix of float so the data
# can be analyze numerically. It also trims any non-numerical entries, such as location, ID, etc. new_mat is returned,
# which is the trimmed version of mat with float elements. first is the first row index of numerical entires, and
# last is the last row index
def matrix_str_to_float(mat, first, last):
    new_mat = np.zeros((last - first, mat.shape[1]), dtype=float)
    for i in range(first, last):  # rows 5 through 14 contain numerical data
        for j in range(0, mat.shape[1]):
            try:
                new_mat[i-first, j] = float(mat[i, j])
            except ValueError:
                print('The value at index ' + str(j) + ' could not be cast to a float.')

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
    file = open(destination_folder + filename, 'w')

    for i in range(0, mat.shape[0]):
        for j in range(0, mat.shape[1]):
            if j < mat.shape[1] - 1:
                file.write(str(mat[i, j]) + ',')
            else:
                file.write(str(mat[i, j]) + '\n')

    file.close()


# This method writes a vector to a .csv file. vec is the vector being written to a file. filename is the name
# # of the .csv file. destination_folder is the path to the destination folder where the .csv file will be stored
def vector_to_file(vec, filename, destination_folder):
    file = open(destination_folder + filename, 'w')

    for i in range(0, vec.shape[0]):
            file.write(str(vec[i]) + '\n')

    file.close()


if __name__ == '__main__': main()