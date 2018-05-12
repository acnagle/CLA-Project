import numpy as np
import os
import glob


def main():
    np.set_printoptions(threshold=np.inf)  # prints a full matrix rather than an abbreviated matrix

    print('\n\t##### EXECUTING DATA_NORM.PY #####')

    # source directories
    path_matrices = '/Users/Alliot/documents/cla-project/data/matrices/original'
    path_matrices_no_na = '/Users/Alliot/documents/cla-project/data/matrices-no-na/original'

    # destination directories for normalized data
    dest_path_matrices_normalized = '/Users/Alliot/documents/cla-project/data/matrices/normalized/'
    dest_path_matrices_no_na_normalized = '/Users/Alliot/documents/cla-project/data/matrices-no-na/normalized/'

    # get all file paths in matrices and matrices-no-na directories
    files_matrices = [filename for filename in glob.glob(os.path.join(path_matrices, "*.csv"))]
    files_matrices_no_na = [filename for filename in glob.glob(os.path.join(path_matrices_no_na, "*.csv"))]

    # normalize and store all matrices in path_matrices_no_na
    for filename in files_matrices_no_na:
        print('Processing file ' + filename[65:] + ' ...')
        mat = np.genfromtxt(open(filename, 'rb'), delimiter=',', dtype='str')
        mat = remove_empty_entries(mat)
        normalize_data(mat)
        matrix_to_file(mat, filename[65:], dest_path_matrices_no_na_normalized)


# This method removes the empty entries (i.e. ',,,') located at the ends of the .csv files in the matrices_no_na
# directory. new_mat is the matrix with empty entries removed.
def remove_empty_entries(mat):
    new_mat = mat

    for col in range(0, mat.shape[1]):
        if mat[0, col] == '':
            new_mat = mat[:, 0:col]
            break

    return new_mat


# This method normalizes the data in a matrix by finding the largest value in each row and dividing each element in
# row by that value. This will cause each point in the data to be between the 0 and 1. mat is the matrix whose rows will
# be normalized.
def normalize_data(mat):
    for i in range(5, 14):  # rows 5 through 13 contain data that must be normalized
        # norm_arr_str contains the string representation of an entire row that is going to be normalized
        norm_arr_str = mat[i, :]

        # convert all elements in norm_arr to float
        # norm_arr is the float representation of norm_arr_str
        norm_arr = np.zeros(len(norm_arr_str), dtype='float')
        for j in range(0, len(norm_arr)):
            try:
                norm_arr[j] = float(norm_arr_str[j])
            except ValueError:
                print('The value at index ' + str(j) + ' could not be cast to a float.')

        # determine the largest value in norm_arr
        max_val = np.amax(norm_arr)

        if max_val != 0:
            # normalize all the elements by dividing each element by max_val
            for k in range(0, len(norm_arr)):
                norm_arr[k] = norm_arr[k] / max_val

                # store the normalize array back into its respective row in mat
                mat[i, k] = str(norm_arr[k])


# Writes a matrix to a csv file. mat is the matrix being written to a file. filename is the name of the .csv file.
# destination_folder is the path to the destination folder where the .csv file will be stored
def matrix_to_file(mat, filename, destination_folder):
    file = open(destination_folder + filename, 'w')

    for i in range(0, mat.shape[0]):
        for j in range(0, mat.shape[1]):
            if j < mat.shape[1] - 1:
                file.write(mat[i, j] + ',')
            else:
                file.write(mat[i, j] + '\n')

    file.close()


if __name__ == '__main__': main()