import numpy as np
import glob
import os
import errno


def main():
    np.set_printoptions(threshold=np.inf)  # prints a full matrix rather than an abbreviated matrix

    print('\n\t##### EXECUTING DATA_PROJECTION.PY #####')

    # source directories
    path_matrices_no_na_eigen_no_alg_ind = '/Users/Alliot/documents/cla project/data/matrices-no-na/eigen-no-alg-ind/'
    path_all_data_eigen_no_alg_ind = '/Users/Alliot/documents/cla project/data/all-data-no-na/eigen-no-alg-ind/'

    # destination directories for projection vectors
    dest_path_matrices_no_na_proj = '/Users/Alliot/documents/cla project/data/matrices-no-na/projections/'
    dest_path_all_data_proj = '/Users/Alliot/documents/cla project/data/all-data-no-na/projections/'

    # get all subdirectories within path_matrices_no_na_eigen_no_alg_ind
    dirnames = [x[0] for x in os.walk(path_matrices_no_na_eigen_no_alg_ind)]

    # create projection vectors for matrices-no-na directory
    for directory in dirnames[1:]:
        # get the eigenvectors filenames in the directory
        eigv1_name = glob.glob(os.path.join(directory, '*eigv1.csv'))
        eigv2_name = glob.glob(os.path.join(directory, '*eigv2.csv'))
        eigv3_name = glob.glob(os.path.join(directory, '*eigv3.csv'))

        # get the matrix filename in the directory
        mat_name = glob.glob(os.path.join(directory, '*matrix.csv'))

        # open eigenvectors and matrix files in directory
        eigv1 = np.genfromtxt(open(eigv1_name[0], 'rb'), delimiter=',', dtype=str)
        eigv2 = np.genfromtxt(open(eigv2_name[0], 'rb'), delimiter=',', dtype=str)
        eigv3 = np.genfromtxt(open(eigv3_name[0], 'rb'), delimiter=',', dtype=str)
        mat = np.genfromtxt(open(mat_name[0], 'rb'), delimiter=',', dtype=str)

        # get mat_name without directory
        for i in range(0, len(mat_name[0])):
            if mat_name[0][i] == '/':
                max_idx = i + 1  # mat_name will be searched through for the last '/'. knowing the index of the last '/'
                # will be useful in determining the mat_name

        mat_name = mat_name[0][max_idx:]
        print('Processing file ' + mat_name + ' ...')

        # final_directory is the final location of mat and its eigenvectors
        final_directory = dest_path_matrices_no_na_proj + mat_name[:-4] + '/'
        proj_name = mat_name[:-4] + '_proj'

        # convert eigenvectors and mat from str entries to float entries
        eigv1 = eigv1.astype(float)
        eigv2 = eigv2.astype(float)
        eigv3 = eigv3.astype(float)
        mat = mat.astype(float)

        # determine the rows of proj_mat_3d, proj_mat_2d, and proj_mat_1d, where proj_mat_ed, for example,
        # is the matrix containing the "projection" of the data matrix, mat, on V = [eigv1 eigv2 eigv3].
        # Each column in proj_mat corresponds to the projection of a particular row in
        # mat on V. let V = [eigv1 eigv2 eigv3], and let A represent mat so that A_i is the ith column in mat.
        # (U^T)A_i = [(eigv1^T)A_i; (eigv2^T)A_i; (eigv3^T)A_i] = proj_mat_3d[i]
        proj_mat_3d = np.zeros((3, mat.shape[1]), dtype=float)
        proj_mat_2d = np.zeros((2, mat.shape[1]), dtype=float)
        proj_mat_1d = np.zeros((1, mat.shape[1]), dtype=float)

        for i in range(0, mat.shape[1]):
            proj_mat_3d[0, i] = eigv1.T.dot(mat[:, i])
            proj_mat_3d[1, i] = eigv2.T.dot(mat[:, i])
            proj_mat_3d[2, i] = eigv3.T.dot(mat[:, i])

            proj_mat_2d[0, i] = eigv1.T.dot(mat[:, i])
            proj_mat_2d[1, i] = eigv2.T.dot(mat[:, i])

            proj_mat_1d[0, i] = eigv1.T.dot(mat[:, i])

        # if final_directory does not exist, create it
        if not os.path.exists(final_directory):
            try:
                os.makedirs(final_directory)
            except OSError as e:
                if e.errno != errno.EEXIST:
                    raise

        matrix_to_file(mat, mat_name, final_directory)
        matrix_to_file(proj_mat_3d, proj_name + '_3d.csv', final_directory)
        matrix_to_file(proj_mat_2d, proj_name + '_2d.csv', final_directory)
        matrix_to_file(proj_mat_1d, proj_name + '_1d.csv', final_directory)

    # create projection vectors for all-data directory
    # get the eigenvectors filenames in the directory
    eigv1_name = glob.glob(os.path.join(path_all_data_eigen_no_alg_ind, '*eigv1.csv'))
    eigv2_name = glob.glob(os.path.join(path_all_data_eigen_no_alg_ind, '*eigv2.csv'))
    eigv3_name = glob.glob(os.path.join(path_all_data_eigen_no_alg_ind, '*eigv3.csv'))

    # get the matrix filename in the directory
    mat_name = glob.glob(os.path.join(path_all_data_eigen_no_alg_ind, '*norm.csv'))

    # open eigenvectors and matrix files in directory
    eigv1 = np.genfromtxt(open(eigv1_name[0], 'rb'), delimiter=',', dtype=str)
    eigv2 = np.genfromtxt(open(eigv2_name[0], 'rb'), delimiter=',', dtype=str)
    eigv3 = np.genfromtxt(open(eigv3_name[0], 'rb'), delimiter=',', dtype=str)
    mat = np.genfromtxt(open(mat_name[0], 'rb'), delimiter=',', dtype=str)

    # get mat_name without directory
    for i in range(0, len(mat_name[0])):
        if mat_name[0][i] == '/':
            max_idx = i + 1  # mat_name will be searched through for the last '/'. knowing the index of the last '/'
            # will be useful in determining the mat_name

    mat_name = mat_name[0][max_idx:]
    print('Processing file ' + mat_name + ' ...')

    # final_directory is the final location of mat and its eigenvectors
    final_directory = dest_path_all_data_proj
    proj_name = mat_name[:-4] + '_proj'

    # convert eigenvectors and mat from str entries to float entries
    eigv1 = eigv1.astype(float)
    eigv2 = eigv2.astype(float)
    eigv3 = eigv3.astype(float)
    mat = mat.astype(float)

    # determine the rows of proj_mat_3d, proj_mat_2d, and proj_mat_1d, where proj_mat_ed, for example,
    # is the matrix containing the "projection" of the data matrix, mat, on V = [eigv1 eigv2 eigv3].
    # Each column in proj_mat corresponds to the projection of a particular row in
    # mat on V. let V = [eigv1 eigv2 eigv3], and let A represent mat so that A_i is the ith column in mat.
    # (U^T)A_i = [(eigv1^T)A_i; (eigv2^T)A_i; (eigv3^T)A_i] = proj_mat_3d[i]
    proj_mat_3d = np.zeros((3, mat.shape[1]), dtype=float)
    proj_mat_2d = np.zeros((2, mat.shape[1]), dtype=float)
    proj_mat_1d = np.zeros((1, mat.shape[1]), dtype=float)

    for i in range(0, mat.shape[1]):
        proj_mat_3d[0, i] = eigv1.T.dot(mat[:, i])
        proj_mat_3d[1, i] = eigv2.T.dot(mat[:, i])
        proj_mat_3d[2, i] = eigv3.T.dot(mat[:, i])

        proj_mat_2d[0, i] = eigv1.T.dot(mat[:, i])
        proj_mat_2d[1, i] = eigv2.T.dot(mat[:, i])

        proj_mat_1d[0, i] = eigv1.T.dot(mat[:, i])

    # if final_directory does not exist, create it
    if not os.path.exists(final_directory):
        try:
            os.makedirs(final_directory)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

    matrix_to_file(mat, mat_name, final_directory)
    matrix_to_file(proj_mat_3d, proj_name + '_3d.csv', final_directory)
    matrix_to_file(proj_mat_2d, proj_name + '_2d.csv', final_directory)
    matrix_to_file(proj_mat_1d, proj_name + '_1d.csv', final_directory)


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


if __name__ == '__main__': main()