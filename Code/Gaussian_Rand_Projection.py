import numpy as np
import glob
import os
import errno

def main():
    np.set_printoptions(threshold=np.inf)  # prints a full matrix rather than an abbreviated matrix

    print('\n\t##### EXECUTING GUASSIAN_RAND_PROJECTION.PY #####')

    # source directories
    path_matrices_no_na_eigen_no_alg_ind = '/Users/Alliot/documents/cla-project/data/matrices-no-na/eigen-no-alg-ind/'
    path_all_data_eigen_no_alg_ind = '/Users/Alliot/Documents/CLA-Project/Data/all-data-no-na/projections/' \
                                     'algal_bloom_locations_summaries_norm.csv'

    # destination directories for projection vectors
    dest_path_matrices_no_na_proj = '/Users/Alliot/documents/cla-project/data/matrices-no-na/gaussian-randn-projections/'
    dest_path_all_data_proj = '/Users/Alliot/documents/cla-project/data/all-data-no-na/gaussian-randn-projections/'

    # get all subdirectories within path_matrices_no_na_eigen_no_alg_ind
    dirnames = [x[0] for x in os.walk(path_matrices_no_na_eigen_no_alg_ind)]

    # create projection vectors for matrices-no-na directory
    for directory in dirnames[1:]:
        # get the matrix filename in the directory
        mat_name = glob.glob(os.path.join(directory, '*matrix.csv'))

        # open matrix files in directory
        mat = np.genfromtxt(open(mat_name[0], 'rb'), delimiter=',', dtype=str)

        # get mat_name without directory
        for i in range(0, len(mat_name[0])):
            if mat_name[0][i] == '/':
                max_idx = i + 1  # mat_name will be searched through for the last '/'. knowing the index of the last '/'
                                 # will be useful in determining the mat_name

        mat_name = mat_name[0][max_idx:]
        print('Processing file ' + mat_name + ' ...')

        # final_directory is the final location of mat
        final_directory = dest_path_matrices_no_na_proj + mat_name[:-4] + '/'
        proj_name = mat_name[:-4] + '_proj'

        mat = mat.astype(float) # convert mat from str entries to float entries

        randn_mat_3d = np.random.randn(3, mat.shape[0])
        proj_mat_3d = randn_mat_3d.dot(mat)

        # if final_directory does not exist, create it
        if not os.path.exists(final_directory):
            try:
                os.makedirs(final_directory)
            except OSError as e:
                if e.errno != errno.EEXIST:
                    raise

        matrix_to_file(mat, mat_name, final_directory)
        matrix_to_file(proj_mat_3d, proj_name + '_3d.csv', final_directory)


    # Create projection matrix for All_year_matrix
    # get the matrix filename in the directory
    mat_name = glob.glob(path_all_data_eigen_no_alg_ind)

    # open matrix files in directory
    mat = np.genfromtxt(open(mat_name[0], 'rb'), delimiter=',', dtype=str)

    # get mat_name without directory
    for i in range(0, len(mat_name[0])):
        if mat_name[0][i] == '/':
            max_idx = i + 1  # mat_name will be searched through for the last '/'. knowing the index of the last '/'
            # will be useful in determining the mat_name

    mat_name = mat_name[0][max_idx:]
    print('Processing file ' + mat_name + ' ...')

    # final_directory is the final location of mat
    final_directory = dest_path_all_data_proj
    proj_name = mat_name[:-4] + '_proj'

    mat = mat.astype(float)  # convert mat from str entries to float entries

    randn_mat_3d = np.random.randn(3, mat.shape[0])
    proj_mat_3d = randn_mat_3d.dot(mat)

    # if final_directory does not exist, create it
    if not os.path.exists(final_directory):
        try:
            os.makedirs(final_directory)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

    matrix_to_file(mat, mat_name, final_directory)
    matrix_to_file(proj_mat_3d, proj_name + '_3d.csv', final_directory)


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