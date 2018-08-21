import numpy as np
import glob
import os
import errno
import Constants


def main():
    np.set_printoptions(threshold=np.inf)  # prints a full matrix rather than an abbreviated matrix

    print("\n\t##### EXECUTING DATA_PROJECTION.PY #####\n")

    # source directories
    path_matrices_no_na_eigen_no_alg_ind = "/Users/Alliot/documents/cla-project/data/matrices-no-na/eigen-no-alg-ind/"
    path_all_data_eigen_no_alg_ind = "/Users/Alliot/documents/cla-project/data/all-data-no-na/eigen-no-alg-ind/"
    path_matrices_no_na_eigen_w_alg_ind = "/Users/Alliot/documents/cla-project/data/matrices-no-na/eigenvectors/"
    path_all_data_eigen_w_alg_ind = "/Users/Alliot/documents/cla-project/data/all-data-no-na/eigenvectors/"

    # destination directories for projection vectors
    dest_path_matrices_no_na_proj = "/Users/Alliot/documents/cla-project/data/matrices-no-na/projections/"
    dest_path_all_data_proj = "/Users/Alliot/documents/cla-project/data/all-data-no-na/projections/"

    # get all subdirectories within path_matrices_no_na_eigen_no_alg_ind
    dirnames_no_alg_ind_no_na = [x[0] for x in os.walk(path_matrices_no_na_eigen_no_alg_ind)]
    dirnames_w_alg_ind_no_na = [x[0] for x in os.walk(path_matrices_no_na_eigen_w_alg_ind)]
    dirnames_no_alg_ind_all_data = [x[0] for x in os.walk(path_all_data_eigen_no_alg_ind)]
    dirnames_w_alg_ind_all_data = [x[0] for x in os.walk(path_all_data_eigen_w_alg_ind)]

    build_projection_vectors(dirnames=dirnames_no_alg_ind_no_na, dest_path=dest_path_matrices_no_na_proj)
    build_projection_vectors(dirnames=dirnames_no_alg_ind_all_data, dest_path=dest_path_all_data_proj)
    build_projection_vectors(dirnames=dirnames_w_alg_ind_no_na, dest_path=dest_path_matrices_no_na_proj)
    build_projection_vectors(dirnames=dirnames_w_alg_ind_all_data, dest_path=dest_path_all_data_proj)

    print("\n")


# This method builds a [3, M] matrix where M is the number of types of measurements (turbidity, water
# temperature, etc.). Each column represents one point that can be plot on 3D coordinate axes. dirnames is an array of
# directories, where each directory contains eigenvectors and the matrix which will have the eigenvectors projected onto
# it. dest_path is the final destination folder for the [3, M].
def build_projection_vectors(dirnames, dest_path):
    if len(dirnames) > 1:
        dirnames = dirnames[1:]

    for directory in dirnames:
        # get the eigenvectors filenames in the directory
        eigv1_name = glob.glob(os.path.join(directory, "*eigv1.csv"))
        eigv2_name = glob.glob(os.path.join(directory, "*eigv2.csv"))
        eigv3_name = glob.glob(os.path.join(directory, "*eigv3.csv"))

        # get the matrix filename in the directory
        mat_name = glob.glob(os.path.join(directory, "*matrix.csv"))

        # open eigenvectors and matrix files in directory
        eigv1 = np.genfromtxt(open(eigv1_name[0], "rb"), delimiter=",", dtype=str)
        eigv2 = np.genfromtxt(open(eigv2_name[0], "rb"), delimiter=",", dtype=str)
        eigv3 = np.genfromtxt(open(eigv3_name[0], "rb"), delimiter=",", dtype=str)
        mat = np.genfromtxt(open(mat_name[0], "rb"), delimiter=",", dtype=str)

        # get mat_name without directory
        for i in range(0, len(mat_name[0])):
            if mat_name[0][i] == "/":
                max_idx = i + 1  # mat_name will be searched through for the last "/". knowing the index of the last "/"
                # will be useful in determining the mat_name

        mat_name = mat_name[0][max_idx:]
        print("Processing file " + mat_name + " ...")

        # final_directory is the final location of mat and its eigenvectors
        proj_name = mat_name[:-4] + "_proj"
        if "eigen-no-alg-ind" in directory:
            proj_name = proj_name + "_no-alg-ind"
        else:
            proj_name = proj_name + "_w-alg-ind"

        final_directory = dest_path + mat_name[:-4] + "/"

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
        proj_mat_3d = np.zeros(shape=(3, mat.shape[Constants.COLUMNS]), dtype=float)
        proj_mat_2d = np.zeros(shape=(2, mat.shape[Constants.COLUMNS]), dtype=float)
        proj_mat_1d = np.zeros(shape=(1, mat.shape[Constants.COLUMNS]), dtype=float)

        for i in range(0, mat.shape[Constants.COLUMNS]):
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

        matrix_to_file(mat, filename=mat_name, destination_folder=final_directory)
        matrix_to_file(proj_mat_3d, filename=proj_name + "_3d.csv", destination_folder=final_directory)
        matrix_to_file(proj_mat_2d, filename=proj_name + "_2d.csv", destination_folder=final_directory)
        matrix_to_file(proj_mat_1d, filename=proj_name + "_1d.csv", destination_folder=final_directory)


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


if __name__ == "__main__": main()