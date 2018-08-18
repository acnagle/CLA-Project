import numpy as np
import glob
import os
import errno
import Constants


def main():
    np.set_printoptions(threshold=np.inf)  # prints a full matrix rather than an abbreviated matrix

    print("\n\t##### EXECUTING GUASSIAN_RAND_PROJECTION.PY #####")

    # source directories
    path_matrices_no_na_eigen_no_alg_ind = "/Users/Alliot/documents/cla-project/data/matrices-no-na/eigen-no-alg-ind/"
    path_all_data_eigen_no_alg_ind = "/Users/Alliot/Documents/CLA-Project/Data/all-data-no-na/eigen-no-alg-ind/"
    path_matrices_no_na_eigen_w_alg_ind = "/Users/Alliot/documents/cla-project/data/matrices-no-na/eigenvectors/"
    path_all_data_eigen_w_alg_ind = "/Users/Alliot/documents/cla-project/data/all-data-no-na/eigenvectors/"

    # destination directories for projection vectors
    dest_path_matrices_no_na_proj = "/Users/Alliot/documents/cla-project/data/matrices-no-na/gaussian-randn-projections/"
    dest_path_all_data_proj = "/Users/Alliot/documents/cla-project/data/all-data-no-na/gaussian-randn-projections/"

    # get all subdirectories within path_matrices_no_na_eigen_no_alg_ind
    dirnames_no_alg_ind_no_na = [x[0] for x in os.walk(path_matrices_no_na_eigen_no_alg_ind)]
    dirnames_w_alg_ind_no_na = [x[0] for x in os.walk(path_matrices_no_na_eigen_w_alg_ind)]
    dirnames_no_alg_ind_all_data = [x[0] for x in os.walk(path_all_data_eigen_no_alg_ind)]
    dirnames_w_alg_ind_all_data = [x[0] for x in os.walk(path_all_data_eigen_w_alg_ind)]

    build_randn_projection_vectors(dirnames=dirnames_no_alg_ind_no_na, dest_path=dest_path_matrices_no_na_proj)
    build_randn_projection_vectors(dirnames=dirnames_no_alg_ind_all_data, dest_path=dest_path_all_data_proj)
    build_randn_projection_vectors(dirnames=dirnames_w_alg_ind_no_na, dest_path=dest_path_matrices_no_na_proj)
    build_randn_projection_vectors(dirnames=dirnames_w_alg_ind_all_data, dest_path=dest_path_all_data_proj)


def build_randn_projection_vectors(dirnames, dest_path):
    if len(dirnames) > 1:
        dirnames = dirnames[1:]

    for directory in dirnames:
        # get the matrix filename in the directory
        mat_name = glob.glob(os.path.join(directory, "*matrix.csv"))

        # open matrix files in directory
        mat = np.genfromtxt(open(mat_name[0], "rb"), delimiter=",", dtype=str)

        # get mat_name without directory
        for i in range(0, len(mat_name[0])):
            if mat_name[0][i] == "/":
                max_idx = i + 1  # mat_name will be searched through for the last "/". knowing the index of the last "/"
                                 # will be useful in determining the mat_name

        mat_name = mat_name[0][max_idx:]
        print("Processing file " + mat_name + " ...")

        # final_directory is the final location of mat
        proj_name = mat_name[:-4] + "_proj_randn"
        if "eigen-no-alg-ind" in directory:
            proj_name = proj_name + "_no-alg-ind"
        else:
            proj_name = proj_name + "_w-alg-ind"

        final_directory = dest_path + mat_name[:-4] + "/"

        mat = mat.astype(dtype=float)     # convert mat from str entries to float entries

        randn_mat_3d = np.random.randn(3, mat.shape[Constants.ROWS])
        proj_mat_3d = randn_mat_3d.dot(mat)

        # if final_directory does not exist, create it
        if not os.path.exists(final_directory):
            try:
                os.makedirs(final_directory)
            except OSError as e:
                if e.errno != errno.EEXIST:
                    raise

        matrix_to_file(mat, filename=mat_name, destination_folder=final_directory)
        matrix_to_file(proj_mat_3d, filename=proj_name + "_3d.csv", destination_folder=final_directory)


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
