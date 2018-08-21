import numpy as np
import os
import errno
import Constants


def main():
    np.set_printoptions(threshold=np.inf)  # prints a full matrix rather than an abbreviated matrix

    print("\n\t##### EXECUTING PCA.PY #####\n")

    # get source directories for normalized data matrices (summer months only!)
    # Original Data
    src_path_all_data = "/Users/Alliot/Documents/CLA-Project/Data/all-data-no-na/original/All_Data_summer_matrix.csv"
    src_path_mendota = "/Users/Alliot/Documents/CLA-Project/Data/all-data-no-na/original/" \
                       "Mendota_All_Data_summer_matrix.csv"
    src_path_monona = "/Users/Alliot/Documents/CLA-Project/Data/all-data-no-na/original/" \
                      "Monona_All_Data_summer_matrix.csv"

    dest_path = "/Users/Alliot/Documents/CLA-Project/Data/all-data-no-na/PCA/"

    # if dest_path does not exist, create it
    if not os.path.exists(dest_path):
        try:
            os.makedirs(dest_path)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

    # read in files from source directories
    mat_all_data = np.genfromtxt(open(src_path_all_data, "rb"), delimiter=",", dtype=float)
    mat_mendota = np.genfromtxt(open(src_path_mendota, "rb"), delimiter=",", dtype=float)
    mat_monona = np.genfromtxt(open(src_path_monona, "rb"), delimiter=",", dtype=float)

    # Remove rows for location, date, and algae-bloom indicator (BloomSheen)
    mat_all_data = np.delete(mat_all_data, obj=[0, 1, 3], axis=Constants.ROWS)
    mat_mendota = np.delete(mat_mendota, obj=[0, 1, 3], axis=Constants.ROWS)
    mat_monona = np.delete(mat_monona, obj=[0, 1, 3], axis=Constants.ROWS)

    print("Processing file", src_path_all_data[65:], " ...")
    mat_all_data_pca = pca(mat_all_data)
    matrix_to_file(mat_all_data_pca, src_path_all_data[65:-4] + "_pca.csv", dest_path)

    print("Processing file", src_path_mendota[65:], " ...")
    mat_mendota_pca = pca(mat_mendota)
    matrix_to_file(mat_mendota_pca, src_path_mendota[65:-4] + "_pca.csv", dest_path)

    print("Processing file", src_path_monona[65:], " ...")
    mat_monona_pca = pca(mat_monona)
    matrix_to_file(mat_monona_pca, src_path_monona[65:-4] + "_pca.csv", dest_path)

    print("\n")


# This method computes the steps required for the PCA. mat is the matrix for which PCA will be performed.
# It returns pca_data, which is a [3, M] matrix, where M is the number of data points in mat. This matrix can be plotted
# in 3D for data visualization
def pca(mat):
    # 1. Subtract the mean along each dimension (row) and then normalize the data
    # mat_adj = mat   # mat_adj is adjusted so that the mean is subtracted out of mat
    # mean = np.mean(mat, axis=Constants.COLUMNS)     # get an array of means for each dimension
    # for i in range(0, mat.shape[Constants.ROWS]):
    #     for j in range(0, mat.shape[Constants.COLUMNS]):
    #         mat_adj[i, j] = mat_adj[i, j] - mean[i]

    # 2. Calculate the covariance matrix
    # mat_cov = np.cov(mat_adj)
    #
    # # 3. Calculate eigenvectors and eigenvalues of the covariance matrix
    # w, v = np.linalg.eig(mat_cov)
    #
    # # Sort the eigenvalues from greatest to smallest and adjust eigenvectors accordingly
    # idx = w.argsort()[::-1]
    # w = w[idx]
    # v = v[:, idx]

    # Or, just calculate the SVD
    cov_mat = np.cov(mat)  # calculate covariance matrix
    u, s, v = np.linalg.svd(cov_mat, full_matrices=True)

    # 4. Form a feature vector. Keep only the top three eigenvectors (singluar vectors) so PCA can be visualized in 3D.
    # The feature vector is a vector of the eigenvectors (singluar vectors)
    feat_vec = np.array([u[:, 0], u[:, 1], u[:, 2]])

    # 5. Put data in its final form.
    pca_data = np.dot(feat_vec, mat)

    return pca_data


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
