import numpy as np
import matplotlib.pyplot as plt
import os
import errno


num_rows_no_ind = 11   # number of measurements per data point for data with no indicator
num_rows_w_ind = 13   # number of measurements per data point for data with indicator
num_rows_3d_proj = 3    # number of rows in a 3D projection matrix


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
    mat_all_data = np.delete(mat_all_data, [0, 1, 3], 0)
    mat_mendota = np.delete(mat_mendota, [0, 1, 3], 0)
    mat_monona = np.delete(mat_monona, [0, 1, 3], 0)

    mat_all_data_pca = pca(mat_all_data)
    mat_mendota_pca = pca(mat_mendota)
    mat_monona_pca = pca(mat_monona)

    print("Processing file", src_path_all_data[65:], " ...")
    matrix_to_file(mat_all_data_pca, src_path_all_data[65:-4] + "_pca.csv", dest_path)
    print("Processing file", src_path_mendota[65:], " ...")
    matrix_to_file(mat_mendota_pca, src_path_mendota[65:-4] + "_pca.csv", dest_path)
    print("Processing file", src_path_monona[65:], " ...")
    matrix_to_file(mat_monona_pca, src_path_monona[65:-4] + "_pca.csv", dest_path)


# This method computes the steps required for the PCA. mat is the matrix for which PCA will be performed.
# It returns pca_data, which is a [3, M] matrix, where M is the number of data points in mat. This matrix can be plotted
# in 3D for data visualization
def pca(mat):
    # 1. Subtract the mean along each dimension (row) and then normalize the data
    mat_adj = mat   # mat_adj is adjusted so that the mean is subtracted out of mat
    mean = np.mean(mat, axis=1)     # get an array of means for each dimension
    for i in range(0, mat.shape[0]):
        for j in range(0, mat.shape[1]):
            mat_adj[i, j] = mat_adj[i, j] - mean[i]

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
    u, s, v = np.linalg.svd(mat, full_matrices=True)

    # 4. Form a feature vector. Keep only the top three eigenvectors (singluar vectors) so PCA can be visualized in 3D.
    # The feature vector is a vector of the eigenvectors (singluar vectors)
    feat_vec = np.array([u[:, 0], u[:, 1], u[:, 2]])

    # 5. Put data in its final form.
    pca_data = np.dot(feat_vec, mat_adj)

    return pca_data


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


# This method normalizes the data in a matrix by finding the largest value in each row and dividing each element in
# row by that value. This will cause each point in the data to be between the 0 and 1. mat is the matrix whose rows will
# be normalized. first is the first row that contains data that needs to be normalized. last is the last row
# (non-inclusive) that contains data that needs to be normalized
def normalize_data(mat, first, last):
    for i in range(first, last):
        # norm_arr_str contains the string representation of an entire row that is going to be normalized
        norm_arr_str = mat[i, :]

        # convert all elements in norm_arr to float
        # norm_arr is the float representation of norm_arr_str
        norm_arr = np.zeros(len(norm_arr_str), dtype="float")
        for j in range(0, len(norm_arr)):
            try:
                norm_arr[j] = float(norm_arr_str[j])
            except ValueError:
                print("The value at index " + str(j) + " could not be cast to a float.")

        # determine the largest value in norm_arr
        max_val = np.amax(norm_arr)

        if max_val != 0:
            # normalize all the elements by dividing each element by max_val
            for k in range(0, len(norm_arr)):
                norm_arr[k] = norm_arr[k] / max_val

                # store the normalize array back into its respective row in mat
                mat[i, k] = str(norm_arr[k])


if __name__ == "__main__": main()