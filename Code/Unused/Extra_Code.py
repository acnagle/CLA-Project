### The purpose of this file is to store and document extra python code that may be useful for later use ###

# # Check for NA in a line of csv file. If any of the rows contain 'NA', the column with that NA entry will be printed
#             if (row[1] == 'NA' or row[14] == 'NA' or row[15] == 'NA' or row[6] == 'NA' or row[7] == 'NA'
#                 or row[8] == 'NA' or row[10] == 'NA' or row[17] == 'NA' or row[18] == 'NA' or row[20] == 'NA'
#                     or row[19] == 'NA' or row[16] == 'NA'):
#
#                 print(col)

# # Reformat dates. The could be useful if dates need to be reformatted for ease of use in processing the data
            # if date[2] != '/':
            #     date = '0' + date
            #
            # if date[5] != '/':
            #     date = date[0:3] + '0' + date[3:]
            #
            # date = date[0:2] + date[3:5] + date[6:8]
            # # NOTE: date has been transformed from xx/xx/xx to xxxxxx. i.e. 8/6/15 is now 080615
            # new_col[2, 0] = date

# Find the largest three eigenvalues and their respective indices by comparing the absolute value of all
# eigenvalues. Variables first, second, and third correspond to first largest, second largest, and third largest
# eigenvalues. first_idx, second_idx, and third_idx are the indices of their respective eigenvalues within w.
# first = second = third = 0
# first_idx = second_idx = third_idx = 0
# for i in range(0, len(w)):
#     if w[i] != 0:
#         if abs(w[i]) > abs(first):
#             third = second
#             second = first
#             first = w[i]
#             third_idx = second_idx
#             second_idx = first_idx
#             first_idx = i
#         elif abs(w[i]) > abs(second):
#             third = second
#             second = w[i]
#             third_idx = second_idx
#             second_idx = i
#         elif abs(w[i]) > abs(third):
#             third = w[i]
#             third_idx = i

# u1 = v[first_idx]
# u2 = v[second_idx]
# u3 = v[third_idx]

# zero pad mat so that it becomes a square matrix. There are two cases to check for in terms of the size of
# mat: n > m, m > n. if m == n, then mat is already square and so we don't zero pad
# if mat.shape[1] > mat.shape[0]:
#     mat = np.pad(mat, ((0, mat.shape[1] - mat.shape[0]), (0, 0)), 'constant', constant_values=0.0)
# elif mat.shape[1] < mat.shape[0]:
#     mat = np.pad(mat, ((0, 0), (0, mat.shape[0] - mat.shape[1])), 'constant', constant_values=0.0)

# Original code for month_index in Matrix_Build_no_na.py
# month_list = np.zeros([9, 1])   # Holds the indices for the beginning of each month in the year matrix
# There are only 9 elements because data is only gathered from March through Sept.
# i = 0   # index for iterating through mat_year
# k = 0   # k is for indexing month_list
# while col[0] != '':
    #     while curr_month == j:
    #         i = i + 1
    #         col = mat_year[:, i]
    #         try:
    #             curr_month = int(col[3][0:2])  # curr_month is the month of the data being processed
    #         except ValueError:
    #             curr_month = col[3][0:2]
    #             if curr_month != '':
    #                 curr_month = int(col[3][0:1])
    #
    #     k = k + 1
    #     if k == 0:
    #         month_list[k] = 0
    #     else:
    #         month_list[k] = i
    #
    #     j = j + 1



## Used in compute eigenvectors.py. Used to put eigenvectors and eigenvalues in order

# Find the largest three eigenvalues and their respective indices.
    # Variables first, second, and third correspond to first largest, second largest, and third largest
    # eigenvalues. first_idx, second_idx, and third_idx are the indices of their respective eigenvalues within w.
    # first = second = third = 0
    # first = second = third = np.amin(w)
    # first_idx = second_idx = third_idx = 0
    # for i in range(0, len(w)):
    #     if w[i] > first:
    #         third = second
    #         second = first
    #         first = w[i]
    #         third_idx = second_idx
    #         second_idx = first_idx
    #         first_idx = i
    #     elif w[i] > second:
    #         third = second
    #         second = w[i]
    #         third_idx = second_idx
    #         second_idx = i
    #     elif w[i] > third:
    #         third = w[i]
    #         third_idx = i

    # eigv1 = np.real(v[first_idx])
    # eigv2 = np.real(v[second_idx])
    # eigv3 = np.real(v[third_idx])
    # eigvals = np.real(-np.sort(-w))

    # eigv1 = v[first_idx]
    # eigv2 = v[second_idx]
    # eigv3 = v[third_idx]
    # eigvals = -np.sort(-w)

    # Compute singular values
    # svdvals = np.zeros(len(eigvals), dtype=float)
    # for i in range(0, len(eigvals)):
    #     if eigvals[i] < 0:
    #         temp = math.sqrt(-eigvals[i])
    #         svdvals[i] = -temp
    #     else:
    #         svdvals[i] = math.sqrt(eigvals[i])



# # create matrices for each summer month for each year
# month_06_2015 = np.genfromtxt(open(summer_month_paths[0], "rb"), delimiter=",", dtype="str")
# month_07_2015 = np.genfromtxt(open(summer_month_paths[1], "rb"), delimiter=",", dtype="str")
# month_08_2015 = np.genfromtxt(open(summer_month_paths[2], "rb"), delimiter=",", dtype="str")
# month_06_2016 = np.genfromtxt(open(summer_month_paths[3], "rb"), delimiter=",", dtype="str")
# month_07_2016 = np.genfromtxt(open(summer_month_paths[4], "rb"), delimiter=",", dtype="str")
# month_08_2016 = np.genfromtxt(open(summer_month_paths[5], "rb"), delimiter=",", dtype="str")
# month_06_2017 = np.genfromtxt(open(summer_month_paths[6], "rb"), delimiter=",", dtype="str")
# month_07_2017 = np.genfromtxt(open(summer_month_paths[7], "rb"), delimiter=",", dtype="str")
# month_08_2017 = np.genfromtxt(open(summer_month_paths[8], "rb"), delimiter=",", dtype="str")
#
# # concatenate all summer month matrices into a single matrix for each year
# summer_2015 = np.hstack((month_06_2015, month_07_2015))
# summer_2015 = np.hstack((summer_2015, month_08_2015))
# summer_2016 = np.hstack((month_06_2016, month_07_2016))
# summer_2016 = np.hstack((summer_2016, month_08_2016))
# summer_2017 = np.hstack((month_06_2017, month_07_2017))
# summer_2017 = np.hstack((summer_2017, month_08_2017))

# Was Used in Gaussian_rand_Projection
# determine the rows of proj_mat_3d, proj_mat_2d, and proj_mat_1d, where proj_mat_ed, for example,
# is the matrix containing the "projection" of the data matrix, mat, on V = [eigv1 eigv2 eigv3].
# Each column in proj_mat corresponds to the projection of a particular row in
# mat on V. let V = [eigv1 eigv2 eigv3], and let A represent mat so that A_i is the ith column in mat.
# (U^T)A_i = [(eigv1^T)A_i; (eigv2^T)A_i; (eigv3^T)A_i] = proj_mat_3d[i]
# proj_mat_3d = np.zeros((3, mat.shape[1]), dtype=float)
# proj_mat_2d = np.zeros((2, mat.shape[1]), dtype=float)
# proj_mat_1d = np.zeros((1, mat.shape[1]), dtype=float)
#
# for i in range(0, mat.shape[1]):
#     proj_mat_3d[0, i] = eigv1.T.dot(mat[:, i])
#     proj_mat_3d[1, i] = eigv2.T.dot(mat[:, i])
#     proj_mat_3d[2, i] = eigv3.T.dot(mat[:, i])
#
#     proj_mat_2d[0, i] = eigv1.T.dot(mat[:, i])
#     proj_mat_2d[1, i] = eigv2.T.dot(mat[:, i])
#
#     proj_mat_1d[0, i] = eigv1.T.dot(mat[:, i])


### FROM GAUSSIAN_KERNEL_RBF
# import numpy as np
# from sklearn.model_selection import train_test_split
# from sklearn import svm
# import matplotlib.pyplot as plt
# from textwrap import wrap
# import errno
# import os
# import Constants
# import sys
#
#
# def main():
#     np.set_printoptions(threshold=np.inf)  # prints a full matrix rather than an abbreviated matrix
#
#     print("\n\t##### EXECUTING POLYNOMIAL_KERNEL.PY #####\n")
#
#     # source directories for normalized data matrices with algae indicator (summer months only!)
#     src_path_all_data_summer_norm_w_ind = "/Users/Alliot/Documents/CLA-Project/Data/all-data-no-na/eigenvectors/" \
#                                           "All_data_summer_matrix/All_Data_summer_matrix.csv"
#     src_path_mendota_norm_w_ind = "/Users/Alliot/Documents/CLA-Project/Data/all-data-no-na/eigenvectors/" \
#                                   "Mendota_All_Data_summer_matrix/Mendota_All_Data_summer_matrix.csv"
#     src_path_monona_norm_w_ind = "/Users/Alliot/Documents/CLA-Project/Data/all-data-no-na/eigenvectors/" \
#                                  "Monona_All_Data_summer_matrix/Monona_All_Data_summer_matrix.csv"
#     src_path_all_data_norm_w_ind = "/Users/Alliot/Documents/CLA-Project/Data/all-data-no-na/eigenvectors/" \
#                                    "All_Data_matrix/All_Data_matrix.csv"
#
#     # source directories for standard data set (No kernel trick, no algae indicator)
#     src_path_all_data_summer_no_ind = "/Users/Alliot/Documents/CLA-Project/Data/all-data-no-na/eigen-no-alg-ind/" \
#                                       "All_Data_summer_matrix/All_Data_summer_matrix.csv"
#     src_path_mendota_no_ind = "/Users/Alliot/Documents/CLA-Project/Data/all-data-no-na/eigen-no-alg-ind/" \
#                               "Mendota_All_Data_summer_matrix/Mendota_All_Data_summer_matrix.csv"
#     src_path_monona_no_ind = "/Users/Alliot/Documents/CLA-Project/Data/all-data-no-na/eigen-no-alg-ind/" \
#                              "Monona_All_Data_summer_matrix/Monona_All_Data_summer_matrix.csv"
#     src_path_all_data_no_ind = "/Users/Alliot/Documents/CLA-Project/Data/all-data-no-na/eigen-no-alg-ind/" \
#                                "All_Data_matrix/All_Data_matrix.csv"
#
#     dest_path = "/Users/Alliot/Documents/CLA-Project/Data/all-data-no-na/kernels/"
#
#     # if dest_path does not exist, create it
#     if not os.path.exists(dest_path):
#         try:
#             os.makedirs(dest_path)
#         except OSError as e:
#             if e.errno != errno.EEXIST:
#                 raise
#
#     # read in files from source directories
#     mat_all_data_summer_norm_w_ind = np.genfromtxt(
#         open(src_path_all_data_summer_norm_w_ind, "rb"),
#         delimiter=",",
#         dtype=float
#     )
#     mat_all_data_summer_no_ind = np.genfromtxt(
#         open(src_path_all_data_summer_no_ind, "rb"),
#         delimiter=",",
#         dtype=float
#     )
#
#     mat_mendota_norm_w_ind = np.genfromtxt(open(src_path_mendota_norm_w_ind, "rb"), delimiter=",", dtype=float)
#     mat_mendota_no_ind = np.genfromtxt(open(src_path_mendota_no_ind, "rb"), delimiter=",", dtype=float)
#
#     mat_monona_norm_w_ind = np.genfromtxt(open(src_path_monona_norm_w_ind, "rb"), delimiter=",", dtype=float)
#     mat_monona_no_ind = np.genfromtxt(open(src_path_monona_no_ind, "rb"), delimiter=",", dtype=float)
#
#     mat_all_data_norm_w_ind = np.genfromtxt(open(src_path_all_data_norm_w_ind, "rb"), delimiter=",", dtype=float)
#     mat_all_data_no_ind = np.genfromtxt(open(src_path_all_data_no_ind, "rb"), delimiter=",", dtype=float)
#
#     # get the labels for each norm matrix. THE ONLY PURPOSE OF THE NORM MATRICES IS TO RETRIEVE THE LABELS!
#     mat_all_data_summer_labels = mat_all_data_summer_norm_w_ind[Constants.ALGAL_BLOOM_SHEEN_NO_LOC, :]
#     mat_mendota_labels = mat_mendota_norm_w_ind[Constants.ALGAL_BLOOM_SHEEN_NO_LOC, :]
#     mat_monona_labels = mat_monona_norm_w_ind[Constants.ALGAL_BLOOM_SHEEN_NO_LOC, :]
#     mat_all_data_labels = mat_all_data_norm_w_ind[Constants.ALGAL_BLOOM_SHEEN_NO_LOC, :]
#
#     # modify the label vectors to be binary. Label 0 indicates no algal bloom, label 1 indicates an algal bloom
#     for i in range(0, len(mat_all_data_summer_labels)):
#         if mat_all_data_summer_labels[i] == 0.5:
#             mat_all_data_summer_labels[i] = 1
#         elif mat_all_data_summer_labels[i] == 0:
#             mat_all_data_summer_labels[i] = -1
#
#     for i in range(0, len(mat_mendota_labels)):
#         if mat_mendota_labels[i] == 0.5:
#             mat_mendota_labels[i] = 1
#         elif mat_mendota_labels[i] == 0:
#             mat_mendota_labels[i] = -1
#
#     for i in range(0, len(mat_monona_labels)):
#         if mat_monona_labels[i] == 0.5:
#             mat_monona_labels[i] = 1
#         elif mat_monona_labels[i] == 0:
#             mat_monona_labels[i] = -1
#
#     for i in range(0, len(mat_all_data_labels)):
#         if mat_all_data_labels[i] == 0.5:
#             mat_all_data_labels[i] = 1
#         if mat_all_data_labels[i] == 0:
#             mat_all_data_labels[i] = -1
#
#     # transpose test and training sets so that they are in the correct format (n_samples, m_features)
#     mat_all_data_summer_no_ind = mat_all_data_summer_no_ind.T
#     mat_mendota_no_ind = mat_mendota_no_ind.T
#     mat_monona_no_ind = mat_monona_no_ind.T
#     mat_all_data_no_ind = mat_all_data_no_ind.T
#
#     # vector of data matrices to try the kernel trick with
#     data_vec = np.array([mat_all_data_summer_no_ind, mat_mendota_no_ind, mat_monona_no_ind, mat_all_data_no_ind])
#
#     # description of each data matrix in data_vec
#     data_desc = ["all lakes, summer months only (June through August)", "Mendota, summer months only",
#                  "Monona, summer months only", "all lakes, all months"]
#
#     # array of label vectors
#     labels_vec = np.array([mat_all_data_summer_labels, mat_mendota_labels, mat_monona_labels, mat_all_data_labels])
#
#     num_iterations = 1
#
#     c = np.linspace(start=10000, stop=50000, num=17)
#
#     for i in range(0, len(data_vec)):
#         # create vectors for plotting error rates for each kernel
#         y_ber = np.zeros(c.shape[0])
#         y_no_alg = np.zeros(c.shape[0])
#         y_alg = np.zeros(c.shape[0])
#
#         for k in range(0, c.shape[0]):
#             svc = svm.SVC(
#                 C=c[k],
#                 kernel="rbf",
#                 gamma="auto",
#                 coef0=0,
#                 probability=False,
#                 shrinking=True,
#                 tol=0.0001,
#                 verbose=False,
#                 max_iter=-1,
#                 decision_function_shape="ovo"
#             )
#
#             cumulative_ber = 0
#             cumulative_no_alg_error = 0
#             cumulative_alg_error = 0
#
#             for j in range(0, num_iterations):
#                 x_train, x_test, y_train, y_test = train_test_split(
#                     data_vec[i],
#                     labels_vec[i],
#                     test_size=0.33,
#                     # random_state=543,
#                     shuffle=True
#                 )
#
#                 svc.fit(x_train, y_train)
#                 pred_labels_test = svc.predict(x_test)
#
#                 ber, no_alg_error, alg_error, _ = calculate_error(pred_labels_test, y_test)
#
#                 cumulative_ber += ber
#                 cumulative_no_alg_error += no_alg_error
#                 cumulative_alg_error += alg_error
#
#             total_ber = cumulative_ber / num_iterations
#             total_no_alg_error = cumulative_no_alg_error / num_iterations
#             total_alg_error = cumulative_alg_error / num_iterations
#
#             y_ber[k] = total_ber
#             y_no_alg[k] = total_no_alg_error
#             y_alg[k] = total_alg_error
#
#             print_results(
#                 title="Results for " + data_desc[i] + " (Kernel type: Gaussian RBF, C = " + str(c[k]) + ")",
#                 no_alg_error=total_no_alg_error,
#                 alg_error=total_alg_error
#             )
#
#         plt.figure()
#         plt.plot(c, y_ber, "b", c, y_no_alg, "g", c, y_alg, "r")
#         plt.ylabel("Error Rate")
#         plt.xlabel("C")
#         plt.legend(("BER", "No Algae", "Algae"))
#         plt.title("\n".join(wrap("Error Rates vs. C for " + data_desc[i] + " (Kernel type: Gaussian RBF)", 60)))
#         plt.savefig(os.path.join(dest_path, "Error Rates vs. C for " + data_desc[i] + " (Gaussian RBF Kernel).png"))
#
#
# # This method calculates the Balanced Error Rate (BER), and the error rates for no algae and algae prediction. This
# # method accepts an array of predicted labels, pred_labels, and an array of target labels, target_labels. This method
# # returns ber (the balanced error rate), no_alg_error (error rate for no algae prediction), and alg_error (error
# # rate for algae prediction). The confusion matrix, mat_conf, is returned as well (see first comment in method for a
# # description of a confusion matrix).
# def calculate_error(pred_labels, target_labels):
#     # Construct a confusion matrix, mat_conf. A confusion matrix consists of the true labels for the data points
#     # along its rows, and the predicted labels from k-nearest neighbors along its columns. The confusion matrix will
#     # be necessary to calculate BER and other relevant errors for evaluation of the kernel trick with linear
#     # classification. mat_conf is a 2x2 matrix because we only have two labels: no algae and algae. Each entry in
#     # mat_conf is the sum of occurrences of each predicted label for each true label.
#     mat_conf = np.zeros(shape=(2, 2), dtype=int)
#
#     if len(pred_labels) != len(target_labels):
#         print("Predicted and target label arrays are not the same length!")
#         sys.exit()
#
#     # no_alg = 0   # number of 0s (no algae) in target_labels
#     # no_alg_error = 0 # number of incorrect predictions on no algae (expected 0 but pred_labels[i] gave 1)
#     # alg = 0   # number of 1s (algae) in target_labels
#     # alg_error = 0 # number of incorrect predictions on algae (expected 1 but pred_labels[i] gave 0)
#     #
#     # for i in range(0, len(pred_labels)):
#     #     if target_labels[i] == 0:
#     #         no_alg += 1
#     #         if pred_labels[i] == 1:
#     #             no_alg_error += 1
#     #     elif target_labels[i] == 1:
#     #         alg += 1
#     #         if pred_labels[i] == 0:
#     #             alg_error += 1
#     #     else:
#     #         print("Unexpected target label: ", target_labels[i])
#     #         sys.exit()
#     #
#     # no_alg_error = no_alg_error / no_alg
#     # alg_error = alg_error / alg
#     #
#     # return no_alg_error, alg_error
#
#     # This for loop will populate mat_conf with the true labels and the predicted labels simultaneously.
#     for i in range(0, len(pred_labels)):
#         if (pred_labels[i] == -1) and (target_labels[i] == -1):
#             mat_conf[0, 0] += 1
#         elif (pred_labels[i] == 1) and (target_labels[i] == -1):
#             mat_conf[0, 1] += 1
#         elif (pred_labels[i] == -1) and (target_labels[i] == 1):
#             mat_conf[1, 0] += 1
#         elif (pred_labels[i] == 1) and (target_labels[i] == 1):
#             mat_conf[1, 1] += 1
#
#     # Calculate relevant errors and accuracies
#     # Given a confusion matrix as follows:
#     # [ a b ]
#     # [ c d ]
#     # We can define the following equations:
#     # Balanced Error Rate (BER) = (b / (a + b) + c / (c + d)) / 2
#     # error per label = each of the terms in the numerator of BER. ex: b / (a + b)
#
#     ber = (mat_conf[0, 1] / (mat_conf[0, 0] + mat_conf[0, 1]) + mat_conf[1, 0] / (mat_conf[1, 1] + mat_conf[1, 0])) / 2
#
#     no_alg_error = mat_conf[0, 1] / (mat_conf[0, 0] + mat_conf[0, 1])
#     alg_error = mat_conf[1, 0] / (mat_conf[1, 1] + mat_conf[1, 0])
#
#     return ber, no_alg_error, alg_error, mat_conf
#
#
# # This method prints the results of the linear classification
# def print_results(title, no_alg_error, alg_error):
#     print(title)
#     print("No Algae Prediction Error:", no_alg_error)
#     print("Algae Prediction Error:", alg_error)
#     print("---------------------------------------------------------------------------\n")
#
#
# if __name__ == "__main__": main()
