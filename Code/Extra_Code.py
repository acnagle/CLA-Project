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