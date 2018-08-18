import csv
import datetime
import numpy as np
import os
import errno
import Constants


def main():
    np.set_printoptions(threshold=np.inf)   # prints a full matrix rather than an abbreviated matrix

    print("\n\t##### EXECUTING MATRIX_BUILD_NO_NA.PY #####")

    # File locations of each csv
    file_paths = [
        "/Users/Alliot/documents/CLA-project/Data/CM2015_edit.csv",
        "/Users/Alliot/documents/CLA-project/Data/CM2016_edit.csv",
        "/Users/Alliot/documents/CLA-project/Data/CM2017_edit.csv"
    ]

    destination_folder = "/Users/Alliot/Documents/CLA-Project/Data/matrices-no-na/original/"

    # Define a matrix for each year of data
    mat15_year = np.genfromtxt(open(file_paths[0], "rb"), delimiter=",", dtype=(str, Constants.STR_LENGTH),
                               usecols=(1, 14, 15, 6, 7, 8, 10, 17, 18, 20, 19, 16), filling_values="NA")
    mat16_year = np.genfromtxt(open(file_paths[1], "rb"), delimiter=",", dtype=(str, Constants.STR_LENGTH),
                               usecols=(1, 14, 15, 6, 7, 8, 10, 17, 18, 20, 19, 16), filling_values="NA")
    mat17_year = np.genfromtxt(open(file_paths[2], "rb"), delimiter=",", dtype=(str, Constants.STR_LENGTH),
                               usecols=(1, 14, 15, 6, 7, 8, 10, 17, 18, 20, 19, 16), filling_values="NA")
    # mat15_year = np.empty(shape=[num_rows, 1264], dtype=(str, 15))  # 1243
    # mat16_year = np.empty(shape=[num_rows, 1638], dtype=(str, 15))  # 1603
    # mat17_year = np.empty(shape=[num_rows, 1914], dtype=(str, 15))  # 1853

    yearly_matrices = [mat15_year, mat16_year, mat17_year]

    # mat_index = 0   # used to index each matrix in yearly_matrices
    # Create matrices for each year
    # for path in file_paths:
    #     print("Processing file " + path[41:] + " ...")
    #     file = open(path, newline="")
    #     file_year = path[43:-9]
    #     matrix_year(mat=yearly_matrices[mat_index])
    #     mat_index = mat_index + 1

    # Create matrices for each year
    print("Processing file " + file_paths[0][41:] + " ...")
    file15_year = file_paths[0][43:-9]
    mat15_year = matrix_year(mat=yearly_matrices[0])
    mat15_year = remove_empty_entries(mat15_year)     # Remove empty elements
    mat15_summer = build_summer_months(mat15_year)    # Create matrix for only summer months

    print("Processing file " + file_paths[1][41:] + " ...")
    file16_year = file_paths[1][43:-9]
    mat16_year = matrix_year(mat=yearly_matrices[1])
    mat16_year = remove_empty_entries(mat16_year)     # Remove empty elements
    mat16_summer = build_summer_months(mat16_year)    # Create matrix for only summer months

    print("Processing file " + file_paths[2][41:] + " ...")
    file17_year = file_paths[2][43:-9]
    mat17_year = matrix_year(mat=yearly_matrices[2])
    mat17_year = remove_empty_entries(mat17_year)     # Remove empty elements
    mat17_summer = build_summer_months(mat17_year)    # Create matrix for only summer months

    mat_all = build_all_year_matrix(mat15=mat15_year, mat16=mat16_year, mat17=mat17_year)
    mat_all_summer = build_summer_months(mat_all)

    # if destination_folder does not exist, create it
    if not os.path.exists(destination_folder):
        try:
            os.makedirs(destination_folder)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

    write_to_csv(mat15_year, month_list=None, filename=file15_year + "_year_matrix.csv",
                 destination_folder=destination_folder)
    write_to_csv(mat15_summer, month_list=None, filename=file15_year + "_summer_matrix.csv",
                 destination_folder=destination_folder)
    write_to_csv(mat16_year, month_list=None, filename=file16_year + "_year_matrix.csv",
                 destination_folder=destination_folder)
    write_to_csv(mat16_summer, month_list=None, filename=file16_year + "_summer_matrix.csv",
                 destination_folder=destination_folder)
    write_to_csv(mat17_year, month_list=None, filename=file17_year + "_year_matrix.csv",
                 destination_folder=destination_folder)
    write_to_csv(mat17_summer, month_list=None, filename=file17_year + "_summer_matrix.csv",
                 destination_folder=destination_folder)
    write_to_csv(mat_all, month_list=None, filename="All_year_matrix.csv", destination_folder=destination_folder)
    write_to_csv(mat_all, month_list=None, filename="All_year_summer_matrix.csv",
                 destination_folder=destination_folder)


# Create matrices for each year. mat is the matrix for a particular year.
def matrix_year(mat):
    mat = np.transpose(mat)     # reconfigure shape of the yearly matrix
    mat = np.delete(mat, obj=0, axis=Constants.COLUMNS)

    new_mat = np.empty(shape=(Constants.NUM_ROWS_NO_NA, mat.shape[Constants.COLUMNS]), dtype=(str, Constants.STR_LENGTH))
    idx = 0
    for i in range(0, mat.shape[Constants.COLUMNS]):
        if "" not in mat[:, i] and "NA" not in mat[:, i]:
            new_mat[Constants.LOCATION, idx] = mat[0, i]                        # Locations
            new_mat[Constants.DATE_TIME, idx] = mat[1, i] + " " + mat[2, i]     # Dates
            new_mat[Constants.ALGAL_BLOOMS, idx] = mat[3, i]                    # algalBlooms
            new_mat[Constants.ALGAL_BLOOM_SHEEN, idx] = mat[4, i]               # algalBloomSheen
            new_mat[Constants.BATHER_LOAD, idx] = mat[5, i]                     # batherLoad
            new_mat[Constants.PLANT_DEBRIS, idx] = mat[6, i]                    # plantDebris
            new_mat[Constants.WATER_APPEARANCE, idx] = mat[7, i]                # waterAppearance
            new_mat[Constants.WATER_FOWL_PRESENCE, idx] = mat[8, i]             # waterfowlPresence
            new_mat[Constants.WAVE_INTENSITY, idx] = mat[9, i]                  # waveIntensity
            new_mat[Constants.WATER_TEMP, idx] = mat[10, i]                     # waterTemp
            new_mat[Constants.TURBIDITY, idx] = mat[11, i]                      # turbidity

            # adjust data entries so that qualitative measurements (except algalBloomSheen) have values [1, 3]
            for j in range(Constants.FIRST_ROW, Constants.NUM_ROWS_NO_NA-1):
                if j != Constants.ALGAL_BLOOM_SHEEN:          # ignore algalBloomSheen measurement
                    if float(new_mat[j, idx]) == 0:
                        new_mat[j, idx] = "1"

            idx = idx + 1

    return new_mat


# This writes the matrix_year and data for each of its months to their own csv files. matrix_year is the data in
# the matrix representing a particular year. month_list is the list of indices for which each month begins
# in the matrix_year matrix. file_year is a string which contains the year for the data in matrix_year.
# destination_folder is the path to the destination folder where the csv file will be stored. If "None" is passed in
# place of month_list, this method will simply just write the entire matrix to a file
def write_to_csv(mat_year, month_list, filename, destination_folder):
    file = open(destination_folder + filename, "w")

    # Create matrix for the year
    for i in range(0, mat_year.shape[Constants.ROWS]):
        for j in range(0, mat_year.shape[Constants.COLUMNS]):
            if j < mat_year.shape[Constants.COLUMNS]-1:
                file.write(mat_year[i, j] + ",")
            else:
                file.write(mat_year[i, j] + "\n")

    if month_list is not None:

        num_months = len(month_list) - 1

        # Create matrix for each month
        for i in range(0, mat_year.shape[Constants.ROWS]-1):
            if i == num_months or (i > 0 and month_list[i] == 0):
                break

            else:
                file = open(destination_folder + filename + "_month_0" + str(i + 3) + "_matrix.csv", "w")
                for j in range(0, mat_year.shape[Constants.ROWS]):
                    try:
                        end = int(month_list[i+1])
                    except IndexError:
                        break

                    for k in range(int(month_list[i]), end):
                        if mat_year[j, k] == "":
                            break
                        if k < month_list[i+1]-1:
                            file.write(mat_year[j, k] + ",")
                        else:
                            file.write(mat_year[j, k] + "\n")

    file.close()


# This method finds the indices for the beginning of each month in the matrix for a particular year. month_list
# is created to hold these indices, and is returned
# mat_year is the matrix for an entire year.
def month_index(mat_year):
    col = mat_year[:, 0]    # Get the first data entry for the year
    # First, determine the number of data entries for each month and create empty matrices based on that number
    try:
        first_month = curr_month = int(col[Constants.DATE_TIME][0:2])  # curr_month is the month of the data being processed
    except ValueError:
        first_month = curr_month = int(col[Constants.DATE_TIME][0:1])

    j = first_month  # index for iterating through all 12 months of the year
    i = 0            # index of iterating through mat_year

    num_months = 1  # count the number of months
    # get number of months for data for same year as mat_year
    while col[0] != "":
        print(curr_month)
        while curr_month == j:
            i = i + 1
            col = mat_year[:, i]
            try:
                curr_month = int(col[Constants.DATE_TIME][0:2])
            except ValueError:
                curr_month = int(col[Constants.DATE_TIME][0:1])

            if curr_month != j:
                num_months = num_months + 1

        j = j + 1

    month_list = np.zeros(shape=num_months)

    col = mat_year[:, 0]  # Get the first data entry for the year
    j = curr_month = first_month
    i = 0
    k = 0   # used to index month_list
    for l in range(0, mat_year.shape[Constants.COLUMNS]):
        while curr_month == j:
            i = i + 1
            col = mat_year[:, i]
            print(i)
            try:
                curr_month = int(col[Constants.DATE_TIME][0:2])
            except ValueError:
                curr_month = int(col[Constants.DATE_TIME][0:1])

        k = k + 1
        month_list[k] = i
        j = j + 1

    return month_list


# This method takes takes in a matrix representing a year's worth of data and turns it into a matrix with only the
# summer months (June, July, August). mat a year's worth of data points. summer_mat is the data points only for the
# summer of that year; summer_mat is returned
def build_summer_months(mat):
    summer_mat = np.transpose([np.empty((Constants.NUM_ROWS_NO_NA, ))])

    for j in range(0, mat.shape[Constants.COLUMNS]):
        try:
            curr_month = int(mat[Constants.DATE_TIME, j][0:2])
        except ValueError:
            curr_month = int(mat[Constants.DATE_TIME, j][0:1])

        if (curr_month is 6) or (curr_month is 7) or (curr_month is 8):
            summer_mat = np.hstack([summer_mat, np.transpose([mat[:, j]])])

    # SPECIAL NOTE: for some reason I can't explain, the above code in this method appends an extra column to the front
    # of mat_val with values that are extremely tiny and large (order of 10^-250 to 10^314 or so). This code deletes
    # that column
    summer_mat = np.delete(summer_mat, obj=0, axis=1)

    return summer_mat


# This method shifts a column in the matrix to make room for the insertion of another column. index is the index
# of the column to be shifted to the right by one. TotalCol is the total number of non-empty columns in mat.
# mat is the matrix whose columns are being shifted. new_col is the column about to be inserted
def insert_column(index, total_col, mat, new_col):
    i = total_col

    if total_col < mat.shape[Constants.COLUMNS]:
        while i > index:
            mat[:, i] = mat[:, i - 1]
            i = i - 1

        # insert the last column (newest data read in from file) into the open column
        mat[:, index] = new_col


# This method removes the empty entries (i.e. ",,,") located at the ends of the .csv files. new_mat is the matrix
# with empty entries removed.
def remove_empty_entries(mat):
    new_mat = mat

    for j in range(0, mat.shape[Constants.COLUMNS]):
        if "" in new_mat[:, j]:
            new_mat = new_mat[:, 0:j]
            break

    return new_mat


# This method takes the matrices from the three years of data and merges them into a single matrix. mat is the matrix
# containing all three years and is returned. mat15, mat16, and mat17 are the matrices from years 2015, 2016, and 2017
# respectively
def build_all_year_matrix(mat15, mat16, mat17):
    mat = np.concatenate((mat15, mat16, mat17), axis=Constants.COLUMNS)

    return mat


if __name__ == "__main__": main()
