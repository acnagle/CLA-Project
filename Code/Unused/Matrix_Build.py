import csv
import datetime
import numpy as np


def main():
    np.set_printoptions(threshold=np.inf)   # prints a full matrix rather than an abbreviated matrix

    print('\n\t##### EXECUTING MATRIX_BUILD.PY #####')

    # File locations of each csv
    file_paths = [
        "/Users/Alliot/documents/CLA-project/Data/CM2015_edit.csv",
        "/Users/Alliot/documents/CLA-project/Data/CM2016_edit.csv",
        "/Users/Alliot/documents/CLA-project/Data/CM2017_edit.csv"
    ]

    destination_folder = '/Users/Alliot/Documents/CLA-Project/Data/matrices/'

    # Define a matrix for each year of data
    mat15_year = np.empty([15, 1264], dtype=(str, 15))
    mat16_year = np.empty([15, 1638], dtype=(str, 15))
    mat17_year = np.empty([15, 1914], dtype=(str, 15))

    yearly_matrices = [mat15_year, mat16_year, mat17_year]
    mat_index = 0   # used to index each matrix in yearly_matrices

    # Create matrices for each year
    for path in file_paths:
        file = open(path, newline='')
        data_reader = csv.reader(file)
        file_year = path[43:-9]
        matrix_year(yearly_matrices[mat_index], data_reader, file_year)
        mat_index = mat_index + 1

    # Create arrays to hold the indices for each month
    indices15_months = month_index(mat15_year)
    indices16_months = month_index(mat16_year)
    indices17_months = month_index(mat17_year)

    write_to_csv(mat15_year, indices15_months, '2015', destination_folder)
    write_to_csv(mat16_year, indices16_months, '2016', destination_folder)
    write_to_csv(mat17_year, indices17_months, '2017', destination_folder)

    # Determine length of each matrix if it were to contain data from all years for each month
    length_month = np.zeros(8, dtype=int)   # only 7 elements because we only have data from 7 months

    for i in range(0, len(length_month)):
        length_month[i] = (indices15_months[i+1] - indices15_months[i]) + \
                          (indices16_months[i+1] - indices16_months[i]) + (indices17_months[i+1] - indices17_months[i])


# Create matrices for each year. mat is the matrix for a particular year. data_reader is used to read each line
# of the csv file. file_year is the year the matrix passed in
def matrix_year(mat, data_reader, file_year):
    data_reader.__next__()  # Remove the first line

    for row in data_reader:
        if row[0] != '':
            col = int(row[0])  # Get the column
            new_col = np.empty([15, 1], dtype=(str, 15))  # Holds the column currently being processed

            new_col[0, 0] = str(col) + "-" + file_year  # ID
            new_col[1, 0] = row[1]                      # Locations (Specific locations for each lake)
            new_col[2, 0] = row[4]                      # Locations (lake names)
            new_col[3, 0] = date = row[14]              # Dates
            new_col[4, 0] = time = row[15]              # Times (24-hour)
            new_col[5, 0] = row[6]                      # algalBlooms
            new_col[6, 0] = row[7]                      # algalBloomSheen
            new_col[7, 0] = row[8]                      # batherLoad
            new_col[8, 0] = row[10]                     # plantDebris
            new_col[9, 0] = row[17]                     # waterAppearance
            new_col[10, 0] = row[18]                    # waterfowlPresence
            new_col[11, 0] = row[20]                    # waveIntensity
            new_col[12, 0] = row[19]                    # waterTemp
            new_col[13, 0] = row[16]                    # turbidity
            new_col[14, 0] = row[24]                    # poor water quality flag

            # adjust column entries for poor water quality flag
            if row[16] == 'NA' or float(row[16]) <= 50:
                new_col[14, 0] = 1

            date = datetime.datetime.strptime(date, '%m/%d/%y')
            time = datetime.datetime.strptime(time, '%H:%M:%S')

            # sort rows based on dates and time
            for i in range(0, mat.shape[1]):  # .shape[1] gets the length of the first row of the matrix
                if mat[3, i] == '':
                    insert_column(i, col, mat, new_col[:, 0])
                    break

                elif date == datetime.datetime.strptime(mat[3, i], '%m/%d/%y'):
                    if time < datetime.datetime.strptime(mat[4, i], '%H:%M:%S'):
                        insert_column(i, col, mat, new_col[:, 0])
                        break

                elif date < datetime.datetime.strptime(mat[3, i], '%m/%d/%y'):
                    insert_column(i, col, mat, new_col[:, 0])
                    break


# This writes the matrix_year and data for each of its months to their own csv files. matrix_year is the data in
# the matrix representing a particular year. month_list is the list of indices for which each month begins
# in the matrix_year matrix. file_year is a string which contains the year for the data in matrix_year.
# destination_folder is the path to the destination folder where the csv file will be stored
def write_to_csv(mat_year, month_list, file_year, destination_folder):
    file = open(destination_folder + file_year + '_year_matrix.csv', 'w')

    # Create matrix for the year
    for i in range(0, mat_year.shape[0]):
        for j in range(0, mat_year.shape[1]):
            if j < mat_year.shape[1]-1:
                file.write(mat_year[i, j] + ',')
            else:
                file.write(mat_year[i, j] + '\n')

    # Create matrix for each month
    for i in range(0, month_list.shape[0]-1):
        if month_list[i] == mat_year.shape[1]-1:
            break

        else:
            file = open(destination_folder + file_year + '_month_0' + str(i + 3) + '_matrix.csv', 'w')
            for j in range(0, mat_year.shape[0]):
                for k in range(int(month_list[i]), int(month_list[i+1])):
                    if k < month_list[i+1]-1:
                        file.write(mat_year[j, k] + ',')
                    else:
                        file.write(mat_year[j, k] + '\n')

    file.close()


# This method finds the indices for the beginning of each month in the matrix for a particular year. month_list
# is created to hold these indices, and is returned
# mat_year is the matrix for an entire year.
def month_index(mat_year):
    month_list = np.zeros([9, 1])   # Holds the indices for the beginning of each month in the year matrix
                                    # There are only 9 elements because data is only gathered from March through Sept.
    col = mat_year[:, 0]    # Get the first data entry for the year
    i = 0   # index for iterating through mat_year
    k = 0   # k is for indexing month_list

    # First, determine the number of data entries for each month and create empty matrices based on that number
    try:
        curr_month = int(col[3][0:2])  # curr_month is the month of the data being processed
    except ValueError:
        curr_month = int(col[3][0:1])

    j = curr_month  # index for iterating through all 12 months of the year

    while col[0] != '':
        while curr_month == j:
            i = i + 1
            col = mat_year[:, i]
            try:
                curr_month = int(col[3][0:2])  # curr_month is the month of the data being processed
            except ValueError:
                curr_month = col[3][0:2]
                if curr_month != '':
                    curr_month = int(col[3][0:1])

        k = k + 1
        if k == 0:
            month_list[k] = 0
        else:
            month_list[k] = i

        j = j + 1

    return month_list


# This method shifts a column in the matrix to make room for the insertion of another column. index is the index
# of the column to be shifted to the right by one. TotalCol is the total number of non-empty columns in mat.
# mat is the matrix whose columns are being shifted. new_col is the column about to be inserted
def insert_column(index, total_col, mat, new_col):
    i = total_col

    if total_col < mat.shape[1]:
        while i > index:
            mat[:, i] = mat[:, i - 1]
            i = i - 1

        # insert the last column (newest data read in from file) into the open column
        mat[:, index] = new_col


if __name__ == "__main__": main()