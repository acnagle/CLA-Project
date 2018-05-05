import csv
import datetime
import numpy as np


def main():
    np.set_printoptions(threshold=np.inf)  # prints a full matrix rather than an abbreviated matrix

    print('\n\t##### EXECUTING MONTHLY_MATRIX_BUILD_ALL.PY #####')

    path = [
        '/Users/Alliot/Documents/CLA Project/Data/matrices/2015_month_03_matrix.csv',    # 5 columns
        '/Users/Alliot/Documents/CLA Project/Data/matrices/2015_month_04_matrix.csv',    # 22 columns
        '/Users/Alliot/Documents/CLA Project/Data/matrices/2015_month_05_matrix.csv',    # 170 columns
        '/Users/Alliot/Documents/CLA Project/Data/matrices/2015_month_06_matrix.csv',    # 360 columns
        '/Users/Alliot/Documents/CLA Project/Data/matrices/2015_month_07_matrix.csv',    # 365 columns
        '/Users/Alliot/Documents/CLA Project/Data/matrices/2015_month_08_matrix.csv',    # 277 columns
        '/Users/Alliot/Documents/CLA Project/Data/matrices/2015_month_09_matrix.csv',    # 64 columns
        '/Users/Alliot/Documents/CLA Project/Data/matrices/2016_month_03_matrix.csv',    # 14 columns
        '/Users/Alliot/Documents/CLA Project/Data/matrices/2016_month_04_matrix.csv',    # 18 columns
        '/Users/Alliot/Documents/CLA Project/Data/matrices/2016_month_05_matrix.csv',    # 180 columns
        '/Users/Alliot/Documents/CLA Project/Data/matrices/2016_month_06_matrix.csv',    # 503 columns
        '/Users/Alliot/Documents/CLA Project/Data/matrices/2016_month_07_matrix.csv',    # 462 columns
        '/Users/Alliot/Documents/CLA Project/Data/matrices/2016_month_08_matrix.csv',    # 378 columns
        '/Users/Alliot/Documents/CLA Project/Data/matrices/2016_month_09_matrix.csv',    # 82 columns
        '/Users/Alliot/Documents/CLA Project/Data/matrices/2017_month_03_matrix.csv',    # 4 columns
        '/Users/Alliot/Documents/CLA Project/Data/matrices/2017_month_04_matrix.csv',    # 76 columns
        '/Users/Alliot/Documents/CLA Project/Data/matrices/2017_month_05_matrix.csv',    # 234 columns
        '/Users/Alliot/Documents/CLA Project/Data/matrices/2017_month_06_matrix.csv',    # 459 columns
        '/Users/Alliot/Documents/CLA Project/Data/matrices/2017_month_07_matrix.csv',    # 501 columns
        '/Users/Alliot/Documents/CLA Project/Data/matrices/2017_month_08_matrix.csv',    # 426 columns
        '/Users/Alliot/Documents/CLA Project/Data/matrices/2017_month_09_matrix.csv',    # 166 columns
        '/Users/Alliot/Documents/CLA Project/Data/matrices/2017_month_010_matrix.csv'     # 47 columns
    ]

    destination_folder = '/Users/Alliot/Documents/CLA Project/Data/matrices/'

    # These matrices contain all the data from each month for all three years
    all_month_03_matrix = np.empty([15, 23], dtype=(str, 15))
    all_month_04_matrix = np.empty([15, 116], dtype=(str, 15))
    all_month_05_matrix = np.empty([15, 584], dtype=(str, 15))
    all_month_06_matrix = np.empty([15, 1322], dtype=(str, 15))
    all_month_07_matrix = np.empty([15, 1328], dtype=(str, 15))
    all_month_08_matrix = np.empty([15, 1081], dtype=(str, 15))
    all_month_09_matrix = np.empty([15, 312], dtype=(str, 15))
    all_month_10_matrix = np.empty([15, 47], dtype=(str, 15))

    # Create 03 matrix
    try:
        file_03_15 = open(path[0], newline='')
        data_reader_15 = csv.reader(file_03_15)
        file_03_16 = open(path[7], newline='')
        data_reader_16 = csv.reader(file_03_16)
        file_03_17 = open(path[14], newline='')
        data_reader_17 = csv.reader(file_03_17)
    except IOError:
        print('There was an error opening the file for 03 month!')
        return

    i = 0
    for row_15 in data_reader_15:
        all_month_03_matrix[i, 0:5] = row_15
        i = i + 1

    i = 0
    for row_16 in data_reader_16:
        all_month_03_matrix[i, 5:19] = row_16
        i = i + 1

    i = 0
    for row_17 in data_reader_17:
        all_month_03_matrix[i, 19:23] = row_17
        i = i + 1

    # Create 04 matrix
    try:
        file_04_15 = open(path[1], newline='')
        data_reader_15 = csv.reader(file_04_15)
        file_04_16 = open(path[8], newline='')
        data_reader_16 = csv.reader(file_04_16)
        file_04_17 = open(path[15], newline='')
        data_reader_17 = csv.reader(file_04_17)
    except IOError:
        print('There was an error opening the file for 04 month!')
        return

    i = 0
    for row_15 in data_reader_15:
        all_month_04_matrix[i, 0:22] = row_15
        i = i + 1

    i = 0
    for row_16 in data_reader_16:
        all_month_04_matrix[i, 22:40] = row_16
        i = i + 1

    i = 0
    for row_17 in data_reader_17:
        all_month_04_matrix[i, 40:116] = row_17
        i = i + 1

    # Create 05 matrix
    try:
        file_05_15 = open(path[2], newline='')
        data_reader_15 = csv.reader(file_05_15)
        file_05_16 = open(path[9], newline='')
        data_reader_16 = csv.reader(file_05_16)
        file_05_17 = open(path[16], newline='')
        data_reader_17 = csv.reader(file_05_17)
    except IOError:
        print('There was an error opening the file for 05 month!')
        return

    i = 0
    for row_15 in data_reader_15:
        all_month_05_matrix[i, 0:170] = row_15
        i = i + 1

    i = 0
    for row_16 in data_reader_16:
        all_month_05_matrix[i, 170:350] = row_16
        i = i + 1

    i = 0
    for row_17 in data_reader_17:
        all_month_05_matrix[i, 350:584] = row_17
        i = i + 1

    # Create 06 matrix
    try:
        file_06_15 = open(path[3], newline='')
        data_reader_15 = csv.reader(file_06_15)
        file_06_16 = open(path[10], newline='')
        data_reader_16 = csv.reader(file_06_16)
        file_06_17 = open(path[17], newline='')
        data_reader_17 = csv.reader(file_06_17)
    except IOError:
        print('There was an error opening the file for 06 month!')
        return

    i = 0
    for row_15 in data_reader_15:
        all_month_06_matrix[i, 0:360] = row_15
        i = i + 1

    i = 0
    for row_16 in data_reader_16:
        all_month_06_matrix[i, 360:863] = row_16
        i = i + 1

    i = 0
    for row_17 in data_reader_17:
        all_month_06_matrix[i, 863:1322] = row_17
        i = i + 1

    # Create 07 matrix
    try:
        file_07_15 = open(path[4], newline='')
        data_reader_15 = csv.reader(file_07_15)
        file_07_16 = open(path[11], newline='')
        data_reader_16 = csv.reader(file_07_16)
        file_07_17 = open(path[18], newline='')
        data_reader_17 = csv.reader(file_07_17)
    except IOError:
        print('There was an error opening the file for 07 month!')
        return

    i = 0
    for row_15 in data_reader_15:
        all_month_07_matrix[i, 0:365] = row_15
        i = i + 1

    i = 0
    for row_16 in data_reader_16:
        all_month_07_matrix[i, 365:827] = row_16
        i = i + 1

    i = 0
    for row_17 in data_reader_17:
        all_month_07_matrix[i, 827:1328] = row_17
        i = i + 1

    # Create 08 matrix
    try:
        file_08_15 = open(path[5], newline='')
        data_reader_15 = csv.reader(file_08_15)
        file_08_16 = open(path[12], newline='')
        data_reader_16 = csv.reader(file_08_16)
        file_08_17 = open(path[19], newline='')
        data_reader_17 = csv.reader(file_08_17)
    except IOError:
        print('There was an error opening the file for 08 month!')
        return

    i = 0
    for row_15 in data_reader_15:
        all_month_08_matrix[i, 0:277] = row_15
        i = i + 1

    i = 0
    for row_16 in data_reader_16:
        all_month_08_matrix[i, 277:655] = row_16
        i = i + 1

    i = 0
    for row_17 in data_reader_17:
        all_month_08_matrix[i, 655:1081] = row_17
        i = i + 1

    # Create 09 matrix
    try:
        file_09_15 = open(path[6], newline='')
        data_reader_15 = csv.reader(file_09_15)
        file_09_16 = open(path[13], newline='')
        data_reader_16 = csv.reader(file_09_16)
        file_09_17 = open(path[20], newline='')
        data_reader_17 = csv.reader(file_09_17)
    except IOError:
        print('There was an error opening the file for 09 month!')
        return

    i = 0
    for row_15 in data_reader_15:
        all_month_09_matrix[i, 0:64] = row_15
        i = i + 1

    i = 0
    for row_16 in data_reader_16:
        all_month_09_matrix[i, 64:146] = row_16
        i = i + 1

    i = 0
    for row_17 in data_reader_17:
        all_month_09_matrix[i, 146:312] = row_17
        i = i + 1

    # Create 10 matrix
    try:
        file_10_17 = open(path[21], newline='')
        data_reader_17 = csv.reader(file_10_17)
    except IOError:
        print('There was an error opening the file for 10 month!')
        return

    i = 0
    for row_17 in data_reader_17:
        all_month_10_matrix[i, 0:47] = row_17
        i = i + 1

    # Write month matrices to file
    matrix_to_file(all_month_03_matrix, '03', destination_folder)
    matrix_to_file(all_month_04_matrix, '04', destination_folder)
    matrix_to_file(all_month_05_matrix, '05', destination_folder)
    matrix_to_file(all_month_06_matrix, '06', destination_folder)
    matrix_to_file(all_month_07_matrix, '07', destination_folder)
    matrix_to_file(all_month_08_matrix, '08', destination_folder)
    matrix_to_file(all_month_09_matrix, '09', destination_folder)
    matrix_to_file(all_month_10_matrix, '10', destination_folder)


# Writes a matrix to a csv file. mat is the matrix being written to a file. month_number is a string of the number of
# month of the year. It is used in writing a file name
def matrix_to_file(mat, month_number, destination_folder):
    file = open(destination_folder + 'all_month_' + month_number + '_year_matrix.csv', 'w')

    for i in range(0, mat.shape[0]):
        for j in range(0, mat.shape[1]):
            if j < mat.shape[1] - 1:
                file.write(mat[i, j] + ',')
            else:
                file.write(mat[i, j] + '\n')

    file.close()


# This method puts all data points in a matrix in order according to dates. mat is the matrix than needs to be sorted.
def order_matrix(mat):
    for i in range(0, mat.shape[1]-1):
        col = np.empty([15, 1], dtype=(str, 15))  # Holds the column currently being processed
        col[0, 0] = mat[0, i]  # ID
        col[1, 0] = mat[1, i]  # Locations (Specific locations on each lake)
        col[2, 0] = mat[2, i]  # Locations (lake names)
        col[3, 0] = mat[3, i]  # Dates
        col[4, 0] = mat[4, i]  # Times (24-hour)
        col[5, 0] = mat[5, i]  # algalBlooms
        col[6, 0] = mat[6, i]  # algalBloomSheen
        col[7, 0] = mat[7, i]  # batherLoad
        col[8, 0] = mat[8, i]  # plantDebris
        col[9, 0] = mat[9, i]  # waterAppearance
        col[10, 0] = mat[10, i]  # waterfowlPresence
        col[11, 0] = mat[11, i]  # waveIntensity
        col[12, 0] = mat[12, i]  # waterTemp
        col[13, 0] = mat[13, i]  # turbidity
        col[14, 0] = mat[14, i]  # poor water quality flag

        date = datetime.datetime.strptime(mat[3, i], '%m/%d/%y')
        time = datetime.datetime.strptime(mat[4, i], '%H:%M:%S')

        for j in range(0, mat.shape[1]-1):
            if date == datetime.datetime.strptime(mat[3, j], '%m/%d/%y'):
                if time < datetime.datetime.strptime(mat[4, j], '%H:%M:%S'):
                    insert_column(j, mat, col[:, 0], i)
                    break

            elif date < datetime.datetime.strptime(mat[3, j], '%m/%d/%y'):
                insert_column(j, mat, col[:, 0], i)
                break


# This method shifts a column in the matrix to make room for the insertion of another column. move_to_index is the index
# of the column where col will be inserted.
# mat is the matrix whose columns are being shifted. new_col is the column about to be inserted. move_from_index is the
# current index of col to be inserted at move_to_index
def insert_column(move_to_index, mat, col, move_from_index):
    mat[:, move_from_index] = str(0)
    i = move_to_index
    print("\n")
    print(move_from_index)
    print(move_to_index)
    while i < move_from_index:
        mat[:, i] = mat[:, i-1]
        i = i - 1

    mat[:, i] = col


if __name__ == "__main__": main()