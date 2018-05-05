import numpy as np

def main():
    np.set_printoptions(threshold=np.inf)  # prints a full matrix rather than an abbreviated matrix

    print('\n\t##### EXECUTING LOCATION_MATRIX_BUILD.PY #####')

    # File locations of each csv
    file_paths = [
        "/Users/Alliot/Documents/CLA Project/Data/matrices/2015_year_matrix.csv",
        "/Users/Alliot/Documents/CLA Project/Data/matrices/2016_year_matrix.csv",
        "/Users/Alliot/Documents/CLA Project/Data/matrices/2017_year_matrix.csv"
    ]

    destination_folder = '/Users/Alliot/Documents/CLA Project/Data/matrices/'

    lakes = ["Waubesa", "Kegonsa", "Monona", "Mendota", "Wingra"]

    # Define a matrix for each year of data
    mat15_year = np.genfromtxt(file_paths[0], dtype=str, delimiter=',')
    mat16_year = np.genfromtxt(file_paths[1], dtype=str, delimiter=',')
    mat17_year = np.genfromtxt(file_paths[2], dtype=str, delimiter=',')

    # This array is used to build the location matrices
    yearly_matrices = [mat15_year, mat16_year, mat17_year]

    # The following three matrices define the number of measurements recorded per location per year
    num_measurements_per_location15 = np.zeros(5, dtype=int)
    num_measurements_per_location16 = np.zeros(5, dtype=int)
    num_measurements_per_location17 = np.zeros(5, dtype=int)

    get_num_locations(mat15_year, num_measurements_per_location15, lakes)
    get_num_locations(mat16_year, num_measurements_per_location16, lakes)
    get_num_locations(mat17_year, num_measurements_per_location17, lakes)

    # Define a matrix for each location where measurements were gathered
    mat_waubesa = np.empty([15, num_measurements_per_location15[0] + num_measurements_per_location16[0] +
                           num_measurements_per_location17[0]], dtype=(str, 15))
    mat_kegonsa = np.empty([15, num_measurements_per_location15[1] + num_measurements_per_location16[1] +
                           num_measurements_per_location17[1]], dtype=(str, 15))
    mat_monona = np.empty([15, num_measurements_per_location15[2] + num_measurements_per_location16[2] +
                           num_measurements_per_location17[2]], dtype=(str, 15))
    mat_mendota = np.empty([15, num_measurements_per_location15[3] + num_measurements_per_location16[3] +
                           num_measurements_per_location17[3]], dtype=(str, 15))
    mat_wingra = np.empty([15, num_measurements_per_location15[4] + num_measurements_per_location16[4] +
                           num_measurements_per_location17[4]], dtype=(str, 15))

    # Create location matrices
    create_location_matrix(yearly_matrices, mat_waubesa, lakes[0])
    create_location_matrix(yearly_matrices, mat_kegonsa, lakes[1])
    create_location_matrix(yearly_matrices, mat_monona, lakes[2])
    create_location_matrix(yearly_matrices, mat_mendota, lakes[3])
    create_location_matrix(yearly_matrices, mat_wingra, lakes[4])

    # Write the matrices to csv files
    matrix_to_file(mat_waubesa, lakes[0], destination_folder)
    matrix_to_file(mat_kegonsa, lakes[1], destination_folder)
    matrix_to_file(mat_monona, lakes[2], destination_folder)
    matrix_to_file(mat_mendota, lakes[3], destination_folder)
    matrix_to_file(mat_wingra, lakes[4], destination_folder)


# Writes a matrix to a csv file. mat is the matrix being written to a file. lake_name is a string of the name of the
# lake. It is used in writing a file name
def matrix_to_file(mat, lake_name, destination_folder):
    file = open(destination_folder + lake_name + '_matrix.csv', 'w')

    for i in range(0, mat.shape[0]):
        for j in range(0, mat.shape[1]):
            if j < mat.shape[1] - 1:
                file.write(mat[i, j] + ',')
            else:
                file.write(mat[i, j] + '\n')

    file.close()


# This method stores the number of occurrences of each element in locs (locations) in their respective indices in
# num_array. So for example, the number of occurrences of the first location (first element in locs) will be stored
# as the first element in num_array. mat is the array containing the locations
def get_num_locations(mat, num_array, locs):
    for i in range(0, len(locs)):
        count = 0   # counts the number of location occurrences
        for j in range(0, mat.shape[1]-1):
            if mat[2, j] == locs[i]:
                count = count + 1

        num_array[i] = count


# This method forms a matrix for a specific location across all years. yearly_matrices is an array containing every
# year data was taken from. mat_loc is matrix created for a specified location. loc is the name of the location
def create_location_matrix(yearly_matrices, mat_loc, loc):
    mat_loc_index = 0   # used to index mat_loc
    for i in range(0, len(yearly_matrices)):
        for j in range(0, yearly_matrices[i].shape[1]-1):
            # print('yearly: ' + str(yearly_matrices[i][:, j]))
            # print('loc: ' + loc)
            if yearly_matrices[i][2, j] == loc:
                mat_loc[:, mat_loc_index] = yearly_matrices[i][:, j]
                mat_loc_index = mat_loc_index + 1


if __name__ == "__main__": main()