import numpy as np
from datetime import datetime


# the height of the matrices. ie the number of measurements per sample. IMPORTANT NOTE: The .csv files being read in
# by this code has 15 rows. The last row is a "poor water quality flag" (binary) that is 1 if the turbidity is below
# 50 and 0 otherwise. By choosing num_rows = 14, I'm eliminating this row.
num_rows = 14

def main():
    np.set_printoptions(threshold=np.inf)  # prints a full matrix rather than an abbreviated matrix

    print("\n\t##### EXECUTING LOCATION_MATRIX_BUILD_NO_NA.PY #####")

    # File locations of each csv
    file_paths = [
        "/Users/Alliot/Documents/CLA-Project/Data/matrices-no-na/original/2015_year_matrix.csv",
        "/Users/Alliot/Documents/CLA-Project/Data/matrices-no-na/original/2016_year_matrix.csv",
        "/Users/Alliot/Documents/CLA-Project/Data/matrices-no-na/original/2017_year_matrix.csv"
    ]

    destination_folder = "/Users/Alliot/Documents/CLA-Project/Data/matrices-no-na/original/"

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
    mat_waubesa = np.empty([num_rows, num_measurements_per_location15[0] + num_measurements_per_location16[0] +
                           num_measurements_per_location17[0]], dtype=(str, 15))
    mat_kegonsa = np.empty([num_rows, num_measurements_per_location15[1] + num_measurements_per_location16[1] +
                           num_measurements_per_location17[1]], dtype=(str, 15))
    mat_monona = np.empty([num_rows, num_measurements_per_location15[2] + num_measurements_per_location16[2] +
                           num_measurements_per_location17[2]], dtype=(str, 15))
    mat_mendota = np.empty([num_rows, num_measurements_per_location15[3] + num_measurements_per_location16[3] +
                           num_measurements_per_location17[3]], dtype=(str, 15))
    mat_wingra = np.empty([num_rows, num_measurements_per_location15[4] + num_measurements_per_location16[4] +
                           num_measurements_per_location17[4]], dtype=(str, 15))

    # Create location matrices
    create_location_matrix(yearly_matrices, mat_waubesa, lakes[0])
    create_location_matrix(yearly_matrices, mat_kegonsa, lakes[1])
    create_location_matrix(yearly_matrices, mat_monona, lakes[2])
    create_location_matrix(yearly_matrices, mat_mendota, lakes[3])
    create_location_matrix(yearly_matrices, mat_wingra, lakes[4])

    # Create location matrices for summer months: June (06), July (07), and August (08)
    mat_waubesa_summer = create_summer_location_matrix(mat_waubesa, lakes[0])
    mat_kegonsa_summer = create_summer_location_matrix(mat_kegonsa, lakes[1])
    mat_monona_summer = create_summer_location_matrix(mat_monona, lakes[2])
    mat_mendota_summer = create_summer_location_matrix(mat_mendota, lakes[3])
    mat_wingra_summer = create_summer_location_matrix(mat_wingra, lakes[4])

    # Write the matrices to csv files
    matrix_to_file(mat_waubesa, lakes[0], destination_folder)
    matrix_to_file(mat_kegonsa, lakes[1], destination_folder)
    matrix_to_file(mat_monona, lakes[2], destination_folder)
    matrix_to_file(mat_mendota, lakes[3], destination_folder)
    matrix_to_file(mat_wingra, lakes[4], destination_folder)

    matrix_to_file(mat_waubesa_summer, lakes[0] + "_summer", destination_folder)
    matrix_to_file(mat_kegonsa_summer, lakes[1] + "_summer", destination_folder)
    matrix_to_file(mat_monona_summer, lakes[2] + "_summer", destination_folder)
    matrix_to_file(mat_mendota_summer, lakes[3] + "_summer", destination_folder)
    matrix_to_file(mat_wingra_summer, lakes[4] + "_summer", destination_folder)


# Writes a matrix to a csv file. mat is the matrix being written to a file. lake_name is a string of the name of the
# lake. It is used in writing a file name. destination_folder is the path to the destination folder where the
# csv file will be stored
def matrix_to_file(mat, lake_name, destination_folder):
    file = open(destination_folder + lake_name + "_matrix.csv", "w")
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
    print("Building " + loc + " matrix ...")
    mat_loc_index = 0   # used to index mat_loc
    for i in range(0, len(yearly_matrices)):
        for j in range(0, yearly_matrices[i].shape[1]-1):
            if yearly_matrices[i][2, j] == loc:
                mat_loc[:, mat_loc_index] = yearly_matrices[i][:, j]
                mat_loc_index = mat_loc_index + 1


# This method creates a matrix for only the summer months (June through August) for each lake location. mat_loc is the
# matrix containing all data points across years 2015, 2016, and 2017 for one lake location (same matrix as the output
# of create_location_matrix method). loc is the name of the lake for which the output matrix of this method is for.
# mat_loc_summer is the output matrix of this method.
def create_summer_location_matrix(mat_loc, loc):
    print("Building " + loc + " summer matrix ...")
    mat_loc_summer = np.transpose([np.empty((num_rows, ))])
    for j in range(1, mat_loc.shape[1]):
        new_date_str = mat_loc[3, j]
        if "6" == new_date_str[:1] or "7" == new_date_str[:1] or "8" == new_date_str[:1]:
            mat_loc_summer = np.hstack([mat_loc_summer, np.transpose([mat_loc[:, j]])])

    # Remove zero column in beginning of mat_loc_summer (zero column came from instantiating mat_loc_summer
    # as an empty array)
    mat_loc_summer = mat_loc_summer[:, 1:]

    return mat_loc_summer

if __name__ == "__main__": main()