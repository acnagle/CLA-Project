import numpy as np
import Constants


def main():
    np.set_printoptions(threshold=np.inf)  # prints a full matrix rather than an abbreviated matrix

    print("\n\t##### EXECUTING LOCATION_MATRIX_BUILD_NO_NA.PY #####\n")

    # File locations of each csv
    file_paths = [
        "/Users/Alliot/Documents/CLA-Project/Data/matrices-no-na/original/2015_year_matrix.csv",
        "/Users/Alliot/Documents/CLA-Project/Data/matrices-no-na/original/2016_year_matrix.csv",
        "/Users/Alliot/Documents/CLA-Project/Data/matrices-no-na/original/2017_year_matrix.csv",
        "/Users/Alliot/Documents/CLA-Project/Data/all-data-no-na/original/All_data_matrix.csv"
    ]

    destination_folder_no_na = "/Users/Alliot/Documents/CLA-Project/Data/matrices-no-na/original/"
    destination_folder_all_data = "/Users/Alliot/Documents/CLA-Project/Data/all-data-no-na/original/"

    lakes = ["Waubesa", "Kegonsa", "Monona", "Mendota", "Wingra"]

    # Define a matrix for each year of data
    mat15_year = np.genfromtxt(file_paths[0], dtype=str, delimiter=',')
    mat16_year = np.genfromtxt(file_paths[1], dtype=str, delimiter=',')
    mat17_year = np.genfromtxt(file_paths[2], dtype=str, delimiter=',')

    # This array is used to build the location matrices
    yearly_matrices = [mat15_year, mat16_year, mat17_year]

    # Define a matrix for each location where measurements were gathered
    mat_waubesa = np.transpose([np.empty(shape=(Constants.NUM_ROWS_NO_NA, ), dtype=(str, Constants.STR_LENGTH))])
    mat_kegonsa = np.transpose([np.empty(shape=(Constants.NUM_ROWS_NO_NA, ), dtype=(str, Constants.STR_LENGTH))])
    mat_monona = np.transpose([np.empty(shape=(Constants.NUM_ROWS_NO_NA, ), dtype=(str, Constants.STR_LENGTH))])
    mat_mendota = np.transpose([np.empty(shape=(Constants.NUM_ROWS_NO_NA, ), dtype=(str, Constants.STR_LENGTH))])
    mat_wingra = np.transpose([np.empty(shape=(Constants.NUM_ROWS_NO_NA, ), dtype=(str, Constants.STR_LENGTH))])

    # Create location matrices
    mat_waubesa = create_location_matrix(yearly_matrices, mat_loc=mat_waubesa, loc=lakes[0])
    mat_kegonsa = create_location_matrix(yearly_matrices, mat_loc=mat_kegonsa, loc=lakes[1])
    mat_monona = create_location_matrix(yearly_matrices, mat_loc=mat_monona, loc=lakes[2])
    mat_mendota = create_location_matrix(yearly_matrices, mat_loc=mat_mendota, loc=lakes[3])
    mat_wingra = create_location_matrix(yearly_matrices, mat_loc=mat_wingra, loc=lakes[4])

    # Create location matrices for summer months: June (06), July (07), and August (08)
    mat_waubesa_summer = create_summer_location_matrix(mat_waubesa, loc=lakes[0], date_row=1)
    mat_kegonsa_summer = create_summer_location_matrix(mat_kegonsa, loc=lakes[1], date_row=1)
    mat_monona_summer = create_summer_location_matrix(mat_monona, loc=lakes[2], date_row=1)
    mat_mendota_summer = create_summer_location_matrix(mat_mendota, loc=lakes[3], date_row=1)
    mat_wingra_summer = create_summer_location_matrix(mat_wingra, loc=lakes[4], date_row=1)

    # Write the matrices to csv files
    matrix_to_file(mat_waubesa, lake_name=lakes[0], destination_folder=destination_folder_no_na)
    matrix_to_file(mat_kegonsa, lake_name=lakes[1], destination_folder=destination_folder_no_na)
    matrix_to_file(mat_monona, lake_name=lakes[2], destination_folder=destination_folder_no_na)
    matrix_to_file(mat_mendota, lake_name=lakes[3], destination_folder=destination_folder_no_na)
    matrix_to_file(mat_wingra, lake_name=lakes[4], destination_folder=destination_folder_no_na)

    matrix_to_file(mat_waubesa_summer, lake_name=lakes[0] + "_summer", destination_folder=destination_folder_no_na)
    matrix_to_file(mat_kegonsa_summer, lake_name=lakes[1] + "_summer", destination_folder=destination_folder_no_na)
    matrix_to_file(mat_monona_summer, lake_name=lakes[2] + "_summer", destination_folder=destination_folder_no_na)
    matrix_to_file(mat_mendota_summer, lake_name=lakes[3] + "_summer", destination_folder=destination_folder_no_na)
    matrix_to_file(mat_wingra_summer, lake_name=lakes[4] + "_summer", destination_folder=destination_folder_no_na)

    # Create location matrices for All_Data_matrix
    mat_all_data = np.genfromtxt(file_paths[3], dtype=str, delimiter=',')

    mat_waubesa = np.transpose([np.empty(shape=(mat_all_data.shape[Constants.ROWS], ),
                                         dtype=(str, Constants.STR_LENGTH))])
    mat_kegonsa = np.transpose([np.empty(shape=(mat_all_data.shape[Constants.ROWS], ),
                                         dtype=(str, Constants.STR_LENGTH))])
    mat_monona = np.transpose([np.empty(shape=(mat_all_data.shape[Constants.ROWS], ),
                                        dtype=(str, Constants.STR_LENGTH))])
    mat_mendota = np.transpose([np.empty(shape=(mat_all_data.shape[Constants.ROWS], ),
                                         dtype=(str, Constants.STR_LENGTH))])
    mat_wingra = np.transpose([np.empty(shape=(mat_all_data.shape[Constants.ROWS], ),
                                        dtype=(str, Constants.STR_LENGTH))])

    for i in range(0, mat_all_data.shape[Constants.COLUMNS]-1):
        location = mat_all_data[Constants.LOCATION, i]
        if lakes[0] in location:
            mat_waubesa = np.hstack([mat_waubesa, np.transpose([mat_all_data[:, i]])])
        if lakes[1] in location:
            mat_kegonsa = np.hstack([mat_kegonsa, np.transpose([mat_all_data[:, i]])])
        if lakes[2] in location:
            mat_monona = np.hstack([mat_monona, np.transpose([mat_all_data[:, i]])])
        if lakes[3] in location:
            mat_mendota = np.hstack([mat_mendota, np.transpose([mat_all_data[:, i]])])
        if lakes[4] in location:
            mat_wingra = np.hstack([mat_wingra, np.transpose([mat_all_data[:, i]])])

    print("Building " + lakes[0] + " All Data matrix ...")
    mat_waubesa = mat_waubesa[:, 1:]
    matrix_to_file(mat_waubesa, lake_name=lakes[0], destination_folder=destination_folder_all_data)
    print("Building " + lakes[1] + " All Data matrix ...")
    mat_kegonsa = mat_kegonsa[:, 1:]
    matrix_to_file(mat_kegonsa, lake_name=lakes[1], destination_folder=destination_folder_all_data)
    print("Building " + lakes[2] + " All Data matrix ...")
    mat_monona = mat_monona[:, 1:]
    matrix_to_file(mat_monona, lake_name=lakes[2], destination_folder=destination_folder_all_data)
    print("Building " + lakes[3] + " All Data matrix ...")
    mat_mendota = mat_mendota[:, 1:]
    matrix_to_file(mat_mendota, lake_name=lakes[3], destination_folder=destination_folder_all_data)
    print("Building " + lakes[4] + " All Data matrix ...")
    mat_wingra = mat_wingra[:, 1:]
    matrix_to_file(mat_wingra, lake_name=lakes[4], destination_folder=destination_folder_all_data)

    mat_waubesa_summer = create_summer_location_matrix(mat_waubesa, loc=lakes[0] + " All Data", date_row=1)
    mat_kegonsa_summer = create_summer_location_matrix(mat_kegonsa, loc=lakes[1] + " All Data", date_row=1)
    mat_monona_summer = create_summer_location_matrix(mat_monona, loc=lakes[2] + " All Data", date_row=1)
    mat_mendota_summer = create_summer_location_matrix(mat_mendota, loc=lakes[3] + " All Data", date_row=1)
    mat_wingra_summer = create_summer_location_matrix(mat_wingra, loc=lakes[4] + " All Data", date_row=1)

    matrix_to_file(mat_waubesa_summer, lake_name=lakes[0] + "_All_Data_summer",
                   destination_folder=destination_folder_all_data)
    matrix_to_file(mat_kegonsa_summer, lake_name=lakes[1] + "_All_Data_summer",
                   destination_folder=destination_folder_all_data)
    matrix_to_file(mat_monona_summer, lake_name=lakes[2] + "_All_Data_summer",
                   destination_folder=destination_folder_all_data)
    matrix_to_file(mat_mendota_summer, lake_name=lakes[3] + "_All_Data_summer",
                   destination_folder=destination_folder_all_data)
    matrix_to_file(mat_wingra_summer, lake_name=lakes[4] + "_All_Data_summer",
                   destination_folder=destination_folder_all_data)

    print("\n")


# Writes a matrix to a csv file. mat is the matrix being written to a file. lake_name is a string of the name of the
# lake. It is used in writing a file name. destination_folder_no_na is the path to the destination folder where the
# csv file will be stored
def matrix_to_file(mat, lake_name, destination_folder):
    file = open(destination_folder + lake_name + "_matrix.csv", "w")
    for i in range(0, mat.shape[Constants.ROWS]):
        for j in range(0, mat.shape[Constants.COLUMNS]):
            if j < mat.shape[Constants.COLUMNS] - 1:
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
        for j in range(0, mat.shape[Constants.COLUMNS]-1):
            if mat[2, j] == locs[i]:
                count = count + 1

        num_array[i] = count


# This method forms a matrix for a specific location across all years. yearly_matrices is an array containing every
# year data was taken from. mat_loc is matrix created for a specified location and is returned. loc is the name of
# the location
def create_location_matrix(yearly_matrices, mat_loc, loc):
    print("Building " + loc + " matrix ...")
    mat_loc_index = 0   # used to index mat_loc
    for i in range(0, len(yearly_matrices)):
        for j in range(0, yearly_matrices[i].shape[Constants.COLUMNS]-1):
            if loc in yearly_matrices[i][Constants.LOCATION, j]:
                # mat_loc[:, mat_loc_index] = yearly_matrices[i][:, j]
                # mat_loc_index = mat_loc_index + 1
                mat_loc = np.hstack([mat_loc, np.transpose([yearly_matrices[i][:, j]])])

    # The first entry is empty from initialization. This entry will remove that entry.
    mat_loc = np.delete(mat_loc, obj=0, axis=Constants.COLUMNS)

    return mat_loc


# This method creates a matrix for only the summer months (June through August) for each lake location. mat_loc is the
# matrix containing all data points across years 2015, 2016, and 2017 for one lake location (same matrix as the output
# of create_location_matrix method). loc is the name of the lake for which the output matrix of this method is for.
# mat_loc_summer is the output matrix of this method. date_row is the row in which the date is stored in mat_loc.
def create_summer_location_matrix(mat_loc, loc, date_row):
    print("Building " + loc + " summer matrix ...")
    mat_loc_summer = np.transpose([np.empty((mat_loc.shape[Constants.ROWS], ))])
    for j in range(1, mat_loc.shape[Constants.COLUMNS]):
        new_date_str = mat_loc[date_row, j]
        if "6" == new_date_str[:1] or "7" == new_date_str[:1] or "8" == new_date_str[:1]:
            mat_loc_summer = np.hstack([mat_loc_summer, np.transpose([mat_loc[:, j]])])

    # Remove zero column in beginning of mat_loc_summer (zero column came from instantiating mat_loc_summer
    # as an empty array)
    mat_loc_summer = mat_loc_summer[:, 1:]

    return mat_loc_summer


if __name__ == "__main__": main()
