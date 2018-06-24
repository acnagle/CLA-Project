import numpy as np
import os
import glob

# the height of the matrices for matrices_no_na file path. ie the number of measurements per sample. IMPORTANT NOTE:
# The .csv files being read in by this code has 15 rows. The last row is a "poor water quality flag" (binary) that is 1
# if the turbidity is below 50 and 0 otherwise. By choosing num_rows = 14, I'm eliminating this row.
num_rows = 14

def main():
    np.set_printoptions(threshold=np.inf)   # prints a full matrix rather than an abbreviated matrix

    print("\n\t##### EXECUTING BUILD_SUMMER_MONTHS_NO_NA.PY #####")

    src_path_no_na = dest_path_no_na = "/Users/Alliot/documents/cla-project/data/matrices-no-na/original/"
    src_path_all_data = dest_path_all_data = "/Users/Alliot/documents/cla-project/data/all-data-no-na/original/"

    summer_month_paths_no_na = []
    summer_month_paths_all_data = []

    # build summer months for matrices_no_na directory
    # get filenames of all files for the summer months: June (06), July (07), and August (08)
    for filename in glob.glob(os.path.join(src_path_no_na, "*.csv")):
        if ("06" in filename or "07" in filename or "08" in filename) and (not "all" in filename):
            summer_month_paths_no_na.append(filename)

    # put the filenames in chronological order
    summer_month_paths_no_na = np.sort(summer_month_paths_no_na)

    # create matrices the summer months of each year
    summer_2015 = np.empty((num_rows, 0))
    summer_2016 = np.empty((num_rows, 0))
    summer_2017 = np.empty((num_rows, 0))

    for path in summer_month_paths_no_na:
        if "2015" in path:
            summer_2015 = np.hstack((summer_2015, np.genfromtxt(open(path, "rb"), delimiter=",", dtype="str")))
        elif "2016" in path:
            summer_2016 = np.hstack((summer_2016, np.genfromtxt(open(path, "rb"), delimiter=",", dtype="str")))
        elif "2017" in path:
            summer_2017 = np.hstack((summer_2017, np.genfromtxt(open(path, "rb"), delimiter=",", dtype="str")))

    # write summer matrices to .csv file
    print("Processing file " + "2015_summer_matrix.csv ...")
    matrix_to_file(summer_2015, "2015_summer_matrix.csv", dest_path_no_na)

    print("Processing file " + "2016_summer_matrix.csv ...")
    matrix_to_file(summer_2016, "2016_summer_matrix.csv", dest_path_no_na)

    print("Processing file " + "2017_summer_matrix.csv ...")
    matrix_to_file(summer_2017, "2017_summer_matrix.csv", dest_path_no_na)

    # build summer months for all-data-no-na directory
    print("Processing file " + "All_Data_summer_matrix.csv ...")
    # get filenames of all files for the summer months: June (06), July (07), and August (08)
    for filename in glob.glob(os.path.join(src_path_all_data, "*.csv")):
        if "summer" in filename:
            summer_month_paths_all_data.append(filename)

    # put the filenames in chronological order
    summer_month_paths_all_data = np.sort(summer_month_paths_all_data)

    # create matrices the summer months of each year
    all_data_summer = np.empty((15, 0))

    for path in summer_month_paths_all_data:
        all_data_summer = np.hstack((all_data_summer, np.genfromtxt(open(path, "rb"), delimiter=",", dtype="str")))

    # write summer matrix to .csv file
    matrix_to_file(all_data_summer, "All_Data_summer_matrix.csv", dest_path_all_data)


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


if __name__ == "__main__": main()

