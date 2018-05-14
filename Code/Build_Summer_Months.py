import csv
import datetime
import numpy as np
import os
import glob

def main():
    np.set_printoptions(threshold=np.inf)   # prints a full matrix rather than an abbreviated matrix

    print("\n\t##### EXECUTING BUILD_SUMMER_MONTHS_NO_NA.PY #####")

    src_path = dest_path = "/Users/Alliot/documents/cla-project/data/matrices-no-na/original/"

    summer_month_paths = []

    # get filenames of all files for the summer months: June (06), July (07), and August (08)
    for filename in glob.glob(os.path.join(src_path, "*.csv")):
        if ("06" in filename or "07" in filename or "08" in filename) and (not "all" in filename):
            summer_month_paths.append(filename)

    # put the filenames in chronological order
    summer_month_paths = np.sort(summer_month_paths)

    # create matrices the summer months of each year
    summer_2015 = np.empty((15, 0))
    summer_2016 = np.empty((15, 0))
    summer_2017 = np.empty((15, 0))

    for path in summer_month_paths:
        if "2015" in path:
            summer_2015 = np.hstack((summer_2015, np.genfromtxt(open(path, "rb"), delimiter=",", dtype="str")))
        elif "2016" in path:
            summer_2016 = np.hstack((summer_2016, np.genfromtxt(open(path, "rb"), delimiter=",", dtype="str")))
        elif "2017" in path:
            summer_2017 = np.hstack((summer_2017, np.genfromtxt(open(path, "rb"), delimiter=",", dtype="str")))

    # write summer matrices to .csv file
    print("Processing file " + "summer_2015_matrix.csv ...")
    matrix_to_file(summer_2015, "summer_2015_matrix.csv", dest_path)

    print("Processing file " + "summer_2016_matrix.csv ...")
    matrix_to_file(summer_2016, "summer_2016_matrix.csv", dest_path)

    print("Processing file " + "summer_2017_matrix.csv ...")
    matrix_to_file(summer_2017, "summer_2017_matrix.csv", dest_path)


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

