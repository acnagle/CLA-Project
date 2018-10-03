import numpy as np
import Constants


def main():
    np.set_printoptions(threshold=np.inf)  # prints a full matrix rather than an abbreviated matrix

    print("\n\t##### EXECUTING FORMAT_DATA.PY #####\n")

    # data paths
    file_paths = [
        "/Users/Alliot/documents/CLA-project/Data/CM2014_edit.csv",
        "/Users/Alliot/documents/CLA-project/Data/CM2015_edit.csv",
        "/Users/Alliot/documents/CLA-project/Data/CM2016_edit.csv",
        "/Users/Alliot/documents/CLA-project/Data/CM2017_edit.csv",
        "/Users/Alliot/documents/cla-project/data/algal_bloom_locations_summaries.csv"
    ]

    # read in the data
    data_2014 = np.genfromtxt(
        open(file_paths[0], "rb"),
        delimiter=",",
        dtype=(str, Constants.STR_LENGTH),
        usecols=(1, 9, 10, 4, 5, 6, 7, 12, 13, 15, 14, 11),
        filling_values="NA",
        skip_header=1
    )

    data_2015 = np.genfromtxt(
        open(file_paths[1], "rb"),
        delimiter=",",
        dtype=(str, Constants.STR_LENGTH),
        usecols=(1, 14, 15, 6, 7, 8, 10, 17, 18, 20, 19, 16),
        filling_values="NA",
        skip_header=1
    )

    data_2016 = np.genfromtxt(
        open(file_paths[2], "rb"),
        delimiter=",",
        dtype=(str, Constants.STR_LENGTH),
        usecols=(1, 14, 15, 6, 7, 8, 10, 17, 18, 20, 19, 16),
        filling_values="NA",
        skip_header=1
    )

    data_2017 = np.genfromtxt(
        open(file_paths[3], "rb"),
        delimiter=",",
        dtype=(str, Constants.STR_LENGTH),
        usecols=(1, 14, 15, 6, 7, 8, 10, 17, 18, 20, 19, 16),
        filling_values="NA",
        skip_header=1
    )

    data_summary = np.genfromtxt(
        open(file_paths[4], "rb"),
        delimiter=",",
        dtype=(str, Constants.STR_LENGTH),
        usecols=(0, 20, 12, 13, 14, 16, 24, 25, 27, 26, 23, 6, 4, 5, 8),
        skip_header=1,
        invalid_raise=False
    )

    # clean data
    data_2014 = clean_data(data_2014, summary=False)
    data_2015 = clean_data(data_2015, summary=False)
    data_2016 = clean_data(data_2016, summary=False)
    data_2017 = clean_data(data_2017, summary=False)
    data_summary = clean_data(data_summary, summary=True)

    # form normalized matrices

    # create labels

    #


# This method takes a matrix and cleans it by removing any feature vectors with "NA", "FALSE", "TRUE", "undefined", or
# empty entries. new_mat is the clean version of mat, and is returned. summary is a boolean flag to indicate whether
# mat is the summary matrix
def clean_data(mat, summary):
    new_mat = np.empty(shape=(mat.shape[Constants.ROWS], Constants.NUM_ROWS_NO_NA), dtype=(str, Constants.STR_LENGTH))

    idx = 0
    if ~summary:
        for i in range(0, mat.shape[Constants.ROWS]):
            if ("NA" not in mat[i, :] and "" not in mat[i, :] and "FALSE" not in mat[i, :] and
                    "TRUE" not in mat[i, :] and "undefined" not in mat[i, :]):
                new_mat[idx, Constants.LOCATION] = mat[i, 0]                      # Locations
                new_mat[idx, Constants.DATE_TIME] = mat[i, 1] + " " + mat[i, 2]   # Dates and Time (24-hour)
                new_mat[idx, Constants.ALGAL_BLOOMS] = mat[i, 3]                  # algalBlooms
                new_mat[idx, Constants.ALGAL_BLOOM_SHEEN] = mat[i, 4]             # algalBloomSheen
                new_mat[idx, Constants.BATHER_LOAD] = mat[i, 5]                   # batherLoad
                new_mat[idx, Constants.PLANT_DEBRIS] = mat[i, 6]                  # plantDebris
                new_mat[idx, Constants.WATER_APPEARANCE] = mat[i, 7]              # waterAppearance
                new_mat[idx, Constants.WATER_FOWL_PRESENCE] = mat[i, 8]           # waterfowlPresence
                new_mat[idx, Constants.WAVE_INTENSITY] = mat[i, 9]                # waveIntensity
                new_mat[idx, Constants.WATER_TEMP] = mat[i, 10]                   # waterTemp
                new_mat[idx, Constants.TURBIDITY] = mat[i, 11]                    # turbidity

                # adjust data entries so that qualitative measurements (except algalBloomSheen) have values [1, 3]
                for j in range(Constants.ALGAL_BLOOMS, Constants.WATER_TEMP):
                    if j != Constants.ALGAL_BLOOM_SHEEN:  # ignore algalBloomSheen measurement
                        if float(new_mat[idx, j]) == 0:
                            new_mat[idx, j] = "1"

                idx += 1

    else:
        for i in range(0, mat.shape[Constants.ROWS]):
            if "" not in mat[i, :] and "NA" not in mat[i, :] and "FALSE" not in mat[i, :] and "#VALUE" not in mat[i, :]:
                new_mat[idx, Constants.LOCATION] = mat[i, 0]                # Locations
                new_mat[idx, Constants.DATE_TIME] = mat[i, 1]               # Date and Time (24-hour)
                new_mat[idx, Constants.ALGAL_BLOOMS] = mat[i, 2]            # algalBlooms
                new_mat[idx, Constants.ALGAL_BLOOM_SHEEN] = mat[i, 3]       # algalBloomSheen
                new_mat[idx, Constants.BATHER_LOAD] = mat[i, 4]             # batherLoad
                new_mat[idx, Constants.PLANT_DEBRIS] = mat[i, 5]            # plantDebris
                new_mat[idx, Constants.WATER_APPEARANCE] = mat[i, 6]        # waterAppearance
                new_mat[idx, Constants.WATER_FOWL_PRESENCE] = mat[i, 7]     # waterfowlPresence
                new_mat[idx, Constants.WAVE_INTENSITY] = mat[i, 8]          # waveIntensity
                new_mat[idx, Constants.WATER_TEMP] = mat[i, 9]              # waterTemp
                new_mat[idx, Constants.TURBIDITY] = mat[i, 10]              # turbidity
                new_mat[idx, Constants.AIR_TEMP] = mat[i, 11]               # airTemp
                new_mat[idx, Constants.PRCP_24_HRS] = mat[i, 12]            # prcp_24rs
                new_mat[idx, Constants.PRCP_48_HRS] = mat[i, 13]            # prcp_48hrs
                new_mat[idx, Constants.WINDSPEED_AVG_24_HRS] = mat[i, 14]   # windspeed_avg_24hr

                # adjust data entries so that qualitative measurements (except algalBloomSheen) have values [1, 3]
                for j in range(Constants.ALGAL_BLOOMS, Constants.WATER_TEMP):
                    if j != Constants.ALGAL_BLOOM_SHEEN:  # ignore algalBloomSheen measurement
                        if float(new_mat[idx, j]) == 0:
                            new_mat[idx, j] = "1"

                idx = idx + 1

    new_mat = new_mat[0:idx, :]

    return new_mat


if __name__ == "__main__": main()