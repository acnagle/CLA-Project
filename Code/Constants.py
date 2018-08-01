# Define useful constants to be used throughout other files
NUM_ROWS_NO_IND = 13    # number of measurements per data point for data with no indicator (All-Data matrices)
NUM_ROWS_W_IND = 15     # number of measurements per data point for data with indicator (All-Data matrices)
NUM_ROWS_3D_PROJ = 3    # number of rows in a 3D projection matrix

FIRST_ROW = 1   # row number in which the data in the matrices becomes quantitative. Adjusting this number is useful
                # for leaving out potentially unwanted measurements, such as location, or date and time.
