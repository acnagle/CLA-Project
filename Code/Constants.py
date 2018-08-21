# Define useful constants to be used throughout other files
NUM_ROWS_NO_IND_ALL_DATA = 13    # number of measurements per data point for data with no indicator (All-Data matrices)
NUM_ROWS_W_IND_ALL_DATA = 15     # number of measurements per data point for data with indicator (All-Data matrices)
NUM_ROWS_3D_PROJ_ALL_DATA = 3    # number of rows in a 3D projection matrix

NUM_ROWS_NO_NA = 11     # The number of measurements per data point for _no_na matrices

FIRST_ROW = 2   # row number in which the data in the matrices becomes quantitative. Adjusting this number is useful
                # for leaving out potentially unwanted measurements, such as location, or date and time.

STR_LENGTH = 22     # STR_LENGTH represents the maximum number of characters in a string for the arrays I use

# Indices for each measurement in the matrices
LOCATION = 0
DATE_TIME = 1
ALGAL_BLOOMS = 2
ALGAL_BLOOM_SHEEN = 3
BATHER_LOAD = 4
PLANT_DEBRIS = 5
WATER_APPEARANCE = 6
WATER_FOWL_PRESENCE = 7
WAVE_INTENSITY = 8
WATER_TEMP = 9
TURBIDITY = 10
AIR_TEMP = 11
PRCP_24_HRS = 12
PRCP_48_HRS = 13
WINDSPEED_AVG_24_HRS = 14

# constants used to represent the axis of a matrix
ROWS = 0
COLUMNS = 1

NUM_LOCATIONS = 5   # Number of locations (lakes) that the data is collected from

NUM_LAKES = 75
