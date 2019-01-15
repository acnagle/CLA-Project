# Define useful constants to be used throughout other files
NUM_ROWS_NO_IND_ALL_DATA = 13    # number of measurements per data point for data with no indicator (All-Data matrices)
NUM_ROWS_W_IND_ALL_DATA = 15     # number of measurements per data point for data with indicator (All-Data matrices)

# number of measurements per data point for data with no indicator (All-Data matrices, excluding location)
NUM_ROWS_NO_IND_NO_LOC_ALL_DATA = 12
# number of measurements per data point for data with indicator (All-Data matrices, excluding location)
NUM_ROWS_W_IND_NO_LOC_ALL_DATA = 14
NUM_ROWS_3D_PROJ_ALL_DATA = 3    # number of rows in a 3D projection matrix
NUM_ROWS_NO_NA = 11     # The number of measurements per data point for _no_na matrices
NUM_ROWS_NO_LOC_NO_NA = 10     # The number of measurements per data point for _no_na matrices (excluding location)

FIRST_ROW = 1   # row number in which the data in the matrices becomes quantitative. Adjusting this number is useful
                # for leaving out potentially unwanted measurements, such as location, or date and time.

STR_LENGTH = 22     # STR_LENGTH represents the maximum number of characters in a string for the arrays I use

# Indices for each measurement in the matrices
### THESE ARE OLD
# LOCATION = 0
# DATE_TIME = 1
# ALGAL_BLOOMS = 2
# ALGAL_BLOOM_SHEEN = 3
# BATHER_LOAD = 4
# PLANT_DEBRIS = 5
# WATER_APPEARANCE = 6
# WATER_FOWL_PRESENCE = 7
# WAVE_INTENSITY = 8
# WATER_TEMP = 9
# TURBIDITY = 10
# AIR_TEMP = 11
# PRCP_24_HRS = 12
# PRCP_48_HRS = 13
# WINDSPEED_AVG_24_HRS = 14

### THESE ARE NEW
LOCATION = 0
DATE = 1
TIME = 2
ALGAL_BLOOMS = 3
ALGAL_BLOOM_SHEEN = 4
WATER_APPEARANCE = 5
WAVE_INTENSITY = 6
WATER_TEMP = 7
TURBIDITY = 8

# Indices for each measurement in the matrices (excluding location)
### THESE ARE OLD
# DATE_TIME_NO_LOC = 0
# ALGAL_BLOOMS_NO_LOC = 1
# ALGAL_BLOOM_SHEEN_NO_LOC = 2
# BATHER_LOAD_NO_LOC = 3
# PLANT_DEBRIS_NO_LOC = 4
# WATER_APPEARANCE_NO_LOC = 5
# WATER_FOWL_PRESENCE_NO_LOC = 6
# WAVE_INTENSITY_NO_LOC = 7
# WATER_TEMP_NO_LOC = 8
# TURBIDITY_NO_LOC = 9
# AIR_TEMP_NO_LOC = 10
# PRCP_24_HRS_NO_LOC = 11
# PRCP_48_HRS_NO_LOC = 12
# WINDSPEED_AVG_24_HRS_NO_LOC = 13

### THESE ARE NEW
DATE_NO_LOC = 0
TIME_NO_LOC = 1
ALGAL_BLOOMS_NO_LOC = 2
ALGAL_BLOOM_SHEEN_NO_LOC = 3
WATER_APPEARANCE_NO_LOC = 4
WAVE_INTENSITY_NO_LOC = 5
WATER_TEMP_NO_LOC = 6
TURBIDITY_NO_LOC = 7

# Weather Data Constants. The commented numbers are the columns in the CSV file that the measurements can be found
WEATHER_DATE_TIME = 0                   # 5
WEATHER_MAX_DRY_BULB_TEMP = 1           # 26
WEATHER_MIN_DRY_BULB_TEMP = 2           # 27
WEATHER_AVG_DRY_BULB_TEMP = 3           # 28
WEATHER_DEPT_NORM_AVG_TEMP = 4          # 29
WEATHER_AVG_REL_HUMIDITY = 5            # 30    removed
WEATHER_AVG_DEW_POINT_TEMP = 6          # 31    some 2018 entries missing this for 2018
WEATHER_AVG_WET_BULB_TEMP = 7           # 32    some 2018 entries missing this for 2018
WEATHER_PRECIP = 8                      # 38
WEATHER_AVG_STAT__ION_PRESSURE = 9        # 41
WEATHER_AVG_SEA_LEVEL_PRESSURE = 10     # 42    some 2018 entries missing this for 2018
WEATHER_AVG_WIND_SPEED = 11             # 43    some 2018 entries missing this for 2018
# WEATHER_PEAK_WIND_SPEED = 12            # 44    some 2018 entries missing this for 2018
# WEATHER_PEAK_WIND_DIRECTION = 13        # 45    some 2018 entries missing this for 2018
WEATHER_SUSTAINED_WIND_SPEED = 12       # 46
WEATHER_SUSTAINED_WIND_DIRECTION = 13   # 47

# constants used to represent the axis of a data matrix
ROWS = 0
COLUMNS = 1

NUM_LOCATIONS = 5   # Number of locations (lakes) that the data is collected from

NUM_LAKES = 75
