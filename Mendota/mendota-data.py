import numpy as np
from sklearn.preprocessing import StandardScaler
import pandas as pd
import torch
import torch.utils.data as data
from torchvision import datasets, transforms

class mendota_data:

    def __get_data:
        data_path = '../Data/'

        filenames = [
        #     'CM2014_edit.csv',
            'CM2015_edit.csv',
            'CM2016_edit.csv',
            'CM2017_edit.csv',
            'CM2018_edit.csv',
            'mdcp.csv',
            'major_ion.csv',
            'Weather_Data.csv'
        ]

        # cla_2014 = pd.read_csv(data_path + filenames[0], low_memory=False)
        cla_2015_raw = pd.read_csv(data_path + filenames[0], low_memory=False)
        cla_2016_raw = pd.read_csv(data_path + filenames[1], low_memory=False)
        cla_2017_raw = pd.read_csv(data_path + filenames[2], low_memory=False)
        cla_2018_raw = pd.read_csv(data_path + filenames[3], low_memory=False)
        mdcp_raw = pd.read_csv(data_path + filenames[4], low_memory=False)    # Mendota buoy
        weather_raw = pd.read_csv(data_path + filenames[6], error_bad_lines=False, low_memory=False)

        ########### Clean Data ###########
        ########### CLA Data ###########

        keep15 = [     # features to keep for years 2015-2017
            'correct_timestamp',
            'collectionSiteId',
            'lake',
            'algalBloom',
            'algalBloomSheen',
            'turbidity',
        #     'waterTemp',
        #     'waveIntensity',
            'lat',
            'long'
        ]

        keep18 = [    # features to keep for 2018
            'sample_collection_time',
            'collectionSiteId',
            'lake',
            'algalBloom',
            'algalBloomSheen',
            'turbidity',
        #     'waterTemp',
        #     'waveIntensity',
            'latitiude',
            'longitude'
        ]

        rename15 = {   # rename features for 2015-2017
            'collectionSiteId': 'site',
            'lat': 'latitude',
            'long': 'longitude',
            'correct_timestamp': 'date'
        }

        rename18 = {   # renamce features for 2018
            'collectionSiteId': 'site',
            'sample_collection_time': 'date',
            'latitiude': 'latitude'
        }

        cla_2015 = cla_2015_raw[keep15]
        cla_2016 = cla_2016_raw[keep15]
        cla_2017 = cla_2017_raw[keep15]
        cla_2018 = cla_2018_raw[keep18]

        cla_2015.rename(rename15, axis='columns', inplace=True)
        cla_2016.rename(rename15, axis='columns', inplace=True)
        cla_2017.rename(rename15, axis='columns', inplace=True)
        cla_2018.rename(rename18, axis='columns', inplace=True)

        # change data types
        numeric = [    # list of numeric features
            'algalBloom',
            'algalBloomSheen',
            'turbidity',
        #     'waterTemp',
        #     'waveIntensity',
            'latitude',
            'longitude'
        ]

        # convert data types
        features = cla_2015.columns.values

        for feat in features:
            if feat in numeric:
                cla_2015[feat] = pd.to_numeric(cla_2015[feat], errors='coerce')
                cla_2016[feat] = pd.to_numeric(cla_2016[feat], errors='coerce')
                cla_2017[feat] = pd.to_numeric(cla_2017[feat], errors='coerce')
                cla_2018[feat] = pd.to_numeric(cla_2018[feat], errors='coerce')

            if feat in ['site', 'lake']:
                cla_2015[feat] = cla_2015[feat].astype(str)
                cla_2016[feat] = cla_2016[feat].astype(str)
                cla_2017[feat] = cla_2017[feat].astype(str)
                cla_2018[feat] = cla_2018[feat].astype(str)

            if feat == 'date':
                cla_2015[feat] = pd.to_datetime(cla_2015[feat], errors='coerce')
                cla_2016[feat] = pd.to_datetime(cla_2016[feat], errors='coerce')
                cla_2017[feat] = pd.to_datetime(cla_2017[feat], errors='coerce')
                cla_2018[feat] = pd.to_datetime(cla_2018[feat], errors='coerce')

        # remove nans
        cla_2015.dropna(axis='rows', how='any', inplace=True)
        cla_2016.dropna(axis='rows', how='any', inplace=True)
        cla_2017.dropna(axis='rows', how='any', inplace=True)
        cla_2018.dropna(axis='rows', how='any', inplace=True)

        # remove any data point not on lake mendota
        cla_2015 = cla_2015[cla_2015['lake'].str.contains('Mendota')]
        cla_2016 = cla_2016[cla_2016['lake'].str.contains('Mendota')]
        cla_2017 = cla_2017[cla_2017['lake'].str.contains('Mendota')]
        cla_2018 = cla_2018[cla_2018['lake'].str.contains('Mendota')]

        # set date as index
        cla_2015.set_index('date', inplace=True)
        cla_2016.set_index('date', inplace=True)
        cla_2017.set_index('date', inplace=True)
        cla_2018.set_index('date', inplace=True)

        # sort data by dates
        cla_2015.sort_values(by='date', inplace=True)
        cla_2016.sort_values(by='date', inplace=True)
        cla_2017.sort_values(by='date', inplace=True)
        cla_2018.sort_values(by='date', inplace=True)

        # resample, ffill and bfill
        cla_2015 = cla_2015.resample('D').mean()
        cla_2015.ffill(inplace=True)
        cla_2015.bfill(inplace=True)

        for date in cla_2015.index:
            if cla_2015.loc[date, 'algalBloomSheen'] > 0:
                cla_2015.loc[date, 'algalBloomSheen'] = 1

        cla_2016 = cla_2016.resample('D').mean()
        cla_2016.ffill(inplace=True)
        cla_2016.bfill(inplace=True)

        for date in cla_2016.index:
            if cla_2016.loc[date, 'algalBloomSheen'] > 0:
                cla_2016.loc[date, 'algalBloomSheen'] = 1

        cla_2017 = cla_2017.resample('D').mean()
        cla_2017.ffill(inplace=True)
        cla_2017.bfill(inplace=True)

        for date in cla_2017.index:
            if cla_2017.loc[date, 'algalBloomSheen'] > 0:
                cla_2017.loc[date, 'algalBloomSheen'] = 1

        cla_2018 = cla_2018.resample('D').mean()
        cla_2018.ffill(inplace=True)
        cla_2018.bfill(inplace=True)

        for date in cla_2018.index:
            if cla_2018.loc[date, 'algalBloomSheen'] > 0:
                cla_2018.loc[date, 'algalBloomSheen'] = 1

        # only keep the dates of the official sampling season of each year
        # cla_2015 = cla_2015[(cla_2015.index >= '2015-5-18') & (cla_2015.index <= '2015-9-4')]
        # cla_2016 = cla_2016[(cla_2016.index >= '2016-5-25') & (cla_2016.index <= '2016-9-4')]
        # cla_2017 = cla_2017[(cla_2017.index >= '2017-5-25') & (cla_2017.index <= '2017-9-4')]
        # cla_2018 = cla_2018[(cla_2018.index >= '2018-5-25') & (cla_2018.index <= '2018-9-4')]

        ########### MDCP Data ###########

        keep_mdcp = [
            'sampledate',
            'sampletime',
            'air_temp',
            'rel_hum',
            'wind_speed',
            'wind_dir',
            'chlor',
            'phycocyanin',
            'do_raw',
            'do_sat',
            'do_wtemp',
            'pco2_ppm',
            'par',
            'par_below'
        ]

        mdcp = mdcp_raw[keep_mdcp]
        mdcp.ffill(inplace=True)
        mdcp.bfill(inplace=True)

        mdcp['date'] = mdcp['sampledate'] + ' ' + mdcp['sampletime']
        mdcp['date'] = pd.to_datetime(mdcp['date'], errors='coerce')
        mdcp.dropna(axis='rows', how='any', inplace=True)

        mdcp = mdcp[[
            'date',
            'air_temp',
            'rel_hum',
            'wind_speed',
            'wind_dir',
            'chlor',
            'phycocyanin',
            'do_raw',
            'do_sat',
            'do_wtemp',
            'pco2_ppm',
            'par',
            'par_below'
        ]]
        mdcp.set_index('date', inplace=True)

        mdcp = mdcp.resample('D').mean()
        mdcp.ffill(inplace=True)
        mdcp.bfill(inplace=True)

        ########### Weather Data ###########

        keep_weather = [
            'DATE',
            'REPORTTPYE',
            'DAILYMaximumDryBulbTemp',
            'DAILYMinimumDryBulbTemp',
            'DAILYAverageDryBulbTemp',
            'DAILYDeptFromNormalAverageTemp',
            'DAILYAverageDewPointTemp',
            'DAILYAverageWetBulbTemp',
            'DAILYPrecip',
            'DAILYAverageStationPressure',
            'DAILYAverageSeaLevelPressure'
        ]

        weather = weather_raw[keep_weather]
        # weather['REPORTTPYE'].dropna(axis='rows', how='any', inplace=True)
        weather = weather.iloc[:-1]  # remove last entry since it has NaN in REPORTTPYE

        weather = weather[weather['REPORTTPYE'].str.contains('SOD')]    # only keep summary of day (SOD) info
        weather = weather.drop(['REPORTTPYE'], axis='columns')
        weather['DATE'] = pd.to_datetime(weather['DATE'], errors='coerce')

        weather.set_index('DATE', inplace=True)
        weather = weather.resample('D').ffill()
        weather.ffill(inplace=True)
        weather.bfill(inplace=True)

        # Join CLA, MDCP, and Weather Data

        # Append CLA data
        cla = cla_2015.append(cla_2016)
        cla = cla.append(cla_2017)
        cla = cla.append(cla_2018)

        # Insert MDCP data
        data = cla.join(mdcp, how='inner')

        # Insert weather data
        data = data.join(weather, how='inner')

        # sine/cosine transformation of month of year and wind direction
        data['cos_month'] = np.cos(2 * np.pi * (data.index.month.values / 12))
        data['sin_month'] = np.sin(2 * np.pi * (data.index.month.values / 12))

        data['cos_wind_dir'] = np.cos(2 * np.pi * (data['wind_dir'] / 12))
        data['sin_wind_dir'] = np.sin(2 * np.pi * (data['wind_dir'] / 12))
        data = data.drop(['wind_dir'], axis='columns')

        # Replace 'T' and 's' in 'DAILYPrecip' column
        for date in data.index:
            if 'T' in data.loc[date, 'DAILYPrecip']:
                data.loc[date, 'DAILYPrecip'] = 0.01
            elif 's' in data.loc[date, 'DAILYPrecip']:
                data.loc[date, 'DAILYPrecip'] = 0

        # Make every feature numeric
        for col in data.columns.values:
            if type(data[col].values[0]) != np.float64:
                data[col] = pd.to_numeric(data[col], errors='coerce')

        # create indicator features for whether there was rain or a bloom one day ago, or within three days or a week ago
        precip = (data['DAILYPrecip'] > 0).astype(int)   # convert precipitation to boolean values
        # data['DAILYPrecip_one_day'] = precip.shift(1)
        # data['DAILYPrecip_three_day'] = precip.rolling(window=3, min_periods=1).sum()    # NOTE THAT THIS IS DEPENDENT ON CURRENT VALUE
        # data['DAILYPrecip_one_week'] = precip.rolling(window=7, min_periods=1).sum()

        # data['algalBloomSheen_one_day'] = data['algalBloomSheen'].shift(1)
        # data['algalBloomSheen_three_day'] = data[['algalBloomSheen']].shift(1).rolling(3).sum()
        # data['algalBloomSheen_one_week'] = data[['algalBloomSheen']].shift(1).rolling(7).sum()

        # shift algalbloomsheen by -1 to predict next day algal bloom
        data['DAILYPrecip_one_day'] = precip
        data['DAILYPrecip_three_day'] = precip.rolling(window=3, min_periods=1).sum()    # NOTE THAT THIS IS DEPENDENT ON CURRENT VALUE
        data['DAILYPrecip_one_week'] = precip.rolling(window=7, min_periods=1).sum()
        data['algalBloomSheen_one_day'] = data['algalBloomSheen']
        data['algalBloomSheen_three_day'] = data[['algalBloomSheen']].rolling(3, min_periods=1).sum()
        data['algalBloomSheen_one_week'] = data[['algalBloomSheen']].rolling(7, min_periods=1).sum()
        data['algalBloomSheen'] = data['algalBloomSheen'].shift(-1)

        data.dropna(axis='rows', how='any', inplace=True)

        # display(data[['DAILYPrecip',
        #       'DAILYPrecip_one_day',
        #       'DAILYPrecip_three_day',
        #       'DAILYPrecip_one_week',
        #       'algalBloomSheen',
        #       'algalBloomSheen_one_day',
        #       'algalBloomSheen_three_day',
        #       'algalBloomSheen_one_week'
        #      ]].head(15))

        labels = data[['algalBloomSheen']]
        data = data.drop(['latitude', 'longitude', 'algalBloom', 'algalBloomSheen'], axis='columns')

        return data, labels



    def get_dataloader:
