# Manage Environment
import os
import pandas as pd
import matplotlib.pyplot as plt

# Set 'bmh' style for plot
plt.style.use('bmh')


# Function that load data in gzip format
# and convert them in a single dataframe
def gz_to_df(path_list, station_number):

    df_clean = pd.DataFrame()

    # Loup through the paths
    for gz_file in path_list:

        df = pd.read_csv(gz_file, sep=';',
                         parse_dates=['date'],
                         compression='gzip', header=0)

        # Choose a station number
        df = df[df['numer_sta'] == station_number]

        # Keep only certain features
        df = df[['numer_sta', 'date', 'pmer', 'cod_tend', 'dd',
                 'ff', 't', 'td', 'u', 'pres', 'niv_bar',
                 'rafper', 'ht_neige', 'rr1', 'rr3']]

        df.replace('mq', float('nan'), inplace=True)

        # Remove feature with too many missing values
        del df["ht_neige"]
        del df["niv_bar"]

        # Fill missing values with the last valid non-null value
        df.fillna(method="ffill", inplace=True)

        # Change feature type
        df['pmer'] = df['pmer'].astype(int)
        df['cod_tend'] = df['cod_tend'].astype(int)
        df['dd'] = df['dd'].astype(int)
        df['ff'] = df['ff'].astype(float)
        df['t'] = df['t'].astype(float)
        df['td'] = df['td'].astype(float)
        df['u'] = df['u'].astype(int)
        df['pres'] = df['pres'].astype(int)
        df['rafper'] = df['rafper'].astype(float)
        df['rr1'] = df['rr1'].astype(float)
        df['rr3'] = df['rr3'].astype(float)

        # Convert kelvin to celsius
        df['temp_celsius'] = df['t'] - 273.15

        # Remove unnecessary variables for prediction
        del df["numer_sta"]
        del df["t"]

        # Concatenate dataframe
        df_clean = pd.concat([df_clean, df], axis=0)

    return df_clean


# url to load data from
url_prefix = 'https://donneespubliques.meteofrance.fr/donnees_libres/Txt/Synop/Archive/'
# For more information: https://donneespubliques.meteofrance.fr/?fond=produit&id_produit=90&id_rubrique=32

# Create lists of date to load data
list_2021_2022_1_9 = [url_prefix + f'synop.202{annee}0{mois}.csv.gz'
                      for annee in range(1, 3) for mois in range(1, 10)]

list_2021_2022_10_12 = [url_prefix + f'synop.202{annee}{mois}.csv.gz'
                        for annee in range(1, 3) for mois in range(10, 13)]

list_2023_1_7 = [url_prefix + f'synop.20230{mois}.csv.gz'
                 for mois in range(1, 8)]

list_2021_2023 = list_2021_2022_1_9 + list_2021_2022_10_12 + list_2023_1_7
list_2021_2023 = sorted(list_2021_2023)

# Load the data
# df_weather = gz_to_df(list_2021_2023, station_number=7005)
# print(df_weather.head())

save_to_csv = os.path.join('data', 'temp_data.csv')
# df_weather.to_csv(save_to_csv, index=False)

# Read the data
df = pd.read_csv(save_to_csv)

# Process the data
df['date'] = pd.to_datetime(df['date'])
df = df[['date', 'temp_celsius']]
df['date'] = df['date'].dt.date
data = df.groupby('date', as_index=False).mean()
# data = data[data.date <= pd.to_datetime('2023-01-18')]
data.set_index(['date'], inplace=True)
# print(data.tail())

# Plot the temperature evolution
data[['temp_celsius']].plot(color='#E74C3C', xlabel='Date',
                            ylabel='Temperature (Celsius)',
                            title='Temperature evolution',
                            legend=False)
plt.axvline(x=pd.to_datetime('2023-01-18'), color='black', ls='--')
# plt.xlim(right=pd.to_datetime('2023-07-23'))

plt.show()
