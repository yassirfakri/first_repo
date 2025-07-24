import pandas as pd
import pdb as debugger

pdb = debugger.set_trace

# Données de prod+conso en puissance annuelle en France 2020

path_original = r"C:\Users\fatih\Downloads\rte_data.csv"
path_no_bad_lines = r"C:\Users\fatih\Downloads\rte_data_no_bad_lines.csv"

df_original = pd.read_csv(path_original, encoding='ISO-8859-1',
                          on_bad_lines='skip', sep=";")

# Comment in Last excel line deleted
df = pd.read_csv(path_no_bad_lines, encoding='ISO-8859-1', sep=";")

sub_df = df[['Date', 'Heures', 'Consommation']]
sub_df = sub_df.dropna()

# Converting dataframe to a timeseries with timestamps
sub_df['time'] = pd.to_datetime(
    df['Date'] + ' ' + df['Heures'], format='%d/%m/%Y %H:%M')
sub_df.index = sub_df['time']
sub_df = sub_df[['Consommation']]

# Consommation moyenne journalière en France en puissance
daily_conso_df = sub_df.resample('1D').mean()

# Consommation annuelle en 2020 en énergie (TWh)

total_conso = daily_conso_df.sum().iloc[0] * 24 * 1e-6
print(f'Consommation totale en 2020 en France : {round(total_conso, 2)} TWh')
pdb()
