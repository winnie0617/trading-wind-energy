import pandas as pd
from preprocess import interpolate

# Converts to pandas dataframe and interpolate
df = pd.read_csv('DataWithNormalTime.csv', header=0, names=['Time', 'Energy Prooduction (kWh)'])
energy_production = interpolate(df)
energy_production.to_csv('energy-interpolated.csv', index_label='Time')