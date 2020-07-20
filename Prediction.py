from keras.models import load_model
import pandas as pd
from datetime import datetime
import webbrowser

datetime.utcnow()

model = load_model('trading_model.h5')

def scale_data(data):
    scaler = MinMaxScaler()
    scaler.fit(data)
    return scaler


df_energy = pd.read_csv('energy-interpolated.csv')
energys = df_energy['Energy Production (kWh)'].to_numpy().reshape(-1, 1)
energys_scaler = scale_data(energys)

df_speed = pd.read_csv('average-wind-speed.csv')
speeds = df_speed['Average Speed (m/s)'].to_numpy().reshape(-1, 1)
speeds_scaler = scale_data(speeds)

df_direction = pd.read_csv('average-wind-direction.csv')
directions = df_direction['Direction (deg N)'].to_numpy().reshape(-1, 1)
directions_scaler = scale_data(directions)



while 1!=2:
    energyData = energys_scaler.transform()
    speedData = speeds_scaler.transform()
    directionData = directions_scaler.transform()
    value = model.predict()
    webbrowser.open("http://3.1.52.222/submit/pred?pwd=7351140636&value="+value)
'''