# from keras.models import load_model
# import pandas as pd
# from datetime import datetime
# import webbrowser
import urllib.request
import requests
import datetime
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import schedule
import time
from datetime import datetime
from preprocess import get_average_speed, get_average_direction

model = load_model('trading_model.h5')

csv_list = ['angerville-1.csv', 'angerville-1-b.csv', 'angerville-2.csv', 'angerville-2-b.csv', 'arville.csv', 'arville-b.csv', 'boissy-la-riviere.csv', 'boissy-la-riviere-b.csv',
            'guitrancourt.csv', 'guitrancourt-b.csv', 'lieusaint.csv', 'lieusaint-b.csv', 'lvs-pussay.csv', 'lvs-pussay-b.csv', 'parc-du-gatinais.csv', 'parc-du-gatinais-b.csv']


def update_data(list):
    for csv in list:
        url = 'https://ai4impact.org/P003/historical/'+csv
        r = requests.get(url, allow_redirects=True)
        open('AppendixData/'+csv, 'wb').write(r.content)
        print('Dataset ' + csv + ' updated')

def scale_data(data):
    scaler = MinMaxScaler()
    scaler.fit(data)
    return scaler


df_energy = pd.read_csv('energy-interpolated.csv')
energys= df_energy['Energy Production (kWh)'].to_numpy().reshape(-1, 1)
energys_scaler = scale_data(energys)

df_speed = pd.read_csv('average-wind-speed.csv')
speeds = df_speed['Average Speed (m/s)'].to_numpy().reshape(-1, 1)
speeds_scaler = scale_data(speeds)

df_direction = pd.read_csv('average-wind-direction.csv')
directions = df_direction['Direction (deg N)'].to_numpy().reshape(-1, 1)
directions_scaler = scale_data(directions)



while 1!=2:
    timeSteps =24
    df_energy = pd.read_csv('energy-interpolated.csv')
    energys= df_energy['Energy Production (kWh)'].to_numpy().reshape(-1, 1)
    df_speed = pd.read_csv('average-wind-speed.csv')
    speeds = df_speed['Average Speed (m/s)'].to_numpy().reshape(-1, 1)
    df_direction = pd.read_csv('average-wind-direction.csv')
    directions = df_direction['Direction (deg N)'].to_numpy().reshape(-1, 1)

    energyData = energys_scaler.transform()
    speedData = speeds_scaler.transform()
    directionData = directions_scaler.transform()
    timeSteps =24
    energyData=[]
    for i in range(timeSteps):
        energyData.append(energys_scaler.transform(energys[-1-i]))
    speedData =[]
    for i in range(timeSteps):
        speedData.append(speeds_scaler.transform(speeds[-1-i]))
    directionData = []
    for i in range(timeSteps):
        directionData.append(directions_scaler.transform(directions[-1-i]))

    min_energy = []
    min_energy_notscaled=[]
    for i in range(timeSteps):
        min = df_energy['Energy Production (kWh)'][-1:-1-i].min()
        min_energy_notscaled.append(min)
    min_energy_scaler = scale_data(min_energy_notscaled)
    for i in range(len(min_energy)):
        min_energy.append[min_energy_scaler.transform(min_energy_notscaledp[i])]

    max_energy = []
    max_energy_notscaled=[]
    for i in range(timeSteps):
        max = df_energy['Energy Production (kWh)'][-1:-1-i].max()
        max_energy_notscaled.append(max)
    max_energy_scaler = scale_data(max_energy_notscaled)
    for i in range(len(max_energy)):
        max_energy.append[max_energy_scaler.transform(max_energy_notscaledp[i])]

    difference_energy_notscaled=[]
    difference_energy=[]
    for i in range(timeSteps):
        difference =df_energy['Energy Production (kWh)'][-1-i] - df_energy['Energy Production (kWh)'][-i-2]))
        difference_energy_notscaled.append(difference)
    difference_energy_scaler = scale_data(difference_energy_notscaled)
    for i in range(len(difference_energy_notscaled)):
        difference_energy.append(difference_energy_notscaled[i])

    std_energy = []
    std_energy_notscaled=[]
    for i in range(timeSteps):
        std = df_energy['Energy Production (kWh)'][-1:-1-i].std()
        std_energy_notscaled.append(std)
    std_energy_scaler = scale_data(std_energy_notscaled)
    for i in range(len(std_energy)):
        std_energy.append[std_energy_scaler.transform(std_energy_notscaledp[i])]

    mean_energy_notscaled = []
    mean_energy =[]
    for i in range(timeSteps):
        mean = df_energy['Energy Production (kWh)'][-1:-1-i].mean()
        mean_energy_notscaled.append(mean)
    mean_energy_scaler = scale_data(mean_energy_notscaled)
    for i in range(len(mean_energy)):
        mean_energy.append[mean_energy_scaler.transform(mean_energy_notscaledp[i])]

    value = model.predict()
    value = energys_scaler.inverse_transform(value)
    print(value)
    webbrowser.open("http://3.1.52.222/submit/pred?pwd=7351140636&value="+value)


def predict():
    print(datetime.now())
    update_data(csv_list)
    get_average_speed(csv_list)
    get_average_direction(csv_list)

schedule.every(10).seconds.do(predict)

while True:
    schedule.run_pending()
    time.sleep(1)
