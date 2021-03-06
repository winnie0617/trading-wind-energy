from keras.models import load_model
import pandas as pd
from datetime import datetime
import webbrowser
import urllib.request
from numpy import array
import requests
import csv
import pandas as pd
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import schedule
import time
from preprocess import get_average_speed, get_average_direction, get_interpolated_energy
import numpy as np


model = load_model('finalTradingModel.h5')
csv_list = ['angerville-1.csv', 'angerville-1-b.csv', 'angerville-2.csv', 'angerville-2-b.csv', 'arville.csv', 'arville-b.csv', 'boissy-la-riviere.csv', 'boissy-la-riviere-b.csv',
            'guitrancourt.csv', 'guitrancourt-b.csv', 'lieusaint.csv', 'lieusaint-b.csv', 'lvs-pussay.csv', 'lvs-pussay-b.csv', 'parc-du-gatinais.csv', 'parc-du-gatinais-b.csv']
time_step = 24

# update all the datasets
def update_data(list):
    for csv in list:
        url = 'https://ai4impact.org/P003/historical/' + csv
        r = requests.get(url, allow_redirects=True)
        open('AppendixData/'+csv, 'wb').write(r.content)
        print('Dataset ' + csv + ' updated')
    url = 'https://ai4impact.org/P003/historical/' + "energy-ile-de-france.csv"
    r = requests.get(url, allow_redirects=True)
    open("energy-ile-de-france.csv", 'wb').write(r.content)

# provide scalers for respective
def scaler_data(data):
    scaler = MinMaxScaler()
    scaler.fit(data)
    return scaler


def scale_data(data):
    scaler = MinMaxScaler()
    scaler.fit(data)
    return scaler.transform(data)


df_energy = pd.read_csv('energy-interpolated.csv')
energys = df_energy['Energy Production (kWh)'].to_numpy().reshape(-1, 1)
energys_scaler = scaler_data(energys)

df_speed = pd.read_csv('average-wind-speed.csv')
speeds = df_speed['Average Speed (m/s)'].to_numpy().reshape(-1, 1)
speeds_scaler = scaler_data(speeds)

df_direction = pd.read_csv('average-wind-direction.csv')
directions = df_direction['Average Direction (deg N)'].to_numpy(
).reshape(-1, 1)
directions_scaler = scaler_data(directions)


def convert(file, scaler, name):
    list = []
    for i in range(time_step):
        if name == "min":
            if i == 0:
                list.append(np.amin(file[-time_step-i:]))
            else:
                list.append(np.amin(file[-time_step-i:-i]))
        if name == "max":
            if i == 0:
                list.append(np.amax(file[-time_step-i:]))
            else:
                list.append(np.amax(file[-time_step-i:-i]))
        if name == "difference":
            list.append(file[-i-1]-file[-2-i])
        if name == "mean":
            if i == 0:
                list.append(np.average(file[-time_step-i:]))
            else:
                list.append(np.average(file[-time_step-i:-i]))
        else:
            list.append(file[-1-i])
    list = np.reshape(list, (-1, 1))
    if name == "min" or name == "max" or name == "difference" or name == "mean":
        Data = scale_data(list)
    else:
        Data = scaler.transform(list)
    return Data


def predict():

    df_energy = pd.read_csv('energy-interpolated.csv')
    energys = df_energy['Energy Production (kWh)'].to_numpy().reshape(-1, 1)
    df_speed = pd.read_csv('average-wind-speed.csv')
    speeds = df_speed['Average Speed (m/s)'].to_numpy().reshape(-1, 1)
    df_direction = pd.read_csv('average-wind-direction.csv')
    directions = df_direction['Average Direction (deg N)'].to_numpy(
    ).reshape(-1, 1)

    # return scaled data
    energy_data = convert(energys, energys_scaler, "energys")
    speed_data = convert(speeds, speeds_scaler, "speeds")
    direction_data = convert(directions, directions_scaler, "directions")
    min_energy = convert(energys, 0, "min")
    max_energy = convert(energys, 0, "max")
    difference_energy = convert(energys, 0, "difference")
    mean_energy = convert(energys, 0, "mean")

    # combine features
    NUM_FEATURES = 7
    y = 0
    x = np.empty((time_step, NUM_FEATURES))
    for i in reversed(range(time_step)):
        x[y] = np.concatenate((energy_data[i], max_energy[i], min_energy[i], difference_energy[i], mean_energy[i],
                               speed_data[i], direction_data[i]))
        y += 1

    x = array(x)
    x = x.reshape((1, time_step, NUM_FEATURES))
    value = model.predict(x)
    value = energys_scaler.inverse_transform(value)
    print('Prediction is ' + str(value[0][0]))
    webbrowser.open(
        "http://3.1.52.222/submit/pred?pwd=7351140636&value="+str(value[0][0]))

    with open('result.csv', 'a',) as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([datetime.now(), value])
# Put all steps together


def generate_prediction():
    print(datetime.now())
    update_data(csv_list)
    get_average_speed(csv_list)
    get_average_direction(csv_list)
    get_interpolated_energy()
    predict()



schedule.every().hour.at(':50').do(generate_prediction)

while True:
    schedule.run_pending()
    time.sleep(1)
