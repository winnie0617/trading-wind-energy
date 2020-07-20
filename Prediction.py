# from keras.models import load_model
# import pandas as pd
# from datetime import datetime
# import webbrowser
import urllib.request
import requests
import schedule
import time
from datetime import datetime
from preprocess import get_average_speed, get_average_direction

'''
datetime.utcnow()

model = load_model('trading_model.h5')
'''

csv_list = ['angerville-1.csv', 'angerville-1-b.csv', 'angerville-2.csv', 'angerville-2-b.csv', 'arville.csv', 'arville-b.csv', 'boissy-la-riviere.csv', 'boissy-la-riviere-b.csv',
            'guitrancourt.csv', 'guitrancourt-b.csv', 'lieusaint.csv', 'lieusaint-b.csv', 'lvs-pussay.csv', 'lvs-pussay-b.csv', 'parc-du-gatinais.csv', 'parc-du-gatinais-b.csv']


def update_data(list):
    for csv in list:
        url = 'https://ai4impact.org/P003/historical/'+csv
        r = requests.get(url, allow_redirects=True)
        open('AppendixData/'+csv, 'wb').write(r.content)
        print('Dataset ' + csv + ' updated')






'''
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

def predict():
    print(datetime.now())
    update_data(csv_list)
    get_average_speed(csv_list)
    get_average_direction(csv_list)

schedule.every(10).seconds.do(predict)

while True:
    schedule.run_pending()
    time.sleep(1)