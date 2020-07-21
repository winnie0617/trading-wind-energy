'''
from keras.models import load_model
import pandas as pd
from datetime import datetime
import webbrowser
import urllib.request
from numpy import array
import requests
import datetime
import pandas as pd
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import schedule
import time
from datetime import datetime
from preprocess import get_average_speed, get_average_direction


model = load_model('trading_model.h5')



def update_data(list):
    for csv in list:
        url = 'https://ai4impact.org/P003/historical/'+csv
        r = requests.get(url, allow_redirects=True)
        open('AppendixData/'+csv, 'wb').write(r.content)
        print('Dataset ' + csv + ' updated')

def scaler_data(data):
    scaler = MinMaxScaler()
    scaler.fit(data)
    return scaler

def scale_data(data):
    scaler = MinMaxScaler()
    scaler.fit(data)
    return scaler.transform(data)

df_energy = pd.read_csv('energy-interpolated.csv')
energys= df_energy['Energy Production (kWh)'].to_numpy().reshape(-1, 1)
energys_scaler = scaler_data(energys)

df_speed = pd.read_csv('average-wind-speed.csv')
speeds = df_speed['Average Speed (m/s)'].to_numpy().reshape(-1, 1)
speeds_scaler = scaler_data(speeds)

df_direction = pd.read_csv('average-wind-direction.csv')
directions = df_direction['Average Direction (deg N)'].to_numpy().reshape(-1, 1)
directions_scaler = scaler_data(directions)



x=0
while x!=1:

    df_energy = pd.read_csv('energy-interpolated.csv')
    energys= df_energy['Energy Production (kWh)'].to_numpy().reshape(-1, 1)
    df_speed = pd.read_csv('average-wind-speed.csv')
    speeds = df_speed['Average Speed (m/s)'].to_numpy().reshape(-1, 1)
    df_direction = pd.read_csv('average-wind-direction.csv')
    directions = df_direction['Average Direction (deg N)'].to_numpy().reshape(-1, 1)

    timeSteps =12
    #return scaled data
    def convert(file, scaler,name):
        list=[]
        for i in range(timeSteps):
            if name== "min":
                list.append(np.amin(file[-timeSteps-i:-1-i]))
                print(list)
            if name== "max":
                list.append(np.amax(file[-timeSteps-i:-1-i]))
            if name== "difference":
                list.append(file[-i-1]-file[-2-i])
            if name == "mean":
                list.append(np.average(file[-timeSteps-i:-1-i]))
            else:
                list.append(file[-1-i])
        list = np.reshape(list,(-1,1))
        if name =="min" or name == "max" or name =="difference" or name =="mean":
            Data=scale_data(list)
        else:
            Data = scaler.transform(list)
        print(Data)
        return(Data)

    energyData = convert(energys,energys_scaler,"energys")
    speedData = convert(speeds,speeds_scaler,"speeds")
    directionData =convert(directions,directions_scaler,"directions" )
    min_energy= convert(energys,0,"min" )
    max_energy = convert(energys,0,"max" )
    difference_energy= convert(energys,0,"difference" )
    mean_energy = convert(energys,0,"mean")

    #####
    #Combine to be done tmr
    NUM_FEATURES = 7
    x = np.empty((timeSteps, NUM_FEATURES))
    for i in range(timeSteps):
        x[i] = np.concatenate((energyData[i], max_energy[i], min_energy[i], difference_energy[i], mean_energy[i],
                        speedData[i], directionData[i]))

    #####
    x=array(x)
    x=x.reshape((1,12,7))
    value = model.predict(x)
    value = energys_scaler.inverse_transform(value)
    print(value)
    x=1
    #webbrowser.open("http://3.1.52.222/submit/pred?pwd=7351140636&value="+value)
    '''


    #####
    Combine to be done tmr
    #####
    value = model.predict()
    value = energys_scaler.inverse_transform(value)
    print(value)
    webbrowser.open("http://3.1.52.222/submit/pred?pwd=7351140636&value="+value)

'''
csv_list = ['angerville-1.csv', 'angerville-1-b.csv', 'angerville-2.csv', 'angerville-2-b.csv', 'arville.csv', 'arville-b.csv', 'boissy-la-riviere.csv', 'boissy-la-riviere-b.csv',
            'guitrancourt.csv', 'guitrancourt-b.csv', 'lieusaint.csv', 'lieusaint-b.csv', 'lvs-pussay.csv', 'lvs-pussay-b.csv', 'parc-du-gatinais.csv', 'parc-du-gatinais-b.csv']

def update_data(list):
    for csv in list:
        url = 'https://ai4impact.org/P003/historical/'+csv
        r = requests.get(url, allow_redirects=True)
        open('AppendixData/'+csv, 'wb').write(r.content)
        print('Dataset ' + csv + ' updated')


def predict():
    print(datetime.now())
    update_data(csv_list)
    get_average_speed(csv_list)
    get_average_direction(csv_list)

schedule.every().hour.at(':15').do(predict)

while True:
    schedule.run_pending()
    time.sleep(1)
'''
