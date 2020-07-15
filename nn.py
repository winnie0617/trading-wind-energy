import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras.layers import Dropout
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import GaussianNoise
from keras.layers import Dense
from keras import optimizers, metrics
from keras.wrappers.scikit_learn import KerasRegressor
from keras.callbacks import EarlyStopping
from keras.layers import LSTM
import csv
from preprocess import interpolate


X_energyDataWithWindow = []
Y_energyDataWithWindow = []
# Variables
WINDOW_SIZE = 6
df = pd.read_csv('average-wind-speed.csv')
speeds = df['Average Speed (m/s)']


def convertData(window_size):
    with open('/Users/issac/Documents/GitHub/trading-wind-energy/energy-interpolated.csv') as csv_file:
        energy_data = []
        csv_reader = csv.reader(csv_file, delimiter=',')
        next(csv_reader, None)
        for row in csv_reader:
            energy_data.append(row[1])
        line_count = 0
        for b in energy_data:
            if line_count-(window_size+18) >= 0:
                y = []
                y.append(energy_data[line_count-1])
                for x in range(window_size):
                    y.append(energy_data[line_count-x-18-1])
                for i in range(window_size):
                    y.append(speeds[line_count-i-18-1])
                X_energyDataWithWindow.append(y[1:window_size*2+1])
                Y_energyDataWithWindow.append(y[0])
            line_count += 1

convertData(WINDOW_SIZE)
#print(X_energyDataWithWindow[1])
print(X_energyDataWithWindow[100])
print(Y_energyDataWithWindow[100])

# Get wind speed data







# Split the data into input and output
x = X_energyDataWithWindow
y = Y_energyDataWithWindow
y = np.reshape(y, (-1, 1))
# Normalization
scaler_x = MinMaxScaler()
scaler_y = MinMaxScaler()
print(scaler_x.fit(x))
xscale = scaler_x.transform(x)
print(scaler_y.fit(y))
yscale = scaler_y.transform(y)


# Split the data into train and test
X_train, X_test, y_train, y_test = train_test_split(
    xscale, yscale, test_size=0.2, random_state=0)

n_features = 1
#X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
#X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

# Build model
model = Sequential()
#model.add(LSTM(48,  activation='tanh', input_shape=(WINDOW_SIZE*2, 1), return_sequences=True))
#model.add(Dense(48, activation='relu',input_dim=96))
##model.add(Dense(32, activation='relu'))
#model.add(LSTM(24, activation='tanh'))
#model.add(Dropout(0.01))
#model.add(Dense(1, activation='linear'))
#model.summary()

model.add(Dense(WINDOW_SIZE, activation='relu',input_dim=WINDOW_SIZE*2))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(16, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(8, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(4, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(1, activation='linear'))
model.summary()
opt = optimizers.Adam(learning_rate=0.001)
model.compile(loss='mean_squared_error', optimizer=opt)
es = EarlyStopping(monitor='val_loss', patience=2)

# Train model
history = model.fit(X_train, y_train, epochs=500,
                    validation_split=0.2, batch_size=32)

# Plot graphs regarding the results
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='validation')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.title('Train and Validation Loss')
plt.legend()
plt.show()

# Evaluate the model on test data
print('Evaluate on test data')
results = model.evaluate(X_test, y_test, batch_size=32)
print('Test loss: ', results)


# Predict
print('Generating Predictions')
predictions_array = model.predict(
    X_test, batch_size=100, callbacks=[es])

# Plot predictions vs actuals
plt.plot(predictions_array[700:1000], label='predictions')
plt.plot(y_test[700:1000], label='actuals')
plt.legend()
plt.title('Predictions vs Actuals - First 1000')
plt.show()


'''
#Test particular prediction
x=1210
print(X_energyDataWithWindow[x])
data=scaler_x.transform([X_energyDataWithWindow[x]])
#data=data.reshape(1,WINDOW_SIZE*2, 1)
datay=model.predict(data)
print(scaler_y.inverse_transform(datay))
print(Y_energyDataWithWindow[x])
'''