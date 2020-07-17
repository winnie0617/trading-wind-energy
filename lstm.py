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

# Configuration
BATCH_SIZE = 32
TIMESTEPS = 24
EPOCH = 20
PATIENCE = 3

df = pd.read_csv('../trading-wind-energy/average-wind-speed.csv')
speeds = df['Average Speed (m/s)']


# def convertData(window_size):
#     with open('../trading-wind-energy/energy-interpolated.csv') as csv_file:
#         energy_data = []
#         csv_reader = csv.reader(csv_file, delimiter=',')
#         next(csv_reader, None)
#         for row in csv_reader:
#             energy_data.append(row[1])
#         line_count = 0
#         for b in energy_data:
#             if line_count-(window_size+18) >= 0:
#                 y = []
#                 y.append(energy_data[line_count-1])
#                 for x in range(window_size):
#                     y.append(energy_data[line_count-x-18-1])
#                 X_energyDataWithWindow.append(y[1:window_size*2+1])
#                 Y_energyDataWithWindow.append(y[0])
#             line_count += 1


# convertData(WINDOW_SIZE)
# # print(X_energyDataWithWindow[1])
# print(X_energyDataWithWindow[100])
# print(Y_energyDataWithWindow[100])

# Get energy production data
df_energy = pd.read_csv('../trading-wind-energy/energy-interpolated.csv')
energys = df_energy['Energy Prooduction (kWh)'].to_numpy().reshape(-1, 1)
scaler_energys = MinMaxScaler()
scaler_energys.fit(energys)
energys_scaled = scaler_energys.transform(energys)

# Get shifted energy production data
num_inputs = energys_scaled.shape[0]
energys_shifted = df_energy['Energy Prooduction (kWh)'].shift(
    periods=-18)[TIMESTEPS-1:num_inputs-18].to_numpy().reshape(-1, 1)
scaler_energys_shifted = MinMaxScaler()
scaler_energys.fit(energys_shifted)
energys_shifted_scaled = scaler_energys.transform(energys_shifted)
# print("e shifted" + str(energys_shifted[:20, 0]))
# print(energys_shifted.shape)
# print("e shifted" + str(energys_shifted[30810:]))

# Get wind speed data
df_speed = pd.read_csv('../trading-wind-energy/average-wind-speed.csv')
speeds = df_speed['Average Speed (m/s)'].to_numpy().reshape(-1, 1)
scaler_speed = MinMaxScaler()
scaler_speed.fit(speeds)
speeds_scaled = scaler_speed.transform(speeds)

# # Split the data into input and output
# x = X_energyDataWithWindow
# y = Y_energyDataWithWindow
# y = np.reshape(y, (-1, 1))

# # Normalization
# scaler_x = MinMaxScaler()
# scaler_y = MinMaxScaler()
# print(scaler_x.fit(x))
# xscale = scaler_x.transform(x)
# print(scaler_y.fit(y))
# yscale = scaler_y.transform(y)


# # Append speed data to input (for now we have more speed data than energy production. This might need to be modified later.)
# num_inputs = xscale.shape[0]
# x_with_speed = np.empty((xscale.shape[0], WINDOW_SIZE*2))
# for i in range(num_inputs):
#     x_with_speed[i] = np.append(xscale[i], speeds_scaled[i: i+WINDOW_SIZE])

# # Transform

# energys_transformed = []
# speeds_transformed = []
# for i in range(TIMESTEPS, num_inputs):
#     energys_transformed.append(energys_scaled[i-TIMESTEPS:i, 0])
#     speeds_transformed.append(speeds_scaled[i-TIMESTEPS:i, 0])
# energys_transformed, speeds_transformed = np.array(
#     energys_transformed), np.array(speeds_transformed)
# print(energys_transformed[:10])
# print("energy_transforemed" + str(energys_transformed.shape))

# Combine energy and speed
x = np.empty((num_inputs, 2))
for i in range(num_inputs):
    x[i] = np.append(energys_scaled[i], speeds_scaled[i])
print(x[:10])
print("x" + str(x.shape))

# Transform
x_transformed = []
for i in range(TIMESTEPS, num_inputs):
    x_transformed.append(x[i-TIMESTEPS:i, :])
x_transformed = np.array(x_transformed)
# print(x_transformed[:10])
# print("x_transformed" + str(x_transformed.shape))
# print(x_transformed[x_transformed.shape[0]-40:x_transformed.shape[0]-17])
print(speeds[-50:-17])
print("energy:")
print(energys[-50:-17])
print("y")
print(energys_shifted[-30:])

# Split the data into train and test
X_train, X_test, y_train, y_test = train_test_split(
    x_transformed[:-17], energys_shifted_scaled, test_size=0.2, random_state=0)

n_features = 1
# X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
# X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

# Build model
model = Sequential()

model.add(LSTM(48,  activation='tanh', input_shape=(
    TIMESTEPS, 2), return_sequences=True))
model.add(Dense(48, activation='relu', input_dim=96))
# model.add(Dense(32, activation='relu'))
model.add(LSTM(24, activation='tanh'))
model.add(Dropout(0.1))
model.add(Dense(1, activation='linear'))
model.summary()


opt = optimizers.Adam(learning_rate=0.001)
model.compile(loss='mean_squared_error', optimizer=opt)
es = EarlyStopping(monitor='val_loss', patience=PATIENCE)

# Train model
history = model.fit(X_train, y_train, epochs=EPOCH,
                    validation_split=0.2, batch_size=BATCH_SIZE)


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
    X_test, batch_size=32, callbacks=[es])

# Plot predictions vs actuals
plt.plot(predictions_array[700:1000], label='predictions')
plt.plot(y_test[700:1000], label='actuals')
plt.legend()
plt.title('Predictions vs Actuals - First 1000')
plt.show()


# # Test particular prediction
# x = 1210
# print(X_energyDataWithWindow[x])
# data = scaler_x.transform([X_energyDataWithWindow[x]])
# data = data.reshape(1, WINDOW_SIZE*2, 1)
# datay = model.predict(data)
# print(scaler_y.inverse_transform(datay))
# print(Y_energyDataWithWindow[x])
