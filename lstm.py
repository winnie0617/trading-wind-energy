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
from keras.regularizers import l2


# Configuration
BATCH_SIZE = 32
TIMESTEPS = 24
EPOCH = 100
PATIENCE = 10
LEARNING_RATE = 0.001

# Get energy production, wind speed and wind direction data

def scale_data(data):
    scaler = MinMaxScaler()
    scaler.fit(data)
    return scaler.transform(data)


# Scale all datasets
df_energy = pd.read_csv('../trading-wind-energy/energy-interpolated.csv')
energys = df_energy['Energy Production (kWh)'].to_numpy().reshape(-1, 1)
energys_scaled = scale_data(energys)

df_speed = pd.read_csv('../trading-wind-energy/average-wind-speed.csv')
speeds = df_speed['Average Speed (m/s)'].to_numpy().reshape(-1, 1)
speeds_scaled = scale_data(speeds)

df_direction = pd.read_csv('../trading-wind-energy/average-wind-direction.csv')
directions = df_direction['Direction (deg N)'].to_numpy().reshape(-1, 1)
directions_scaled = scale_data(directions)


# Get shifted energy production data
num_inputs = speeds_scaled.shape[0]
energys_shifted = df_energy['Energy Production (kWh)'].shift(
    periods=-18)[TIMESTEPS-1:num_inputs-18].to_numpy().reshape(-1, 1)
energys_shifted_scaled = scale_data(energys_shifted)
# print("e shifted" + str(energys_shifted[:20, 0]))
# print(energys_shifted.shape)
# print("e shifted" + str(energys_shifted[30810:]))

'''
# Get all speeds
df_all_speed = pd.read_csv('../trading-wind-energy/all-wind-speed.csv')
speed = df_all_speed['Speed1 (m/s)'].to_numpy().reshape(-1, 1)
scaled1 = scale_data(speed)
speed = df_all_speed['Speed2 (m/s)'].to_numpy().reshape(-1, 1)
scaled2 = scale_data(speed)
speed = df_all_speed['Speed3 (m/s)'].to_numpy().reshape(-1, 1)
scaled3 = scale_data(speed)
speed = df_all_speed['Speed4 (m/s)'].to_numpy().reshape(-1, 1)
scaled4 = scale_data(speed)
speed = df_all_speed['Speed5 (m/s)'].to_numpy().reshape(-1, 1)
scaled5 = scale_data(speed)
speed = df_all_speed['Speed6 (m/s)'].to_numpy().reshape(-1, 1)
scaled6 = scale_data(speed)
speed = df_all_speed['Speed7 (m/s)'].to_numpy().reshape(-1, 1)
scaled7 = scale_data(speed)
speed = df_all_speed['Speed8 (m/s)'].to_numpy().reshape(-1, 1)
scaled8 = scale_data(speed)
'''

# Get max energy production in the window
max_energy = []
for i in range(num_inputs):
    if i < TIMESTEPS:
        max = df_energy['Energy Production (kWh)'][0:i+1].max()
    else:
        max = df_energy['Energy Production (kWh)'][i-TIMESTEPS:i+1].max()
    max_energy.append(max)
max_energy = np.array(max_energy).reshape(-1, 1)
max_energy_scaled = scale_data(max_energy)

# Get min energy production in the window
min_energy = []
for i in range(num_inputs):
    if i < TIMESTEPS:
        min = df_energy['Energy Production (kWh)'][0:i+1].min() 
    else:
        min = df_energy['Energy Production (kWh)'][i-TIMESTEPS:i+1].min()
    min_energy.append(min)
min_energy = np.array(min_energy).reshape(-1, 1)
min_energy_scaled = scale_data(min_energy)

# Get difference
print(df_energy['Energy Production (kWh)'].shape)
print(num_inputs)
difference = [0]
for i in range(1, num_inputs):
    difference.append(
        df_energy['Energy Production (kWh)'][i] - df_energy['Energy Production (kWh)'][i-1])
difference = np.array(difference).reshape(-1, 1)
# scaler = MinMaxScaler((-1,1))
# scaler.fit(difference)
# difference_scaled = scaler.transform(difference)
difference_scaled = scale_data(difference)
print(difference[:20])


# Get mean energy production in the window
mean_energy = []
for i in range(num_inputs):
    if i < TIMESTEPS:
        mean = df_energy['Energy Production (kWh)'][0:i+1].mean() 
    else:
        mean = df_energy['Energy Production (kWh)'][i-TIMESTEPS:i+1].mean()
    mean_energy.append(mean)
mean_energy = np.array(mean_energy).reshape(-1, 1)
mean_energy_scaled = scale_data(mean_energy)

# Get standard deviation of energy production in the window
sd_energy = []
for i in range(num_inputs):
    if i == 0:
        sd =0
    elif i < TIMESTEPS:
        sd = df_energy['Energy Production (kWh)'][0:i+1].std()
    else:
        sd = df_energy['Energy Production (kWh)'][i-TIMESTEPS:i+1].std()
    sd_energy.append(sd)
sd_energy = np.array(sd_energy).reshape(-1, 1)
sd_energy_scaled = scale_data(sd_energy)


# Combine everything
NUM_FEATURES = 7
x = np.empty((num_inputs, NUM_FEATURES))
print(max_energy_scaled.shape)
print(energys_scaled.shape)
for i in range(num_inputs):
    x[i] = np.concatenate((energys_scaled[i], max_energy_scaled[i], min_energy_scaled[i], difference_scaled[i], mean_energy_scaled[i],
                     speeds_scaled[i], directions_scaled[i]))
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
# print(speeds[-50:-17])
# print("energy:")
# print(energys[-50:-17])
# print("y")
# print(energys_shifted[-30:])

# Split the data into train and test
X_train, X_test, y_train, y_test = train_test_split(
    x_transformed[:-17], energys_shifted_scaled, test_size=0.2, shuffle=False)


# X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
# X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

# Build model
model = Sequential()

# model.add(LSTM(12, activation='tanh', input_shape=(
#     TIMESTEPS, NUM_FEATURES), return_sequences=False, activity_regularizer=l2(0.001)))
# model.add(Dropout(0.1))
model.add(LSTM(24, input_shape=(TIMESTEPS, NUM_FEATURES), return_sequences=False, activity_regularizer=l2(0.001)))
# model.add(Dropout(0.1))
# model.add(LSTM(32,return_sequences=False))
model.add(Dropout(0.1))
model.add(Dense(12, activation='relu'))
model.add(Dense(1, activation='linear'))
# model.add(LSTM(48,  activation='tanh', input_shape=(TIMESTEPS, 3), return_sequences=False))
# model.add(Dropout(0.1))

# 
# 
# model.add(LSTM(24,  activation='tanh', input_shape=(
#     TIMESTEPS, NUM_FEATURES), return_sequences=True, activity_regularizer=l2(0.001)))
# model.add(LSTM(12,return_sequences=True))
# model.add(Dropout(0.1))
# model.add(LSTM(4,return_sequences=False))
# model.add(Dropout(0.1))
# model.add(Dense(1, activation='tanh'))
# model.summary()


opt = optimizers.Adam(learning_rate=LEARNING_RATE)
model.compile(loss='mean_squared_error', optimizer=opt)
es = EarlyStopping(monitor='val_loss', patience=PATIENCE, min_delta=0.0001)

# Train model
history = model.fit(X_train, y_train, epochs=EPOCH,
                    validation_split=0.2, batch_size=BATCH_SIZE, callbacks=[es], shuffle=False)


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
results = model.evaluate(X_test, y_test, batch_size=BATCH_SIZE)
print('Test loss: ', results)


# Predict
print('Generating Predictions')
predictions_array = model.predict(X_test, batch_size=BATCH_SIZE)

# Plot predictions vs actuals
plt.plot(predictions_array, label='predictions')
plt.plot(y_test, label='actuals')
plt.legend()
plt.title('Predictions vs Actuals')
plt.show()

# Plot prediction vs actual scatter plot
plt.scatter(y_test, predictions_array)
plt.xlabel('Predicted Value')
plt.xlabel('Actual Value')
plt.show()

# # Test particular prediction
# x = 1210
# print(X_energyDataWithWindow[x])
# data = scaler_x.transform([X_energyDataWithWindow[x]])
# data = data.reshape(1, WINDOW_SIZE*2, 1)
# datay = model.predict(data)
# print(scaler_y.inverse_transform(datay))
# print(Y_energyDataWithWindow[x])
