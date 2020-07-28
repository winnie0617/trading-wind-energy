# trading-wind-energy

## Objective
Why is this important?

## Packages?
Keras

'''python
model = Sequential()
model.add(LSTM(24, input_shape=(TIMESTEPS, NUM_FEATURES), return_sequences=False, activity_regularizer=l2(0.001)))
model.add(Dropout(0.1))
model.add(Dense(12, activation='relu'))
model.add(Dense(1, activation='linear'))
'''

## Preliminary Data Analysis
There are two main datasets - Wind Energy Production and Wind Forecasts. In order to have a rough idea of the features we are going to use in our neural network model, we created several plots to gain a better understanding of the data.

### Wind Energy Production Dataset
![Wind energy production plot](Graphs/wind-energy-production.png)
As observed in the plot, there is no obvious periodicity(?) Hard to predict
![Autocorrelation](Graphs/autocorrelation.png)

### Wind Speed and Direction of 8 Wind Farms
![Wind speed vs energy](Graphs/speed_vs_energy.png)
We averaged the wind speed data from the 8 farms and plotted them against the energy production. It can be seen that there is a positive correlation between the average wind speed and the energy production. Therefore, we are going to take the average wind speed as the second input.
![Wind direction vs energy](Graphs/direction_vs_energy.png)
We also plotted the wind direction vs energy production for each wind farm, one of which is shown above. Although the relationship between the two is non linear and less obvious, we still included it as a feature as it is observed that energy production tends to be higher in two directions.

## Data Preprocessing
Speed and direction data are produced every 6 hours. Since we are required to make a prediction every hour, we linearly interpolated the datasets to time base of 1 hour. Missing data in all data sets are also linearly interpolated. Then, as mentioned above, we averaged the wind speed data across the 8 wind farms.\
In order to avoid large data points from dominating the learning, we normalized all our datasets to be between 0 and 1.

## Persistence as Benchmark
We started off by building a persistence model as our benchmark. Test loss of future models we develop will be compared to the persistence loss.
Used MSE, and got 0.09

## LSTM

## Limitations
