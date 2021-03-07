import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import GRU
from keras.layers import Dropout
from keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_percentage_error
from tensorflow.keras import backend as K
import sys
import sklearn

# get Arguments from System
DATA_SOURCE = sys.argv[1]

# Hyperparameter
KERNEL_INITIALIZER = sys.argv[2]
BATCH_SIZE = int(sys.argv[3])
DROPOUT_RATE = float(sys.argv[4])
NEURON_UNITS = int(sys.argv[5])
LEARNING_OPTIMIZER = sys.argv[6]
EPOCH = int(sys.argv[7])
EARLY_STOPPING = EarlyStopping(monitor='loss', patience=3)

def create_dataset(dataset, lookback=1):
    dataX = []
    dataY = []
    for i in range(len(dataset) - lookback - 1):
        a = dataset[i: (i + lookback), 0]
        dataX.append(a)
        dataY.append(dataset[i + lookback, 0])
    return np.array(dataX), np.array(dataY)


np.random.seed(7)

df = pd.read_csv(DATA_SOURCE, usecols=[2, 3, 4, 5, 6, 7, 8, 9, 10])
df = df.dropna()
cols = list(df)[0:9]

dataset = df[cols].astype(str)
for i in cols:
    for j in range(0, len(dataset)):
        dataset[i][j] = dataset[i][j].replace(',', '')

print(dataset)
dataset = dataset.astype(float)
dataset = dataset.values

# data scaling, minMaxScaler vs StandardScaler
scaler = MinMaxScaler(feature_range=(0, 1))
# scaler = StandardScaler()
dataset = scaler.fit_transform(dataset[:, 0:1])

train_dataset = dataset[:int(len(dataset) * 0.7), :]
test_dataset = dataset[int(len(dataset) * 0.7):, :]

lookback = 1
trainX, trainY = create_dataset(train_dataset, lookback)
testX, testY = create_dataset(test_dataset, lookback)

trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

# Model Building
model = Sequential()
# LSTM 1st Layer
model.add(GRU(units=NEURON_UNITS, return_sequences=True, input_shape=(1, lookback)))

# LSTM 2nd Layer
model.add(GRU(units=NEURON_UNITS, return_sequences=False))

# Dense Layer
model.add(Dense(1, kernel_initializer=KERNEL_INITIALIZER))

# Dropout Layer
model.add(Dropout(DROPOUT_RATE))

# tf.keras.backend.set_epsilon
K.set_epsilon(1)

# Model Compile
model.compile(
    loss='mean_squared_error',
    optimizer=LEARNING_OPTIMIZER,
    metrics=[
        'MeanSquaredError',
        'RootMeanSquaredError',
    ]
)

# Model Fit
model.fit(
    trainX,
    trainY,
    epochs=EPOCH,
    batch_size=BATCH_SIZE,
    verbose=2,
    callbacks=[EARLY_STOPPING]
)

# make predictions
trainPredict = model.predict(trainX)
testPredict = model.predict(testX)
# invert predictions
trainPredict = scaler.inverse_transform(trainPredict)
trainY = scaler.inverse_transform([trainY])
testPredict = scaler.inverse_transform(testPredict)
testY = scaler.inverse_transform([testY])

# calculate mean squared error
trainMSE = mean_squared_error(trainY[0], trainPredict[:])
testMSE = mean_squared_error(testY[0], testPredict[:, 0])

# calculate root mean squared error
trainRMSE = math.sqrt(mean_squared_error(trainY[0], trainPredict[:, 0]))
testRMSE = math.sqrt(mean_squared_error(testY[0], testPredict[:, 0]))

# calculate MAPE
testMAPE = mean_absolute_percentage_error(testY[0], testPredict[:, 0])

# print('Train Score: %.2f MSE' % (trainMSE))
# print('Train Score: %.2f RMSE' % (trainRMSE))
print('Test Score: %.2f MSE' % (testMSE))
print('Test Score: %.2f RMSE' % (testRMSE))
print('Test Score: %.2f MAPE' % (testMAPE * 100))

# scoreY = np.reshape(testY, (testY.shape[1], 1, testY.shape[0]))
# score = model.evaluate(
#     testX,
#     scoreY,
#     batch_size=BATCH_SIZE,
#     verbose=0
# )
# print('Predictions and Actual Gold Prices - LSTM (All Time / 20 Years):')
# print('MSE ' + str(score[1]))
# print('RMSE ' + str(score[2]))

trainPredictPlot = np.empty_like(dataset)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[lookback:len(trainPredict) + lookback, :] = trainPredict
# shift test predictions for plotting
testPredictPlot = np.empty_like(dataset)
testPredictPlot[:, :] = np.nan
testPredictPlot[len(trainPredict) + (lookback * 2) + 1:len(dataset) - 1, :] = testPredict
# plot baseline and predictions
plt.plot(scaler.inverse_transform(dataset), label="Actual Price")
plt.plot(trainPredictPlot, label="Train Prediction")
plt.plot(testPredictPlot, label="Test Prediction")

# for 3 years
# yticksvalue = np.arange(14000000, 30000000, 3000000)
# ytickslabel = [
#     "14,000,000", "17,000,000", "20,000,000", "23,000,000",
#     "26,000,000", "29,000,000"
# ]

# for 20 years
yticksvalue = np.arange(15000000, 31000000, 3000000)
ytickslabel = [
    "15,000,000", "18,000,000", "21,000,000",
    "24,000,000", "27,000,000", "30,000,000"
]
plt.yticks(yticksvalue, ytickslabel)

plt.xlabel("Days")
plt.ylabel("Gold Price")
plt.legend()
plt.show()