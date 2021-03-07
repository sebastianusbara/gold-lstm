# Import modules and packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt
from datetime import datetime
from pylab import rcParams

# Feature Scaling
from sklearn.preprocessing import StandardScaler

from tensorflow.keras.callbacks \
    import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard

# Import Libraries and packages from Keras
from tensorflow.keras import backend as K
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dropout
from tensorflow.keras.optimizers import Adam

# Importing Training Set
dataset_train = pd.read_csv("data/train-1800.csv", index_col=0, delimiter=",", decimal=".")
cols = list(dataset_train)[1:11]

# Extract dates (will be used in visualization)
datelist_train = list(dataset_train['Date'])
datelist_train = [dt.datetime.strptime(date, '%b %d %Y').date()
                  for date in datelist_train]

dataset_train = dataset_train[cols].astype(str)
for i in cols:
    for j in range(0, len(dataset_train)):
        dataset_train[i][j] = dataset_train[i][j].replace(',', '')

dataset_train = dataset_train.astype(float)

# Using multiple features (predictors)
training_set = dataset_train.values
sc = StandardScaler()
training_set_scaled = sc.fit_transform(training_set)

sc_predict = StandardScaler()
sc_predict.fit_transform(training_set[:, 0:1])

# Creating a data structure with 90 timestamps and 1 output
X_train = []
y_train = []

n_future = 30  # Number of days we want top predict into the future
n_past = 1  # Number of past days we want to use to predict the future

for i in range(n_past, len(training_set_scaled) - n_future + 1):
    X_train.append(training_set_scaled[i - n_past:i, 0:dataset_train.shape[1] - 1])
    y_train.append(training_set_scaled[i + n_future - 1:i + n_future, 0])

X_train, y_train = np.array(X_train), np.array(y_train)

# Initializing the Neural Network based on LSTM
model = Sequential()

# Adding 1st LSTM layer
model.add(LSTM(units=64, return_sequences=True, input_shape=(
    n_past, dataset_train.shape[1] - 1)))

# Adding 2nd LSTM layer
model.add(LSTM(units=10, return_sequences=False))

# Adding Dropout
model.add(Dropout(0.25))

# Output layer
model.add(Dense(units=1, activation='linear'))


def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true)))


# tf.keras.backend.set_epsilon
K.set_epsilon(1)

# Compiling the Neural Network
model.compile(
    # optimizer=Adam(learning_rate=0.001),
    loss=['mean_squared_error', root_mean_squared_error, 'mean_absolute_percentage_error'],
    metrics=[
        'MeanSquaredError',
        'RootMeanSquaredError',
        'MeanAbsolutePercentageError'
    ]
)

es = EarlyStopping(monitor='val_loss', min_delta=1e-10, patience=10, verbose=1)
rlr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, verbose=1)
mcp = ModelCheckpoint(
    filepath='weights.h5',
    monitor='val_loss',
    verbose=1,
    save_best_only=True,
    save_weights_only=True
)

tb = TensorBoard('logs')

BATCH_SIZE = 256

history = model.fit(
    X_train,
    y_train,
    shuffle=True,
    epochs=30,
    callbacks=[es, rlr, mcp, tb],
    validation_split=0.2,
    verbose=1,
    batch_size=BATCH_SIZE,
    validation_data=(X_train, y_train)
)

# Generate list of sequence of days for predictions
datelist_future = pd.date_range(datelist_train[-1], periods=n_future, freq='1d').tolist()

# Convert Pandas Timestamp to Datetime object (for transformation) --> FUTURE
datelist_future_ = []
for this_timestamp in datelist_future:
    datelist_future_.append(this_timestamp.date())

# Perform predictions
predictions_future = model.predict(X_train[-n_future:])
predictions_train = model.predict(X_train[n_past:])

# Inverse the predictions to original measurements

# ---> Special function: convert <datetime.date> to <Timestamp>
def datetime_to_timestamp(x):
    return datetime.strptime(x.strftime('%Y%m%d'), '%Y%m%d')


y_pred_future = sc_predict.inverse_transform(predictions_future)
y_pred_train = sc_predict.inverse_transform(predictions_train)


PREDICTIONS_FUTURE = pd.DataFrame(y_pred_future, columns=['IDR']).set_index(
    pd.Series(datelist_future))
PREDICTION_TRAIN = pd.DataFrame(y_pred_train, columns=['IDR']).set_index(
    pd.Series(datelist_train[2 * n_past + n_future - 1:]))

# show accuracy
score = model.evaluate(X_train, y_train, batch_size=BATCH_SIZE, verbose=0)
print('Predictions and Actual Gold Prices - LSTM (5 Year / 1800 Days):')
print('MSE ' + str(score[1]))
print('RMSE ' + str(score[2]))
print('MAPE ' + str(score[3]))

# Convert <datetime.date> to <Timestamp> for PREDCITION_TRAIN
PREDICTION_TRAIN.index = PREDICTION_TRAIN.index.to_series().apply(datetime_to_timestamp)

# Set plot size
rcParams['figure.figsize'] = 14, 5

# Plot parameters
START_DATE_FOR_PLOTTING = '2001-01-05'
actualDate = pd.DataFrame(datelist_train, index=datelist_train).index

plt.plot(PREDICTIONS_FUTURE.index, PREDICTIONS_FUTURE['IDR'], color='r', label='Predicted Gold Price')
plt.plot(PREDICTION_TRAIN.loc[START_DATE_FOR_PLOTTING:].index, PREDICTION_TRAIN.loc[START_DATE_FOR_PLOTTING:]['IDR'],
         color='orange', label='Training predictions')
plt.plot(actualDate[n_future + 1:], dataset_train.loc[n_future + 1:]['IDR'],
         color='b', label='Actual Gold Price')

plt.axvline(x=min(PREDICTIONS_FUTURE.index), color='green', linewidth=2, linestyle='--')

plt.grid(which='major', color='#cccccc', alpha=0.5)

yticksvalue = np.arange(12000000, 24000000, 2000000)
ytickslabel = [12, 14, 16, 18, 20, 22]
plt.yticks(yticksvalue, ytickslabel)
plt.legend(shadow=True)
plt.title(
    'Predictions and Actual Gold Prices - LSTM (5 Year / 1800 Days)',
    family='Arial', fontsize=12
)
plt.ylabel('Gold Price (Value in Million Rupiah)', family='Arial', fontsize=10)
plt.xticks(fontsize=8)
plt.show()
