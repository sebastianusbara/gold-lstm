# univariate lstm example
import pandas
import tensorflow as tf
from numpy import array
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense
import data_cleaning


# split a univariate sequence into samples
def split_sequence(sequence, n_steps):
    X, y = list(), list()
    for i in range(len(sequence)):
        # find the end of this pattern
        end_ix = i + n_steps
        # check if we are beyond the sequence
        if end_ix > len(sequence) - 1:
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
        X.append(seq_x)
        y.append(seq_y)
    return array(X), array(y)


# define input sequence
raw_seq = data_cleaning.data().to_numpy()

# choose a number of time steps
n_steps = 4
# split into samples
X, y = split_sequence(raw_seq, n_steps)
# reshape from [samples, timesteps] into [samples, timesteps, features]
n_features = 1
X = X.reshape((X.shape[0], X.shape[1], n_features))

# define model
model = Sequential()
model.add(LSTM(10, activation='relu', input_shape=(n_steps, n_features)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')
# fit model
model.fit(X, y, epochs=10, verbose=1)
# demonstrate archive

# x_input = array(data_cleaning.data().to_numpy())
x_input = array([10, 20, 30, 40])

x_input = x_input.reshape((1, n_steps, n_features))
x_input = tf.cast(x_input, dtype='float32')
print(x_input)

# verbose: 0 for no logging to stdout, 1 for progress bar logging, 2 for one log line per epoch.
yhat = model.predict(x_input, verbose=1)
print(yhat)
