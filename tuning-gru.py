import os
import numpy as np
import pandas as pd
import wrangle as wr

from tensorflow.keras import backend as K
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import GRU
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

# Perform the Search!
from sklearn.model_selection import GridSearchCV

# Mounting point
MP = 'data'

# Importing the training set
dataset_train = pd.read_csv('data/validation.csv')
cols = list(dataset_train)[1:11]
dataset_train = dataset_train[cols].astype(str)
training_set = dataset_train.iloc[:, 1:2].values

# Feature Scaling
from sklearn.preprocessing import MinMaxScaler

sc = MinMaxScaler(feature_range=(0, 1))
training_set_scaled = sc.fit_transform(training_set)

# Creating a data structure with 60 timesteps and 1 output
X_train = []
y_train = []
for i in range(420, 432):
    X_train.append(training_set_scaled[i - 420:i, 0])
    y_train.append(training_set_scaled[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train)

# Reshaping
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))


def gold_data():
    '''Load and preprocess(cleaning) the dataset'''
    df = pd.read_csv(os.path.join(MP, 'validation.csv'))

    # then some minimal data cleanup
    df.drop("Date", axis=1, inplace=True)

    # separate to x and y
    y = df.IDR.values
    x = df.drop('IDR', axis=1).values

    return x, y


# Load the dataset
x, y = gold_data()

# Normalize every feature to mean 0, std 1
x = wr.df_rescale_meanzero(pd.DataFrame(x)).values

input_dim = x.shape[1]  # number of columns


def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true)))


# Function to create model, required for KerasRegressor
def create_model(first_neuron=64,
                 activation='linear',
                 kernel_initializer='uniform',
                 dropout_rate=0.25,
                 optimizer='Adam'):
    # Create model
    model = Sequential()

    # Adding 1st GRU layer 432, 8
    model.add(GRU(units=64, return_sequences=True, input_shape=(dataset_train.shape[1] - 1, 1)))

    # Adding 2nd GRU layer

    model.add(GRU(units=10, return_sequences=False))

    # L1
    model.add(Dense(first_neuron,
                    input_dim=input_dim,
                    kernel_initializer=kernel_initializer,
                    activation=activation))
    # Dropout
    model.add(Dropout(dropout_rate))
    # L2
    model.add(Dense(1, kernel_initializer=kernel_initializer, activation='sigmoid'))
    # Compile model
    model.compile(loss=['mean_squared_error', root_mean_squared_error, 'mean_absolute_percentage_error'],
                  optimizer=optimizer,
                  metrics=[
                      'MeanSquaredError',
                      'RootMeanSquaredError',
                      'MeanAbsolutePercentageError'
                  ])
    return model


# Create the model
model = KerasRegressor(build_fn=create_model)

# Model Design Components 32, 64, 128
first_neurons = [64]

activation = ['relu']
# You can also try 'tanh', 'sigmoid', 'hard_sigmoid', 'linear'
# 'relu', 'linear', 'sigmoid', 'tanh'

kernel_initializer = ['ones']
#     'lecun_uniform', 'zero', 'ones', 'glorot_normal', 'glorot_uniform',
#     'he_normal', 'he_uniform','uniform', 'normal', 'orthogonal', 'constant', 'random'

# 'Adam', 'SGD', 'Adadelta'
optimizer = [
    'RMSProp'
]  # 'SGD', 'RMSProp', 'Adagrad', 'Adam'

# Hyperparameters
epochs = [20]  # You can also try 20, 30, 40, etc...
# batch_size = [64, 128, 256, 512, 1024]  # You can also try 2, 4, 8, 16, 32, 64, 128 etc...
# dropout_rate = [0.0, 0.5, 0.10, 0.15, 0.20, 0.25, 0.30]  # No dropout, but you can also try 0.1, 0.2 etc...
batch_size = [1024] # 16, 32, 64, 128, 256, 512, 1024
dropout_rate = [0.2] # 0.0, 0.2, 0.3, 0.4

# Prepare the Grid
param_grid = dict(epochs=epochs,
                  batch_size=batch_size,
                  optimizer=optimizer,
                  dropout_rate=dropout_rate,
                  activation=activation,
                  kernel_initializer=kernel_initializer,
                  first_neuron=first_neurons)

grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=1, cv=3, verbose=1)
# grid_result = grid.fit(x, y)
grid_result = grid.fit(X_train, y_train, epochs=20, validation_data=(X_train, y_train))

best_parameters = grid_result.best_params_
best_accuracy = grid_result.best_score_
print('jancuk')
print(best_parameters)
print(best_accuracy)
