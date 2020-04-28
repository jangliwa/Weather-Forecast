import numpy as np
import pandas as pd
import tensorflow as tf
import random
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.layers import Input, Dense, Activation
from MyLibrary import plot_prediction, slice_data, random_mini_batches

# Load data
data = pd.read_csv('./JejkowiceWeatherData.csv', header=0, names=None, sep=';')
print(data[:][:5])

# Normalize data
data_raw = data[['PRECTOT', 'WS2M', 'RH2M', 'T2MDEW', 'T2M', 'ALLSKY', 'PS']]
data_norm = (data_raw - np.mean(data_raw)) / np.std(data_raw)
data[['PRECTOT', 'WS2M', 'RH2M', 'T2MDEW', 'T2M', 'ALLSKY', 'PS']] = data_norm
print(data[:][:5])


# # Plot data to see
# for i in range(4, 11):
#     plt.figure()
#     plt.plot(data.index, data.values[:, i])
#     plt.title(data.columns[i])
#     plt.xlim(0, 7316)
#     plt.show()


# Slice data for sets
WINDOW_SIZE = 20
TARGET_SIZE = 3
PARAM_SIZE = 5
X, Y = slice_data(data, 0, WINDOW_SIZE, TARGET_SIZE, PARAM_SIZE)

# Slice data for training, validation and test set
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.07, random_state=17)
X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.075, random_state=13)
T_test = X_test[:, :, -1:]
T_val = X_val[:, :, -1:]
X_train = X_train[:, :, 0:-1]
X_test = X_test[:, :, 0:-1]
X_val = X_val[:, :, 0:-1]


# See how data is sliced
idx = 120
plt.figure()
plt.plot(range(-WINDOW_SIZE, 0), data['T2M'][idx - WINDOW_SIZE:idx])
for i in range(0, TARGET_SIZE):
    plt.scatter(i, Y_train[idx][i], c='red', marker='x')
plt.xlabel('day')
plt.ylabel('temperature')
plt.grid()
plt.show()


# Weather forecast model
def TemperatureModel(input_shape):
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.LSTM(8, input_shape=input_shape))
    model.add(tf.keras.layers.Dense(TARGET_SIZE))

    return model


temperatureModel = TemperatureModel(X_train.shape[1:])
temperatureModel.compile(optimizer='adam', loss='mean_absolute_error')

# RNN LSTM model train
EPOCHS = 50
BATCH_SIZE = 32
history = temperatureModel.fit(x=X_train, y=Y_train, epochs=EPOCHS, batch_size=BATCH_SIZE)

# Predict model
preds_train = temperatureModel.predict(X_train)
preds_val = temperatureModel.predict(X_val)
preds_test = temperatureModel.predict(X_test)

# Unnorm data outputs
T2M_std = np.mean(data_raw['T2M'])
T2M_mean = np.std(data_raw['T2M'])

preds_train_unorm = preds_train * T2M_std + T2M_mean
preds_val_unorm = preds_val * T2M_std + T2M_mean
preds_test_unorm = preds_test * T2M_std + T2M_mean

Y_train_unorm = Y_train * T2M_std + T2M_mean
Y_val_unorm = Y_val * T2M_std + T2M_mean
Y_test_unorm = Y_test * T2M_std + T2M_mean

# Compute errors
error_train = np.mean(abs(Y_train_unorm - preds_train_unorm))
error_val = np.mean(abs(Y_val_unorm - preds_val_unorm))
error_test = np.mean(abs(Y_test_unorm - preds_test_unorm))

print("Train Error = " + str(error_train))
print("Val Error = " + str(error_val))
print("Test Error = " + str(error_test))

y_pred = temperatureModel.predict(X_test)

plot_prediction(data, T_test, Y_test, y_pred, T2M_std, T2M_mean, 5)


# np.where(np.max(abs(y_pred_unorm - Y_test_unorm))==abs(y_pred_unorm - Y_test_unorm))
# idx = 121 - największy błąd

#### 20 -> 3

# ALL WITHOUT T2M AND T2MDEW
# Train Error = 2.9377259103656774
# Val Error = 2.876724142785318
# Test Error = 3.02865237292544

# ALL WITHOUT T2M
# Train Error = 2.1891543051039966
# Val Error = 2.1214739990408376
# Test Error = 2.2250853060558065

# ALL
# Train Error = 2.110606648331005
# Val Error = 2.0283972503021332
# Test Error = 2.138190759291433

# ONLY T2M
# Train Error = 2.2997683991225757
# Val Error = 2.141032513069617
# Test Error = 2.3165256818115547

# T2M and T2MDEW
# Train Error = 2.310273363008425
# Val Error = 2.146403584909995
# Test Error = 2.3528088735053423
