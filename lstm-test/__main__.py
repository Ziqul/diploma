from keras.models import Sequential
from keras.layers.core import Dense
from keras.layers import LeakyReLU
from keras.layers.recurrent import LSTM
from keras.optimizers import Adam
import numpy as np


look_back = 10
max_data_val = 10000
max_error_val = 0.00000001

model = Sequential()
model.add(LSTM(units=50,
               return_sequences=True))
model.add(LSTM(units=50,
               return_sequences=False))
model.add(Dense(1, activation=LeakyReLU()))
model.compile(loss="mean_squared_error", optimizer=Adam(lr=0.00001))

data = np.array(range(0, max_data_val + 1)) / max_data_val
x, y = [], []

for i, data_i in enumerate(data, look_back):
    # break if we are on pre-last element
    if i == len(data) - 1:
        break

    x_i = []

    for j in reversed(range(0, look_back)):
        x_i.append(data[i - j])

    x.append([x_i])
    y.append([data[i + 1]])

break_point = int(np.around(len(x) * 0.85))

print('break_point', break_point)

x_train = np.array(x[:break_point])
y_train = np.array(y[:break_point])

x_test = np.array(x[break_point:])
y_test = np.array(y[break_point:])


fits_amount = 0
prev_error = np.inf
batch_size = 10
batch_size_limit_exceeded = 0
epochs_amount = 10

while True:
    fits_amount += 1

    pre_fit_weights = model.get_weights()

    error = model.evaluate(x_test, y_test, batch_size=batch_size)

    print('batch_size: ', batch_size)
    print('Prediction error: ', error)

    if error <= max_error_val:
        print('FINISHED')

        print('fits_amount: ', fits_amount)

        print('Predict: ', x_test[0] * max_data_val)
        print('Prediction: ',
              model.predict(np.array([x_test[0]])) * max_data_val)

        break

    # Error is not going down and we cannot make batch bigger
    if error > prev_error and batch_size == len(x_train):
        if epochs_amount > 1:
            epochs_amount -= 1

        batch_size_limit_exceeded += 1
    elif error > prev_error and batch_size == len(x_train):
        batch_size_limit_exceeded -= 1

    if batch_size == len(x_train):
        print('batch_size_limit_exceeded: ', batch_size_limit_exceeded)

    if batch_size_limit_exceeded >= 10:
        print('CANNOT APPROACH max_error_val THAT WAS SET')

        print('fits_amount: ', fits_amount)

        print('Predict: ', x_test[0] * max_data_val)
        print('Prediction: ',
              model.predict(np.array([x_test[0]])) * max_data_val)

        break

    model.fit(
        x_train,
        y_train,
        batch_size=batch_size,
        nb_epoch=epochs_amount,
        shuffle=True)

    post_fit_error = model.evaluate(x_test, y_test, batch_size=batch_size)

    if post_fit_error > prev_error:
        model.set_weights(pre_fit_weights)

    if post_fit_error > prev_error and batch_size < len(x_train):

        new_batch_size = \
            batch_size + \
            int(10 ** np.floor(np.log10(batch_size)))

        if new_batch_size >= len(x_train):
            new_batch_size = len(x_train)

        batch_size = new_batch_size

    print('new_batch_size: ', batch_size)

    prev_error = post_fit_error
