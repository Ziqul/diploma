from keras.models import Sequential
from keras.layers.core import Dense
from keras.layers import LeakyReLU
from keras.layers.recurrent import LSTM
from keras.optimizers import Adam
import numpy as np


look_back = 10
max_data_val = 10000
max_error_val = 0.0000001
initial_batch_size = 10
initial_epochs_amount = 10
batch_size_limit_exceeded_times = 0

model = Sequential()
model.add(LSTM(units=50,
               return_sequences=True))
model.add(LSTM(units=50,
               return_sequences=False))
model.add(Dense(1))
model.add(LeakyReLU())
model.compile(loss="mean_squared_error", optimizer=Adam(lr=0.0001))

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

raw_break_point = int(np.around(len(x) * 0.85))

break_point = raw_break_point - (raw_break_point % 100)

x_train = np.array(x[:break_point])
y_train = np.array(y[:break_point])

x_test = np.array(x[break_point:])
y_test = np.array(y[break_point:])

print('Data timestemps amount: ', len(x))
print('Train timestemps amount: ', len(x_train))
print('Test timestemps amount: ', len(x_test))

fit_attempt = 0
prev_error = np.inf
batch_size = initial_batch_size
epochs_amount = initial_epochs_amount

while True:
    fit_attempt += 1

    pre_fit_weights = model.get_weights()

    print('Current batch size: ', batch_size)

    print('Fitting...')
    model.fit(
        x_train,
        y_train,
        batch_size=batch_size,
        epochs=epochs_amount,
        shuffle=True)

    latest_error = model.evaluate(x_test, y_test, batch_size=batch_size)

    if latest_error <= max_error_val:
        print('FINISHED')
        print('Fit attempts: ', fit_attempt)

        print('Trying to predict: ', x_test[0] * max_data_val)
        print('Prediction: ',
              model.predict(np.array([x_test[0]])) * max_data_val)

        break

    print('Error: ', latest_error)

    if latest_error >= prev_error:
        model.set_weights(pre_fit_weights)

        # Error not going down - we can make batch bigger
        if batch_size < len(x_train):

            new_batch_size = \
                batch_size + \
                int(10 ** np.floor(np.log10(batch_size)))

            if new_batch_size >= len(x_train):
                new_batch_size = len(x_train)

            batch_size = new_batch_size

            print('Batch size changed')

        # Error not going down and we cannot make batch bigger
        if batch_size == len(x_train):
            batch_size_limit_exceeded_times += 1

            if epochs_amount > 1:
                epochs_amount -= 1
    else:
        if batch_size == len(x_train):
            batch_size_limit_exceeded_times -= 1

        prev_error = latest_error

    if batch_size == len(x_train):
        print(
            'batch_size_limit_exceeded_times: ',
            batch_size_limit_exceeded_times)

    if batch_size_limit_exceeded_times >= 10:
        print('BATCH SIZE EXCEEDED TOO MANY TIMES')
        print('Fit attempts: ', fit_attempt)

        print('Trying to predict: ', x_test[0] * max_data_val)
        print('Prediction: ',
              model.predict(np.array([x_test[0]])) * max_data_val)

        break
