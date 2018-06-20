from keras.models import Sequential, load_model
from keras.layers.core import Dense, Activation
from keras.layers import LeakyReLU
from keras.layers.recurrent import LSTM
from keras.optimizers import Adam
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import numpy as np
import traceback
import time
import csv
import os

look_back = 7

def main():
    max_error_val = 0.0002
    initial_batch_size = 10
    initial_epochs_amount = 10
    batch_size_limit_exceeded_times = 0

    model = Sequential()
    model.add(LSTM(units=200,
                   return_sequences=True))
    model.add(LSTM(units=200,
                   return_sequences=False))
    model.add(Dense(1))
    # model.add(Activation('sigmoid'))
    model.add(LeakyReLU())
    model.compile(loss="mean_squared_error", optimizer=Adam(lr=0.00005))

    raw_data = []

    with open('data/exchange-rate-of-australian-doll.csv') as csvfile:
        spamreader = csv.reader(csvfile)
        for row in spamreader:
            raw_data.append(float(row[0]))

    max_data_val = np.max(raw_data)

    data = np.array(raw_data) / max_data_val
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

    raw_break_point = int(np.around(len(x) * 0.75))

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
        try:
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

            latest_error = \
                model.evaluate(x_test, y_test, batch_size=batch_size)

            if latest_error <= max_error_val:
                print('FINISHED')
                print('Fit attempts: ', fit_attempt)

                finish(
                    y_train,
                    x_test, y_test,
                    max_data_val,
                    model, latest_error)

                break

            print('Error: ', latest_error)

            if latest_error >= prev_error:
                model.set_weights(pre_fit_weights)

                epochs_amount *= 2

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

            else:
                epochs_amount = initial_epochs_amount

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

                finish(
                    y_train,
                    x_test, y_test,
                    max_data_val,
                    model, prev_error)

                break
        except:
            print('\n')
            print('EMERGENT ERROR')
            traceback.print_exc()
            print('Fit attempts: ', fit_attempt)

            finish(
                y_train,
                x_test, y_test,
                max_data_val,
                model, prev_error)

            break


def finish(train_data, x_test_data, y_test_data, max_data_val, model, error):
    filename = str(time.time()) + '_' + str(error)

    save(filename, model, error)
    plot(filename, train_data, x_test_data, y_test_data, max_data_val, model)
    plot2(filename, train_data, x_test_data, y_test_data, max_data_val, model)


def plot(filename, train_data, x_test_data, y_test_data, max_data_val, model):
    plt_train_data = \
        train_data.reshape((train_data.size,)) * max_data_val

    plt_test_data = \
        y_test_data.reshape((y_test_data.size,)) * max_data_val

    test_predict = model.predict(x_test_data)

    plt_test_predict = \
        test_predict.reshape((test_predict.size,)) * max_data_val

    plt.plot(
        range(
            0, len(plt_train_data)),
        plt_train_data, 'b-')

    plt.plot(
        range(
            len(plt_train_data), len(plt_train_data) + len(plt_test_data)),
        plt_test_data, 'b-')

    plt.plot(
        range(
            len(plt_train_data),
            len(plt_train_data) + len(plt_test_predict)),
        plt_test_predict, 'r--')

    dirname = os.path.join('.', 'models', filename)

    if not os.path.exists(dirname):
        os.makedirs(dirname)

    plt.savefig(os.path.join(dirname, 'image1.png'))

    plt.show()


def plot2(filename, train_data, x_test_data, y_test_data, max_data_val, model):
    plt_train_data = \
        train_data.reshape((train_data.size,)) * max_data_val

    plt_test_data = \
        y_test_data.reshape((y_test_data.size,)) * max_data_val

    plt.plot(
        range(
            0, len(plt_train_data)),
        plt_train_data, 'b-')

    plt.plot(
        range(
            len(plt_train_data), len(plt_train_data) + len(plt_test_data)),
        plt_test_data, 'b-')

    test_data_start = x_test_data[0]

    test_predict = []
    for i in range(0, 10):

        one_predict = model.predict(np.array([test_data_start]))

        test_predict.append(one_predict[0][0])

        test_data_start = np.hstack([test_data_start, one_predict])

        test_data_start = np.array([test_data_start[0, :][1:]])

    plt_test_predict = np.array(test_predict) * max_data_val

    plt.plot(
        range(
            len(plt_train_data),
            len(plt_train_data) + len(plt_test_predict)),
        plt_test_predict, 'y--')

    test_data_start = x_test_data[0]

    test_predict = []
    for i in range(0, 5):

        one_predict = model.predict(np.array([test_data_start]))

        test_predict.append(one_predict[0][0])

        test_data_start = np.hstack([test_data_start, one_predict])

        test_data_start = np.array([test_data_start[0, :][1:]])

    plt_test_predict = np.array(test_predict) * max_data_val

    plt.plot(
        range(
            len(plt_train_data),
            len(plt_train_data) + len(plt_test_predict)),
        plt_test_predict, 'g--')

    test_data_start = x_test_data[40]

    test_predict = []
    for i in range(0, 10):

        one_predict = model.predict(np.array([test_data_start]))

        test_predict.append(one_predict[0][0])

        test_data_start = np.hstack([test_data_start, one_predict])

        test_data_start = np.array([test_data_start[0, :][1:]])

    plt_test_predict = np.array(test_predict) * max_data_val

    plt.plot(
        range(
            len(plt_train_data) + 40,
            len(plt_train_data) + len(plt_test_predict) + 40),
        plt_test_predict, 'y--')

    test_data_start = x_test_data[40]

    test_predict = []
    for i in range(0, 5):

        one_predict = model.predict(np.array([test_data_start]))

        test_predict.append(one_predict[0][0])

        test_data_start = np.hstack([test_data_start, one_predict])

        test_data_start = np.array([test_data_start[0, :][1:]])

    plt_test_predict = np.array(test_predict) * max_data_val

    plt.plot(
        range(
            len(plt_train_data) + 40,
            len(plt_train_data) + len(plt_test_predict) + 40),
        plt_test_predict, 'g--')

    test_data_start = x_test_data[80]

    test_predict = []
    for i in range(0, 10):

        one_predict = model.predict(np.array([test_data_start]))

        test_predict.append(one_predict[0][0])

        test_data_start = np.hstack([test_data_start, one_predict])

        test_data_start = np.array([test_data_start[0, :][1:]])

    plt_test_predict = np.array(test_predict) * max_data_val

    plt.plot(
        range(
            len(plt_train_data) + 80,
            len(plt_train_data) + len(plt_test_predict) + 80),
        plt_test_predict, 'y--')

    test_data_start = x_test_data[80]

    test_predict = []
    for i in range(0, 5):

        one_predict = model.predict(np.array([test_data_start]))

        test_predict.append(one_predict[0][0])

        test_data_start = np.hstack([test_data_start, one_predict])

        test_data_start = np.array([test_data_start[0, :][1:]])

    plt_test_predict = np.array(test_predict) * max_data_val

    plt.plot(
        range(
            len(plt_train_data) + 80,
            len(plt_train_data) + len(plt_test_predict) + 80),
        plt_test_predict, 'g--')

    dirname = os.path.join('.', 'models', filename)

    if not os.path.exists(dirname):
        os.makedirs(dirname)

    plt.savefig(os.path.join(dirname, 'image2.png'))

    plt.show()


def save(filename, model, error):
    dirname = os.path.join('.', 'models', filename)

    if not os.path.exists(dirname):
        os.makedirs(dirname)

    model.save(os.path.join(dirname, 'model.h5'))


if __name__ == '__main__':
    main()
