import numpy as np
import matplotlib.pyplot as plt
import random
import math


def slice_data(dataset, idx_start, window_size, target_size, param_size):
    data = []
    labels = []

    idx_end = len(dataset) - window_size - target_size + 1

    for i in range(idx_start, idx_end):
        ind_d = range(i, i + window_size)
        # 'PRECTOT', 'WS2M', 'T2MDEW', 'ALLSKY', 'RH2M', 'PS'
        data.append(dataset[['PRECTOT', 'WS2M', 'ALLSKY', 'RH2M', 'PS', 'T2M']].values[ind_d])
        ind_l = range((i + window_size), (i + window_size + target_size))
        labels.append(dataset['T2M'].values[ind_l])
    data = np.vstack(data[:]).reshape(idx_end, window_size, param_size + 1)
    labels = np.vstack(labels[:]).reshape(idx_end, target_size)

    return data, labels



def plot_prediction(data, T_test, Y_test, y_pred, T2M_std, T2M_mean, num_plots):
    y_pred_unorm = y_pred * T2M_std + T2M_mean
    Y_test_unorm = Y_test * T2M_std + T2M_mean
    X_temp = T_test * T2M_std + T2M_mean
    X_unorm = data['T2M'].values * T2M_std + T2M_mean

    for k in range(0, num_plots):
        idx = random.randint(0, Y_test.shape[0] - 1)
        plt.figure()
        plt.plot(range(-T_test.shape[1], 0), X_temp[idx, :])
        for l in range(0, Y_test.shape[1]):
            plt.scatter(l, Y_test_unorm[idx, l], c='red', marker='x')
            plt.scatter(l, y_pred_unorm[idx, l], c='green', marker='x')
        plt.xlabel('day')
        plt.ylabel('temperature')
        plt.ylim(np.min(X_unorm), np.max(X_unorm))
        plt.legend(['historical data', 'real data', 'predicted data'])
        plt.grid()
        plt.show()


def random_mini_batches(X, Y, mini_batch_size=64, seed=0):

    m = X.shape[0]  # number of training examples
    mini_batches = []
    np.random.seed(seed)

    # Step 1: Shuffle (X, Y)
    permutation = list(np.random.permutation(m))
    shuffled_X = X[permutation, :]
    shuffled_Y = Y[permutation, :]

    # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
    num_complete_minibatches = math.floor(
        m / mini_batch_size)  # number of mini batches of size mini_batch_size in your partitionning
    for k in range(0, num_complete_minibatches):
        mini_batch_X = shuffled_X[k * mini_batch_size: k * mini_batch_size + mini_batch_size, :]
        mini_batch_Y = shuffled_Y[k * mini_batch_size: k * mini_batch_size + mini_batch_size, :]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    return mini_batches