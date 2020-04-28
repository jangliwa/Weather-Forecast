import numpy as np
import pandas as pd
import tensorflow as tf
import random
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.layers import Input, Dense, Activation
from MyLibrary import plot_prediction, slice_data, random_mini_batches
from tensorflow.python.framework import ops

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
def create_placeholders(n_x, n_y):
    X = tf.placeholder(tf.float32, shape=(n_x, None), name='X')
    Y = tf.placeholder(tf.float32, shape=(n_y, None), name='Y')

    return X, Y


# Create placeholders
Xph, Yph = create_placeholders(X_train.shape[1] * X_train.shape[2], Y_train.shape[1])
print("X = " + str(Xph))
print("Y = " + str(Yph))


# Initialize parameters with Xavier initializer, 4-layer NN
def initialize_parameters(Xph, Yph):
    tf.set_random_seed(2)  # so that your "random" numbers match ours

    l1 = 5
    l2 = 5
    l3 = 5

    W1 = tf.get_variable("W1", [l1, Xph.shape[0].value], initializer=tf.contrib.layers.xavier_initializer(seed=1))
    b1 = tf.get_variable("b1", [l1, 1], initializer=tf.zeros_initializer())
    W2 = tf.get_variable("W2", [l2, l1], initializer=tf.contrib.layers.xavier_initializer(seed=1))
    b2 = tf.get_variable("b2", [l2, 1], initializer=tf.zeros_initializer())
    W3 = tf.get_variable("W3", [l3, l2], initializer=tf.contrib.layers.xavier_initializer(seed=1))
    b3 = tf.get_variable("b3", [l3, 1], initializer=tf.zeros_initializer())
    W4 = tf.get_variable("W4", [Yph.shape[0].value, l3], initializer=tf.contrib.layers.xavier_initializer(seed=1))
    b4 = tf.get_variable("b4", [Yph.shape[0].value, 1], initializer=tf.zeros_initializer())

    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2,
                  "W3": W3,
                  "b3": b3,
                  "W4": W4,
                  "b4": b4
                  }

    return parameters


tf.reset_default_graph()
with tf.Session() as sess:
    parameters = initialize_parameters(Xph, Yph)
    print("W1 = " + str(parameters["W1"]))
    print("b1 = " + str(parameters["b1"]))
    print("W2 = " + str(parameters["W2"]))
    print("b2 = " + str(parameters["b2"]))


# Forward propagation
def forward_propagation(Xph, parameters):
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']
    W3 = parameters['W3']
    b3 = parameters['b3']
    W4 = parameters['W4']
    b4 = parameters['b4']

    Z1 = tf.add(tf.matmul(W1, Xph), b1)
    A1 = tf.nn.relu(Z1)
    Z2 = tf.add(tf.matmul(W2, A1), b2)
    A2 = tf.nn.relu(Z2)
    Z3 = tf.add(tf.matmul(W3, A2), b3)
    A3 = tf.nn.relu(Z3)
    Z4 = tf.add(tf.matmul(W4, A3), b4)

    return Z4

# Forward propagation_dropout
def forward_propagation_dropout(Xph, parameters, kp):
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']
    W3 = parameters['W3']
    b3 = parameters['b3']
    W4 = parameters['W4']
    b4 = parameters['b4']

    Z1 = tf.add(tf.matmul(W1, Xph), b1)
    A1 = tf.nn.relu(Z1)
    A1 = tf.nn.dropout(A1, keep_prob=kp)
    Z2 = tf.add(tf.matmul(W2, A1), b2)
    A2 = tf.nn.relu(Z2)
    A2 = tf.nn.dropout(A2, keep_prob=kp)
    Z3 = tf.add(tf.matmul(W3, A2), b3)
    A3 = tf.nn.relu(Z3)
    A3 = tf.nn.dropout(A3, keep_prob=kp)
    Z4 = tf.add(tf.matmul(W4, A3), b4)

    return Z4

tf.reset_default_graph()

with tf.Session() as sess:
    Xph, Yph = create_placeholders(X_train.shape[1] * X_train.shape[2], Y_train.shape[1])
    parameters = initialize_parameters(Xph, Yph)
    Z4 = forward_propagation(Xph, parameters)
    print("Z4 = " + str(Z4))


# Cost function without regularization
def compute_cost(Z4, Y):
    logits = Z4
    labels = Y

    cost = tf.keras.losses.MAE(y_true=labels, y_pred=logits)

    return cost


tf.reset_default_graph()

with tf.Session() as sess:
    Xph, Yph = create_placeholders(X_train.shape[1] * X_train.shape[2], Y_train.shape[1])
    parameters = initialize_parameters(Xph, Yph)
    Z4 = forward_propagation(Xph, parameters)
    cost = compute_cost(Z4, Yph)
    print("cost = " + str(cost))


# Cost function with L2 regularization
def compute_cost_L2(Z4, Y, minibatch_size, lambd, parameters):
    logits = Z4
    labels = Y

    cost = lambd * (1 / minibatch_size) * (
                tf.nn.l2_loss(parameters["W1"]) + tf.nn.l2_loss(parameters["W2"]) + tf.nn.l2_loss(
            parameters["W3"]) + tf.nn.l2_loss(parameters["W3"]))

    return cost


# Model function
def model(X_train, Y_train, learning_rate=0.0001,
          num_epochs=1000, minibatch_size=32, print_cost=True):
    ops.reset_default_graph()
    tf.set_random_seed(1)
    seed = 3
    (n_x, m) = (X_train.shape[1] * X_train.shape[2], X_train.shape[0])
    n_y = Y_train.shape[1]
    costs = []

    # Create Placeholders of shape (n_x, n_y)
    Xph, Yph = create_placeholders(n_x, n_y)

    # Initialize parameters
    parameters = initialize_parameters(Xph, Yph)

    # Forward propagation: Build the forward propagation in the tensorflow graph
    Z4 = forward_propagation_dropout(Xph, parameters, 0.95)
    #Z4 = forward_propagation(Xph, parameters)

    # Cost function: Add cost function to tensorflow graph
    cost = compute_cost(Z4, Yph)

    l2_cost = compute_cost_L2(Z4, Y, minibatch_size, 0.001, parameters)
    cost = cost + l2_cost

    # Backpropagation using AdamOptimizer.
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

    # Initialize all the variables
    init = tf.global_variables_initializer()

    # Start the session to compute the tensorflow graph
    with tf.Session() as sess:

        sess.run(init)

        for epoch in range(num_epochs):

            epoch_cost = 0.
            num_minibatches = int(m / minibatch_size)
            seed = seed + 1
            minibatches = random_mini_batches(X_train, Y_train, minibatch_size, seed)

            for minibatch in minibatches:
                # Select a minibatch
                (minibatch_X, minibatch_Y) = minibatch
                minibatch_X = np.reshape(minibatch_X,
                                         (minibatch_X.shape[0], minibatch_X.shape[1] * minibatch_X.shape[2]))
                minibatch_X = minibatch_X.T
                minibatch_Y = minibatch_Y.T
                # Run the graph on minibatch
                _, minibatch_cost = sess.run([optimizer, cost], feed_dict={Xph: minibatch_X, Yph: minibatch_Y})

                epoch_cost += minibatch_cost / minibatch_size

            # Print the cost every epoch
            if print_cost == True and epoch % 100 == 0:
                print("Cost after epoch %i: %f" % (epoch, np.sum(epoch_cost)))
            if print_cost == True and epoch % 5 == 0:
                costs.append(epoch_cost)

        # Plot the cost function
        plt.plot(np.squeeze(costs))
        plt.ylabel('cost')
        plt.xlabel('iterations (per fives)')
        plt.title("Learning rate =" + str(learning_rate))
        plt.legend(['1 day', '2 day', '3 day', '4 day', '5 day'])
        plt.grid()
        plt.show()

        # Save parameters in variable
        parameters = sess.run(parameters)
        print("Parameters have been trained!")

        return parameters


parameters = model(X_train, Y_train)

# Predict model
tf.reset_default_graph()

with tf.Session() as sess:
    Xph, Yph = create_placeholders(X_train.shape[1] * X_train.shape[2], X_train.shape[1])
    z4 = forward_propagation(Xph, parameters)
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1] * X_train.shape[2])).T
    preds_train = sess.run(z4, feed_dict={Xph: X_train})
    preds_train = preds_train.T

tf.reset_default_graph()

with tf.Session() as sess:
    Xph, Yph = create_placeholders(X_val.shape[1] * X_val.shape[2], Y_val.shape[1])
    z4 = forward_propagation(Xph, parameters)
    X_val = np.reshape(X_val, (X_val.shape[0], X_val.shape[1] * X_val.shape[2])).T
    preds_val = sess.run(z4, feed_dict={Xph: X_val})
    preds_val = preds_val.T

tf.reset_default_graph()

with tf.Session() as sess:
    Xph, Yph = create_placeholders(X_test.shape[1] * X_test.shape[2], Y_test.shape[1])
    z4 = forward_propagation(Xph, parameters)
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1] * X_test.shape[2])).T
    preds_test = sess.run(z4, feed_dict={Xph: X_test})
    preds_test = preds_test.T

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

y_pred = preds_test

plot_prediction(data, T_test, Y_test, y_pred, T2M_std, T2M_mean, 5)
