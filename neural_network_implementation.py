import numpy as np
import pandas as pd

rgen = np.random.default_rng()

# dataset generator
def data_generator(n_features, n_values):
    features = rgen.random((n_features, n_values))
    weights = rgen.random((1, n_values))[0]
    targets = np.random.choice([0, 1], n_features)
    data = pd.DataFrame(features, columns=['x0', 'x1', 'x2'])
    data['targets'] = targets
    return data, weights


# Defining the linear regression function
bias = 0.5
l_rate = 0.01
epochs = 50
epoch_loss = []

# Compute the weighted sum
def get_weighted_sum(features, weights, bias):
    return np.dot(features, weights) + bias

# Defining the activation function
def sigmoid(w_sum):
    return 1/(1+np.exp(-w_sum))

# Defining the loss function
def cross_entropy(target, prediction):
    return -(target * np.log10(prediction) + (1 - target) * np.log10(1 - prediction))

# Updating the weights
def update_weights(weights, l_rate, target, prediction, feature):
    new_weights = []
    for x, w in zip(feature, weights):
        new_weight = w + l_rate * (target - prediction) * x
        new_weights.append(new_weight)
    return new_weights

# Updating the bias
def update_bias(bias, l_rate, target, prediction):
    return bias + l_rate * (target - prediction)


# Generating a dataset
data, weights = data_generator(50, 3)

# Implementing a model training function
def train_model(data, weights, bias, l_rate, epochs):
    for e in range(epochs):
        losses = []
        for i in range(len(data)):
            feature = data.loc[i][:-1]
            target = data.loc[i][-1]
            w_sum = get_weighted_sum(feature, weights, bias)
            prediction = sigmoid(w_sum)
            loss = cross_entropy(target, prediction)
            losses.append(loss)
            # gradient descent
            weights = update_weights(
                weights, l_rate, target, prediction, feature)
            bias = update_bias(bias, l_rate, target, prediction)
        avg_loss = sum(losses) / len(losses)
        epoch_loss.append(avg_loss)
        print("**********************************************************")
        print("epoch", e)
        print(avg_loss)


# train the model using the model training function
train_model(data, weights, bias, l_rate, epochs)

# plot the average loss
df = pd.DataFrame(epoch_loss)
df_plot = df.plot(kind="line", grid=True).get_figure()
df_plot.savefig("training_loss.png")
