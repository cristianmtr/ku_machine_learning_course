import numpy as np
import matplotlib.pyplot as plt

plt.style.use('ggplot')


def alternative_sigmoid(x):
    return x/(1+np.abs(x))


def alternative_sigmoid_derivative(x):
    return 1/((1+np.abs(x))**2)


def cost(preds, true):
    return np.sum((preds-true)**2)/len(preds)


class NeuralNetwork:
    def __init__(self, x, y, n_hidden_layers, lr):
        self.lr = lr
        self.input = x
        input_layer_size = self.input.shape[1]
        self.weights1 = np.random.rand(input_layer_size, n_hidden_layers) * np.sqrt(2.0/input_layer_size)
        self.weights2 = np.random.rand(n_hidden_layers, 1) * np.sqrt(2.0/n_hidden_layers)
        self.y = y
        self.output = np.zeros(y.shape)

    def feedforward(self):
        self.layer1 = alternative_sigmoid(np.dot(self.input, self.weights1))
        self.output = alternative_sigmoid(np.dot(self.layer1, self.weights2))

    def backprop(self):
        # application of the chain rule to find derivative of the loss function with respect to weights2 and weights1
        d_weights2 = np.dot(self.layer1.T, (2*(self.y - self.output) * alternative_sigmoid_derivative(self.output)))
        d_weights1 = np.dot(self.input.T,  (np.dot(2*(self.y - self.output) * alternative_sigmoid_derivative(self.output), self.weights2.T) * alternative_sigmoid_derivative(self.layer1)))

        # update the weights with the derivative (slope) of the loss function
        self.weights1 += self.lr * d_weights1
        self.weights2 += self.lr * d_weights2

    def predict(self, x):
        layer1 = alternative_sigmoid(np.dot(x, self.weights1))
        output = alternative_sigmoid(np.dot(layer1, self.weights2))
        return output

    def train(self, epochs):
        for i in range(epochs):
            self.feedforward()
            self.backprop()
            if i % 1000 == 0:
                print("training cost at epoch", i, ":", cost(self.predict(self.input), self.y))


def main():
    # data
    train_data = np.loadtxt("sincTrain25.dt")
    x_train = train_data[:, 0]
    x_train = x_train.reshape((x_train.shape[0],-1))
    y_train = train_data[:, 1]
    y_train = y_train.reshape((y_train.shape[0],-1))
    val_data = np.loadtxt("sincValidate10.dt")
    x_val = val_data[:, 0]
    x_val = x_val.reshape((x_val.shape[0],-1))
    y_val = val_data[:, 1]
    y_val = y_val.reshape((y_val.shape[0],-1))

    nn = NeuralNetwork(x_train, y_train, 20, 0.00001)

    nn.train(30000)

    val_preds = nn.predict(x_val)
    train_preds = nn.predict(x_train)

    val_cost = cost(val_preds, y_val)
    print('val cost', val_cost)

    plt.figure(figsize=(12, 12))
    plt.scatter(x_train, y_train, c='green', label='training')
    plt.scatter(x_val, y_val, c='blue', label='validation')
    plt.scatter(x_val, val_preds, c='yellow', label='val. prediction')
    plt.scatter(x_train, train_preds, c='orange', label='train prediction')
    plt.legend()
    plt.title("Model predictions")
    plt.xlabel("X")
    plt.ylabel("y")
    plt.gcf().savefig("model.png")
    plt.show()


if __name__ == "__main__":
    main()
