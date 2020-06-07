import numpy as np
from matplotlib import pyplot as plt


def my_zero_one_loss(true, pred):
    sum = 0
    for i in range(len(true)):
        if true[i] != pred[i]:
            sum += 1

    return sum/len(true)


class MyLogisticRegression(object):
    def __init__(self, x, y, num_steps, lr):
        # init weights
        self.W = np.zeros(x.shape[1])

        # perform gr. desc. for each step
        for _ in range(num_steps):
            self.W = self.step(self.W, lr, x, y)

        y_pred = self.sigmoid(np.matmul(x, self.W))
        y_pred = (y_pred > .5).astype(int)
        print("train loss", my_zero_one_loss(y, y_pred))

    def predict(self, x):
        y_pred = self.sigmoid(np.matmul(x, self.W))
        y_pred = (y_pred > .5).astype(int)
        return y_pred

    def sigmoid(self, x):
        return 1.0 / (1 + np.exp(-x))

    def step(self, W, lr, x, y):
        h = self.sigmoid(np.matmul(x, W))
        grad = np.matmul(x.T, (h - y)) / len(y)
        W = W - lr * grad
        return W


def load_data():
    print("loading data...")
    train = np.loadtxt("IrisTrainML.dt")

    x_train = train[:, :2]
    y_train = train[:, 2]

    # remove class 2
    indices_to_keep = np.where(y_train != 2)
    x_train = x_train[indices_to_keep]
    y_train = y_train[indices_to_keep]

    # add intercept
    intercept = np.ones((x_train.shape[0], 1))
    # add intercept to original data
    x_train = np.hstack((x_train, intercept))
    print("x train sample", x_train[0])
    print("x_train shape", x_train.shape)

    test = np.loadtxt("IrisTestML.dt")
    x_test = test[:, :2]
    y_test = test[:, 2]

    # remove class 2
    indices_to_keep = np.where(y_test != 2)
    x_test = x_test[indices_to_keep]
    y_test = y_test[indices_to_keep]

    # add intercept
    intercept = np.ones((x_test.shape[0], 1))
    # add intercept to original data
    x_test = np.hstack((x_test, intercept))

    print("x_test sample", x_test[0])
    print("x_test shape", x_test.shape)

    return x_train, y_train, x_test, y_test


def ex5():
    x_train, y_train, x_test, y_test = load_data()

    lr = MyLogisticRegression(x_train,
                              y_train,
                              100000,
                              0.04)

    print("weights for Logistic regression model: %s" % (lr.W))

    preds = lr.predict(x_test)
    print("test loss", my_zero_one_loss(y_test, preds))


if __name__ == "__main__":
    ex5()
