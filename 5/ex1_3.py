import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from matplotlib import pyplot as plt


class Normalizer(object):
    def __init__(self, data):
        self.data_shape = data.shape[1]
        self.feature_params = {}
        for i in range(self.data_shape):
            feature = data[:, i]
            feature_mean = np.mean(feature)
            feature_std = np.std(feature)
            self.feature_params[i] = [feature_mean, feature_std]

    def normalize(self, data):
        normalized_data = np.zeros((22, 2), dtype=object)
        print("normalizing data...")
        assert data.shape[1] == self.data_shape
        new_data = np.zeros_like(data)
        for i in range(self.data_shape):
            # print("feature ", i)
            feature = data[:, i]
            feature_mean = self.feature_params[i][0]
            feature_std = self.feature_params[i][1]
            new_data[:, i] = (feature-feature_mean)/feature_std
            normalized_data[i, 0] = "%.5f" % np.mean(new_data[:, i])
            normalized_data[i, 1] = "%.5f" % np.var(new_data[:, i])
            # print("mean of normalized data: %.20f" %np.mean(new_data[:,i]))
            # print("variance of normalized data: %.5f" %np.var(new_data[:,i]))
        df = pd.DataFrame(normalized_data, columns=[
                          "mean", "variance"])
        return new_data, df


def zero_one_loss(true, pred):
    sum = 0
    for i in range(len(true)):
        if true[i] != pred[i]:
            sum += 1

    return sum/len(true)


def load_data():
    train = np.loadtxt("parkinsonsTrainStatML.dt")
    x_train = train[:, :-1]
    y_train = train[:, -1]

    normalizer = Normalizer(x_train)
    x_train_normalized, df = normalizer.normalize(x_train)
    with open("x_train_normalization.csv", "w") as f:
        print("saving stats about normalization of train data at x_train_normalization.csv")
        f.writelines(df.to_csv())

    test = np.loadtxt("parkinsonsTestStatML.dt")
    x_test = test[:, :-1]
    y_test = test[:, -1]
    x_test_normalized, df = normalizer.normalize(x_test)
    with open("x_test_normalization.csv", "w") as f:
        print("saving stats about normalization of test data at x_test_normalization.csv")
        f.writelines(df.to_csv())

    return x_train_normalized, y_train, x_test_normalized, y_test


def kernel_gram_matrix(x_train, x_train2, gamma):
    matrix = np.zeros((len(x_train), len(x_train2)))
    for i in range(len(x_train)):
        for j in range(len(x_train2)):
            x_i = x_train[i]
            x_j = x_train2[j]
            result = np.exp(-1 * gamma * np.linalg.norm(x_i-x_j)**2)
            matrix[i, j] = result
    return matrix


def grid_search(x_train, y_train):
    c_options = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
    gamma_options = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100]
    lowest_loss = None
    # c, gamma
    lowest_loss_params = []
    # grid search
    kfolds = list(StratifiedKFold(
        n_splits=5, shuffle=True).split(x_train, y_train))
    # import pprint
    # pprint.pprint(kfolds)
    for c in c_options:
        for gamma in gamma_options:
            losses_for_params = []
            for train_index, test_index in kfolds:
                x_train_fold = x_train[train_index]
                y_train_fold = y_train[train_index]
                x_test_fold = x_train[test_index]
                y_test_fold = y_train[test_index]

                model = SVC(C=c, kernel='precomputed')
                model.fit(kernel_gram_matrix(x_train_fold,
                                             x_train_fold, gamma), y_train_fold)
                preds_fold = model.predict(kernel_gram_matrix(
                    x_test_fold, x_train_fold, gamma))
                loss = zero_one_loss(y_test_fold, preds_fold)
                losses_for_params.append(loss)
            loss = np.mean(losses_for_params)
            if lowest_loss is None or loss < lowest_loss:
                lowest_loss = loss
                lowest_loss_params = [c, gamma]
                # print("new minimum: %s" %lowest_loss)

    print("lowest loss during grid search = %s. c = %s, gamma = %s" %
          (lowest_loss, lowest_loss_params[0], lowest_loss_params[1]))
    return lowest_loss_params[0], lowest_loss_params[1]


def train():
    x_train_full, y_train_full, x_test, y_test = load_data()

    # to be plotted
    train_sizes = []
    val_losses = []
    test_losses = []
    bounds = []

    for i in range(9, len(y_train_full), 10):
        train_size = len(y_train_full) - i
        x_val = x_train_full[train_size:]
        y_val = y_train_full[train_size:]
        x_train = x_train_full[:train_size]
        y_train = y_train_full[:train_size]
        print("train set length = ", len(y_train))
        print("val set length = ", len(y_val))

        c, gamma = grid_search(x_train, y_train)

        model = SVC(C=c, kernel='precomputed')
        model.fit(kernel_gram_matrix(x_train, x_train, gamma), y_train)
        # train_preds = model.predict(kernel_gram_matrix(x_train, x_train, gamma))
        # train_loss = zero_one_loss(y_train, train_preds)
        # print("loss on train set = %.4f" %train_loss)

        val_preds = model.predict(kernel_gram_matrix(x_val, x_train, gamma))
        val_loss = zero_one_loss(y_val, val_preds)
        print("loss on val set = %.4f" % val_loss)

        test_preds = model.predict(kernel_gram_matrix(x_test, x_train, gamma))
        test_loss = zero_one_loss(y_test, test_preds)
        print("loss on test set = %.4f" % test_loss)

        train_sizes.append(len(x_train))
        val_losses.append(val_loss)
        test_losses.append(test_loss)

        delta = 0.05
        n_val = len(y_val)
        epsilon = np.sqrt((np.log(1/delta))/(2*n_val))
        bound = val_loss + epsilon
        bounds.append(bound)

    np.save("data", [
        train_sizes, bounds, val_losses, test_losses
    ])


def plot():
    train_sizes, bounds, val_losses, test_losses = np.load("data.npy")

    plt.style.use('ggplot')
    fig = plt.figure(figsize=(12, 12))
    ax1 = fig.add_subplot(111)
    # ax2 = ax1.twiny()

    ax1.scatter(train_sizes, bounds, c='r', label='bound')
    ax1.scatter(train_sizes, val_losses, c='b', label='val. loss')
    ax1.scatter(train_sizes, test_losses, c='cyan', label='test loss')

    ax1.set_xlabel("size of training set", fontdict={"fontsize": 18})
    ax1.set_ylabel("bound / loss", fontdict={"fontsize": 18})

    ax1.set_xticks(train_sizes)
    # ax1.set_xticklabels(train_sizes)
    ax1.tick_params(axis="both", labelsize=18)
    ax1.legend()

    # ax1.scatter(np.flip(train_sizes), val_losses, c='b', label='val. loss')
    # ax2.set_xticks(np.flip(train_sizes))
    # ax2.set_xticklabels(np.flip(train_sizes))
    # ax2.tick_params(axis="both", labelsize=18)

    fig.savefig('ex1_3.png', bbox_inches='tight')
    # plt.show()


def ex1_3():
    print("ex 1.3...")

    train()
    plot()


if __name__ == "__main__":
    ex1_3()
