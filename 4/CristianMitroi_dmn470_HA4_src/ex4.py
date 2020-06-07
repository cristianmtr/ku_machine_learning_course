import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold


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
    kfolds = list(StratifiedKFold(n_splits=5, shuffle=True).split(x_train, y_train))
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


def ex4():
    print("ex 4...")
    x_train, y_train, x_test, y_test = load_data()

    c, gamma = grid_search(x_train, y_train)

    model = SVC(C=c, kernel='precomputed')
    model.fit(kernel_gram_matrix(x_train, x_train, gamma), y_train)

    train_preds = model.predict(kernel_gram_matrix(x_train, x_train, gamma))
    loss = zero_one_loss(y_train, train_preds)
    print("loss on train set = %.4f" %loss)

    test_preds = model.predict(kernel_gram_matrix(x_test, x_train, gamma))
    loss = zero_one_loss(y_test, test_preds)
    print("loss on test set = %.4f" %loss)


if __name__ == "__main__":
    ex4()
