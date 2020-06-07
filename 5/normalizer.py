import numpy as np
import pandas as pd


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
        normalized_data = np.zeros((self.data_shape, 2), dtype=object)
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
