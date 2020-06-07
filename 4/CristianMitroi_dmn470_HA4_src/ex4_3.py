import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
from sklearn.svm import SVC
from sklearn.datasets import make_classification

iris = datasets.load_iris()


def ex4_3():
    print("ex 4.3...")
    X, y = make_classification(n_features=2, n_redundant=0, n_informative=2,
                               n_clusters_per_class=1, n_classes=2, random_state=46)
    print(np.unique(y))

    for C in [0.01, 1, 100]:
        plt.clf()
        clf = SVC(kernel='rbf', C=C, gamma=0.1).fit(X, y)

        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        h = np.abs((x_max / x_min)/100)
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                             np.arange(y_min, y_max, h))

        plt.subplot(1, 1, 1)
        ravel = np.c_[xx.ravel(), yy.ravel()]
        Z = clf.predict(ravel)
        Z = Z.reshape(xx.shape)
        plt.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.8)

        plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired)
        plt.xlabel('feature X')
        plt.ylabel('feature Y')
        plt.xlim(xx.min(), xx.max())
        plt.title('SVC on artificial dataset, C=%s' % C)
        print("saving figure to 'ex4_3_C_%s.png'" % C)
        plt.gcf().savefig('ex4_3_C_%s.png' % C, bbox_inches="tight")


if __name__ == "__main__":
    ex4_3()
