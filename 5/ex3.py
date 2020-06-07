from matplotlib import pyplot as plt
from sklearn.datasets import make_classification
from normalizer import Normalizer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import zero_one_loss
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier


def ex3():
    print("ex 3...")
    X, y = load_wine(return_X_y=True)

    x_train, x_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=46)

    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(x_train, y_train)
    predictions = knn.predict(x_test)
    loss = zero_one_loss(y_test, predictions)
    print("loss of knn without normalization", loss)

    # plt.figure(figsize=(12, 12))
    # plt.scatter(x_train[:, 0], x_train[:, 1], c=y_train)
    # plt.title("Train data before normalization")
    # plt.gcf().savefig("ex3_before.png", bbox_inches='tight')

    normalizer = Normalizer(x_train)
    x_train_normalized, _ = normalizer.normalize(x_train)
    x_test_normalized, _ = normalizer.normalize(x_test)

    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(x_train_normalized, y_train)
    predictions = knn.predict(x_test_normalized)
    loss = zero_one_loss(y_test, predictions)
    print("loss of knn with normalization", loss)

    # plt.figure(figsize=(12, 12))
    # plt.scatter(x_train_normalized[:, 0], x_train_normalized[:, 1], c=y_train)
    # plt.title("Train data after normalization")
    # plt.gcf().savefig("ex3_after.png", bbox_inches='tight')

    rf = RandomForestClassifier(n_estimators=100, max_depth=2,
                                random_state=0)
    rf.fit(x_train, y_train)
    predictions = rf.predict(x_test)
    loss = zero_one_loss(y_test, predictions)
    print("loss of rf without normalization", loss)

    rf = RandomForestClassifier(n_estimators=100, max_depth=2,
                                random_state=0)
    rf.fit(x_train_normalized, y_train)
    predictions = rf.predict(x_test_normalized)
    loss = zero_one_loss(y_test, predictions)
    print("loss of rf after normalization", loss)

    return None


if __name__ == "__main__":
    ex3()
