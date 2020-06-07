import tqdm

from sklearn.model_selection import train_test_split

from matplotlib import pyplot as plt
import numpy as np
import os


def knn(data, labels, newpoint, k):
    """returns label of nearest K neighbours for newpoint"""
    dists = np.linalg.norm(data-newpoint, axis=1)
    index_to_sort_by = np.argsort(dists)
    sorted_labels = labels[index_to_sort_by][:k]  # or K

    return np.bincount(sorted_labels).argmax()


def classify_two_digits(K_VALUES, X_train, y_train, X_test, y_test, X_val, y_val, digits):
    print('classifying ', digits)
    # train
    ix = np.isin(y_train, [digits[0], digits[1]])
    X_train_digits = X_train[ix]
    y_train_digits = y_train[ix]

    # test
    ix = np.isin(y_test, [digits[0], digits[1]])
    X_test_digits = X_test[ix]
    y_test_digits = y_test[ix]

    # val
    ix = np.isin(y_val, [digits[0], digits[1]])
    X_val_digits = X_val[ix]
    y_val_digits = y_val[ix]

    print('shapes:')
    print(X_train_digits.shape)
    print(y_train_digits.shape)
    print(X_test_digits.shape)
    print(y_test_digits.shape)
    print(X_val_digits.shape)
    print(y_val_digits.shape)

    file_val = 'preds_val_%s%s.npy' % (digits[0], digits[1])
    file_test = 'preds_test_%s%s.npy' % (digits[0], digits[1])
    if not os.path.exists(file_val) or not os.path.exists(file_test):

        preds_val_digits = np.repeat(-1, len(K_VALUES)*y_val_digits.shape[0]).reshape(
            (len(K_VALUES), y_val_digits.shape[0]))

        preds_test_digits = np.repeat(-1, len(K_VALUES)*y_test_digits.shape[0]).reshape(
            (len(K_VALUES), y_test_digits.shape[0]))

        for k_index, k in enumerate(tqdm.tqdm(K_VALUES)):
            for i in range(len(X_val_digits)):
                pred_val = knn(X_train_digits, y_train_digits,
                               X_val_digits[i], k)
                preds_val_digits[k_index, i] = pred_val
            for i in range(len(X_test_digits)):
                pred_test = knn(X_train_digits, y_train_digits,
                                X_test_digits[i], k)
                preds_test_digits[k_index, i] = pred_test

        np.save(file_val, preds_val_digits)
        np.save(file_test, preds_test_digits)

    preds_val_digits = np.load(file_val)
    preds_test_digits = np.load(file_test)

    errors = []
    for k_index in range(len(K_VALUES)):
        preds_val = preds_val_digits[k_index]
        preds_test = preds_test_digits[k_index]
        print('k', K_VALUES[k_index])
        val_loss = np.mean(preds_val != y_val_digits)
        print('val', val_loss)
        test_loss = np.mean(preds_test != y_test_digits)
        print('test', test_loss)
        errors.append((val_loss, test_loss))

    errors = np.array(errors)

    plt.clf()
    plt.scatter([i for i in K_VALUES], errors[:, 0], label='val error')
    plt.scatter([i for i in K_VALUES], errors[:, 1], label='test error')
    plt.xticks(K_VALUES)
    plt.legend()
    plt.xlabel('values of K')
    plt.ylabel('error')
    plt.title("kNN perf. on classifying %s and %s" % (digits[0], digits[1]))
    plt.gcf().savefig("ex4-%s%s.png" % (digits[0], digits[1]))


def main():
    mnist_train = np.loadtxt("MNIST-cropped-txt/MNIST-Train-cropped.txt")

    mnist_train_labels = np.loadtxt(
        "MNIST-cropped-txt/MNIST-Train-Labels-cropped.txt", dtype=int)

    X_test = np.loadtxt("MNIST-cropped-txt/MNIST-Test-cropped.txt")
    y_test = np.loadtxt(
        "MNIST-cropped-txt/MNIST-Test-Labels-cropped.txt", dtype=int)

    X_test = X_test.reshape((-1, 784))

    mnist_train = mnist_train.reshape((10000, 784))

    # Test reduced sample

    # X_train, X_val, y_train, y_val = train_test_split(mnist_train, mnist_train_labels, test_size=0.7, random_state=42)

    # X_val, _, y_val, _ = train_test_split(X_val, y_val, test_size=0.7, random_state=42)

    # X_test, _, y_test, _ = train_test_split(X_test, y_test, test_size=0.7, random_state=42)

    X_train, X_val, y_train, y_val = train_test_split(
        mnist_train, mnist_train_labels, test_size=0.2, random_state=42)

    K_VALUES = [i for i in range(1, 34, 2)]
    print(K_VALUES)

    print('val shapes ', X_val.shape, y_val.shape)
    print('train shapes ', X_train.shape, y_train.shape)
    print('test shapes ', X_test.shape, y_test.shape)
    # K_VALUES, X_train, y_train, X_test, y_test, X_val, y_val, digits):
    classify_two_digits(K_VALUES, X_train, y_train,
                        X_test, y_test, X_val, y_val, [0, 1])
    classify_two_digits(K_VALUES, X_train, y_train,
                        X_test, y_test, X_val, y_val, [0, 8])
    classify_two_digits(K_VALUES, X_train, y_train,
                        X_test, y_test, X_val, y_val, [5, 6])


if __name__ == "__main__":
    main()
