import numpy as np
import matplotlib.pyplot as plt

def lin_reg(X,y, title):
    X = np.vstack([X, np.repeat(1, len(X))]).T
    xt_x = np.linalg.pinv(np.matmul(X.T, X))
    w_t = np.matmul(np.matmul(xt_x, X.T), y.T)

    w = w_t[0]
    b = w_t[1]
    X = X[:, 0]
    print('w = %.4f, b = %.4f' % (w, b))
    preds = w * X + b
    with np.printoptions(precision=4):
        print('predictions = ', preds)
        print('y = ', y)
    mse = np.sum((preds-y)**2)/len(preds)
    print('mse %.4f' %mse)
    
    # plot
    plt.clf()
    plt.scatter(X, y, c='b',label='truth')
    plt.scatter(X, preds, label='predictions', c='red')
    plt.plot([X[0],X[-1]],[preds[0],preds[-1]], c='orange', label='regression line')
    plt.legend()
    # Temperature was measured in units of 1000 kelvin and energy radiation per cm2
# per second.
    plt.xlabel("Temperature - units of 1000 kelvin")
    plt.ylabel("energy radiations per cm2 per second")
    plt.title(title)

    return preds, mse, plt.gcf()


def main():
    """
    [[1.309 2.138]
 [1.471 3.421]
 [1.49  3.597]
 [1.565 4.34 ]
 [1.611 4.882]
 [1.68  5.66 ]]
 """
    data = np.loadtxt("DanWood.dt")
    X = data[:, 0]
    y = data[:, 1]
    preds, mse, fig = lin_reg(X,y,"Linear regression")
    fig.savefig('ex6-1.png')
    
    # variance and mse
    variance = np.var(y)
    print('variance %.4f' %variance)
    quotient = mse/variance
    print('mse/var = %.4f' %quotient)

    # nonlinear
    X = X**3
    preds, mse, fig = lin_reg(X,y, "Linear regression x^3")
    fig.savefig('ex6-2.png')



if __name__ == "__main__":
    main()
