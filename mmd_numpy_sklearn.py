# Compute MMD (maximum mean discrepancy) using numpy and scikit-learn.

import numpy as np
from sklearn import metrics


def mmd_linear(X, Y):
    """MMD 线性核(linear kernel) 即 k(x,y) = <x,y>
    线性MMD的优化版本
    原始版本:
        def mmd_linear(X, Y):
            XX = np.dot(X, X.T)
            YY = np.dot(Y, Y.T)
            XY = np.dot(X, Y.T)
            return XX.mean() + YY.mean() - 2 * XY.mean()

    参数:
        X {[n_sample1, dim]} -- [X matrix]
        Y {[n_sample2, dim]} -- [Y matrix]

    返回:
        [scalar] -- [MMD value]
    """
    delta = X.mean(0) - Y.mean(0)
    return delta.dot(delta.T)


def mmd_rbf(X, Y, gamma=1.0):
    """MMD 径向基函数核(rbf kernel此处高斯核) 即 k(x,y) = exp(-gamma * ||x-y||^2 / 2)

    参数:
        X {[n_sample1, dim]} -- [X matrix]
        Y {[n_sample2, dim]} -- [Y matrix]

    参数关键字:
        gamma {float} -- [kernel parameter] (default: {1.0})

    返回:
        [scalar] -- [MMD value]
    """
    XX = metrics.pairwise.rbf_kernel(X, X, gamma)
    YY = metrics.pairwise.rbf_kernel(Y, Y, gamma)
    XY = metrics.pairwise.rbf_kernel(X, Y, gamma)
    return XX.mean() + YY.mean() - 2 * XY.mean()


def mmd_poly(X, Y, degree=2, gamma=1, coef0=0):
    """MMD 多项式核 即 k(x,y) = (gamma <X, Y> + coef0)^degree

    参数:
        X {[n_sample1, dim]} -- [X matrix]
        Y {[n_sample2, dim]} -- [Y matrix]

    参数关键字:
        degree {int} -- [degree] (default: {2})
        gamma {int} -- [gamma] (default: {1})
        coef0 {int} -- [constant item] (default: {0})

    返回:
        [scalar] -- [MMD value]
    """
    XX = metrics.pairwise.polynomial_kernel(X, X, degree, gamma, coef0)
    YY = metrics.pairwise.polynomial_kernel(Y, Y, degree, gamma, coef0)
    XY = metrics.pairwise.polynomial_kernel(X, Y, degree, gamma, coef0)
    return XX.mean() + YY.mean() - 2 * XY.mean()


if __name__ == '__main__':
    a = np.arange(1, 10).reshape(3, 3)
    b = [[6, 2, 0], [4, 5, 7], [5, 2, 5], [0, 2, 5]]
    b = np.array(b)
    print("a矩阵：\n",a)
    print("b矩阵：\n",b,"\n", "*"*20)
    print(mmd_linear(a, b))
    print(mmd_rbf(a, b))
    print(mmd_poly(a, b))
