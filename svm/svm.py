#!/bin/env python3
import numpy as np
import pdb
import numpy.random as nrnd
from scipy.optimize import minimize


class SVM:

    def __ones(X):
        nX = np.ones((X.shape[0], X.shape[1]+1))
        nX[:, :-1] = X
        return nX

    def __init__(self):
        self.W = None
        self.loss = []

    def fit(self, X, y, tau=1e-5):
        """
        tau: inverse C from SVR
        """
        X = SVM.__ones(X)
        dim = X.shape[1]
        if self.W is None:
            self.W = nrnd.randn(dim)
        self.W = minimize(lambda x: self.__loss(X, y, tau, x), self.W, method='Newton-CG', jac=lambda x: self.__gradient(X, y, tau, x)).x
        return self

    def predict(self, X):
        return np.sum(self.W * SVM.__ones(X), axis=1)

    def __loss(self, X, y, tau, W):
        """
        Standard SVM loss
        """
        return np.sum(np.abs(np.sum(W * X, axis=1) - y)) + tau * np.sum(W[:-1] * W[:-1])/2

    def __gradient(self, X, y, tau, W):
        """
        __loss gradient
        """
        return tau * W + np.sum(np.sign(np.sum(W * X, axis=1) - y) * np.transpose(X), axis=1)

if __name__ == "__main__":
    from sklearn.metrics import mean_squared_error, median_absolute_error
    from sklearn.cross_validation import train_test_split
    from sklearn.datasets import make_regression
    X, y = make_regression(n_samples=500, n_features=20, n_informative=10, noise=1.0)
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.5)
    a = SVM()
    a.fit(X, y, tau=1e-2)
    y_pred = a.predict(x_test)
    print(mean_squared_error(y_test, y_pred))
    print(median_absolute_error(y_test, y_pred))
    a = SVM()
    a.fit(X, y, tau=1e-2)
    y_pred = a.predict(x_test)
    print(mean_squared_error(y_test, y_pred))
    print(median_absolute_error(y_test, y_pred))
    from sklearn.svm import SVR
    a = SVR(kernel='linear')
    a.fit(X, y)
    y_pred = a.predict(x_test)
    print(mean_squared_error(y_test, y_pred))
    print(median_absolute_error(y_test, y_pred))
