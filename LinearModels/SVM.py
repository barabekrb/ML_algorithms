import numpy as np
import pandas as pd
import random
from typing import Optional


class MySVM:
    def __init__(self, n_iter:int = 10, learning_rate:int = 0.001, C:float = 1., sgd_sample: Optional["int | float"] = None, random_state:int = 42):
        self.n_iter = n_iter
        self.learning_rate = learning_rate
        self.weights = None
        self.b = None   
        self.c = C
        self.sgd_sample = sgd_sample
        self.random_state = random_state

    def __str__(self):
        return f"MySVM class: n_iter={self.n_iter}, learning_rate={self.learning_rate}"
    

    def fit(self,X:pd.DataFrame, y:pd.Series, verbose:bool = False):
        random.seed(self.random_state)
        y.replace(0, -1, inplace=True)
        self.weights = np.ones(X.shape[1])
        self.b = 1
        if self.sgd_sample and self.sgd_sample < 1:
            self.sgd_sample = round(self.sgd_sample * X.shape[0]) 
        loss = self.weights @ self.weights.T + self.c / len(y) * sum(np.where(1 - y * ( self.weights @ X.T + self.b), 0, 1 - y * ( self.weights @ X.T + self.b)))
        if verbose:
            print(f"start | loss: {loss}")
        for i in range(self.n_iter):
            if self.sgd_sample:
                sample_row_idx = random.sample(range(X.shape[0]), self.sgd_sample)
                for j in sample_row_idx:
                    if y[j]*(self.weights @ X.iloc[j].T + self.b) >= 1:
                        gradW = 2 * self.weights
                        gradB = 0
                    else:
                        gradW = 2 * self.weights - self.c * y[j] *  X[j]
                        gradB = - self.c * y[j]
                    self.weights = self.weights - self.learning_rate * gradW
                    self.b = self.b - self.learning_rate * gradB
                    loss = self.weights @ self.weights.T + self.c / len(y) * sum(np.where(1 - y * ( self.weights @ X.T + self.b), 0, 1 - y * ( self.weights @ X.T + self.b)))
                    if verbose:
                        if i%verbose==0 and i>0:
                            print(f"{i} | loss: {loss}")
            else:
                for j, r in X.iterrows():
                    if y[j]*(self.weights @ r.T + self.b) >= 1:
                        gradW = 2 * self.weights
                        gradB = 0
                    else:
                        gradW = 2 * self.weights - self.c * y[j] * r
                        gradB = - self.c * y[j]
                    self.weights = self.weights - self.learning_rate * gradW
                    self.b = self.b - self.learning_rate * gradB
                    loss = self.weights @ self.weights.T + self.c / len(y) * sum(np.where(1 - y * ( self.weights @ X.T + self.b), 0, 1 - y * ( self.weights @ X.T + self.b)))
                    if verbose:
                        if i%verbose==0 and i>0:
                            print(f"{i} | loss: {loss}")



    def get_coef(self):
        return self.weights, self.b
    

    def predict(self, X:pd.DataFrame):
        y = np.sign(self.weights @ X.T + self.b)
        return np.int64(np.where(y < 0, 0, y))