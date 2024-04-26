import numpy as np
import pandas as pd
import random
from typing import Optional, Union, Callable


def mae(y_, y_p):
    return 1/y_.shape[0] * np.sum(np.abs(y_ - y_p))


def mse(y_, y_p):
    return 1/y_.shape[0] * np.sum(np.square(y_ - y_p))


def rmse(y_, y_p):
    return np.sqrt(1/y_.shape[0] * np.sum(np.square(y_ - y_p)))


def mape(y_, y_p):
    return 100/y_.shape[0] * np.sum(np.abs((y_ - y_p)/y_))


def r2(y_, y_p):
    return 1 - (np.sum(np.square(y_ - y_p)))/(np.sum(np.square(y_ - np.mean(y_))))
    

class LinReg():

    metrics = { 'mae': mae,
                'mse': mse,
                'rmse': rmse,
                'mape': mape,
                'r2': r2,
                None : None
    }

    def __init__(self, n_iter: int = 100, learning_rate: Optional[Union[Callable, float]] = 0.1, metric: str = None, reg: str = None, 
                 l1_coef: float = 0.0, l2_coef: float = 0.0, sgd_sample : Optional[Union[int,float]]=None, random_state:int = 42) -> None:
        self.n_iter = n_iter
        self.learning_rate = learning_rate
        self.weights = None
        self.metric = metric
        self.metric_value = 0
        self.reg = reg
        self.l1_coef = l1_coef
        self.l2_coef = l2_coef
        self.lr_f = callable(self.learning_rate)
        self.sgd_sample = sgd_sample
        random.seed(random_state)
    

    def __str__(self) -> str:
        return f"LinReg class: n_iter={self.n_iter}, learning_rate={self.learning_rate}"
    

    def fit(self, X_, y_, verbose=False):
        X_ = np.concatenate((np.array([np.ones(X_.shape[0])]).T, X_), axis=1)
        self.weights = np.ones(X_.shape[1])
        if not self.sgd_sample:
            for i in range(1,self.n_iter+1):
                y_p = X_ @ self.weights.T
                er = y_p - y_
                d_w = 2/y_.shape[0] * (er @ X_)
                if self.reg == 'l1' or self.reg == 'elasticnet':
                    d_w+= self.l1_coef * np.sign(self.weights)
                if self.reg == 'l2' or self.reg == 'elasticnet':
                    d_w+= self.l2_coef * 2 * self.weights
                if self.lr_f:
                    self.weights = self.weights - self.learning_rate(i) * d_w
                else:
                    self.weights = self.weights - self.learning_rate * d_w
        else:
            if 0<self.sgd_sample<1:
                self.sgd_sample = int(self.sgd_sample * X_.shape[0])
            
            y_ = pd.Series(y_)
            X_ = pd.DataFrame(X_)
            for i in range(1,self.n_iter+1):
                sample_rows_idx = random.sample(range(X_.shape[0]), self.sgd_sample)
                X_b = X_.iloc[sample_rows_idx].to_numpy()
                y_b = y_.iloc[sample_rows_idx].to_numpy()
                y_p = X_b @ self.weights.T
                er = y_p - y_b
                d_w = 2/self.sgd_sample * np.dot(er,X_b)
                if self.reg == 'l1' or self.reg == 'elasticnet':
                    d_w+= self.l1_coef * np.sign(self.weights)
                if self.reg == 'l2' or self.reg == 'elasticnet':
                    d_w+= self.l2_coef * 2 * self.weights
                if self.lr_f:
                    self.weights = self.weights - self.learning_rate(i) * d_w
                else:
                    self.weights = self.weights - self.learning_rate * d_w
        
        if self.metric:
            self.metric_value = self.metrics[self.metric](y_, X_ @ self.weights.T)

    def get_coef(self):
        return self.weights[1:]


    def predict(self, X_):
        X_ = np.concatenate((np.array([np.ones(X_.shape[0])]).T, X_), axis=1)
        return X_ @ self.weights.T
    

    def get_best_score(self):
        return self.metric_value