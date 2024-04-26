import numpy as np
import pandas as pd
from typing import Optional, Union, Callable
import random

def accuracy(y_, yp):
    return np.sum(y_==yp)/y_.shape[0]


def precision(y_, yp):
    return np.sum(y_[yp==1])/len(yp[yp==1])


def recall(y_, yp):
    return np.sum(y_[yp==1])/len(y_[y_==1])


def f1(y_, yp):
    pres = precision(y_, yp)
    rec = recall(y_, yp)
    return (2) * (pres*rec)/(pres + rec)


def roc_auc(y_, yp):
    n = np.sum(y_[y_== 1])
    p = y_.shape[0] - n
    ans = 0
    ones = 0
    z_p = pd.DataFrame(np.concatenate((y_.to_numpy().reshape((y_.shape[0],1)), yp.reshape((yp.shape[0],1))),axis=1), columns=['class', 'prob'])
    z_p.sort_values(by=["prob"],ascending=False, inplace=True, ignore_index=True)
    for ind, row in z_p.iterrows():
        if row["class"]==1:
            continue
        if row["class"]==0:
            eq = z_p[:ind]["prob"]==row["prob"]
            uneq = z_p[:ind]["prob"]!=row["prob"]
            ans += z_p[:ind]["class"][z_p["class"]==1].where(uneq).count()
            ans += 0.5 * z_p[:ind]["class"][z_p["class"]==1].where(eq).count()
    return ans/n/p





class LogReg():

    metrics = { 'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'roc_auc': roc_auc,
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
        return f"LogReg class: n_iter={self.n_iter}, learning_rate={self.learning_rate}"
    

    def fit(self, X_, y_, verbose=False):
        X_ = np.concatenate((np.array([np.ones(X_.shape[0])]).T, X_), axis=1)
        self.weights = np.ones(X_.shape[1])
        if not self.sgd_sample:
            for i in range(1,self.n_iter+1):
                y_p = 1/(1 + np.exp(-1 * (X_ @ self.weights.T)))
                er = y_p - y_
                d_w = 1/y_.shape[0] * (er @ X_)
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
                y_p = 1/(1 + np.exp(-1 * (X_b @ self.weights.T)))
                er = y_p - y_b
                d_w = 1/self.sgd_sample * np.dot(er,X_b)
                if self.reg == 'l1' or self.reg == 'elasticnet':
                    d_w+= self.l1_coef * np.sign(self.weights)
                if self.reg == 'l2' or self.reg == 'elasticnet':
                    d_w+= self.l2_coef * 2 * self.weights
                if self.lr_f:
                    self.weights = self.weights - self.learning_rate(i) * d_w
                else:
                    self.weights = self.weights - self.learning_rate * d_w
        
        if self.metric not in [None, 'roc_auc']:
            pred = 1/(1 + np.exp(-1 * (X_ @ self.weights.T)))
            self.metric_value = self.metrics[self.metric](y_, (pred>0.5).astype(int))
        if self.metric == 'roc_auc':
            self.metric_value = self.metrics[self.metric](y_, 1/(1 + np.exp(-1 * (X_ @ self.weights.T))))
   
            
    def get_coef(self):
        return self.weights[1:]


    def predict_proba(self, X_):
        X_ = np.concatenate((np.array([np.ones(X_.shape[0])]).T, X_), axis=1)
        return 1/(1 + np.exp(-1 * (X_ @ self.weights.T)))
    
    
    def predict(self, X_):
        X_ = np.concatenate((np.array([np.ones(X_.shape[0])]).T, X_), axis=1)
        pred = 1/(1 + np.exp(-1 * (X_ @ self.weights.T)))
        return (pred>0.5).astype(int)
    

    def get_best_score(self):
        return self.metric_value