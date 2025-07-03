import numpy as np
import pandas as pd 
import random
from DecisionTreeModels.RegressionTree import MyTreeReg



class MyForestReg:
    def __init__(self, n_estimators:int = 10, 
                 max_features:float = 0.5, 
                 max_samples: float = 0.5, 
                 random_state:int = 42,
                 max_depth: int = 5,
                 min_samples_split: int = 2,
                 max_leafs: int = 20,
                 bins: int = 16,
                 oob_score: str = None):
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.max_samples = max_samples
        self.random_state = random_state
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_leafs = max_leafs
        self.bins = bins
        self.leafs_cnt = 0
        self.trees = []
        self.fi = dict({})



    def __str__(self):
        return f"MyForestReg class: n_estimators={self.n_estimators}, max_features={self.max_features}, max_samples={self.max_samples}, max_depth={self.max_depth}, min_samples_split={self.min_samples_split}, max_leafs={self.max_leafs}, bins={self.bins}, random_state={self.random_state}"
    

    def fit(self, X:pd.DataFrame, y:pd.Series):
        random.seed(self.random_state)
        cols_smpl_cnt = round(self.max_features * X.shape[1])
        rows_smpl_cnt = round(self.max_samples * X.shape[0])
        for col in X.columns:
            self.fi[col] = 0
        for i in range(self.n_estimators):
            cols_idx = random.sample(list(X.columns), cols_smpl_cnt)
            rows_idx = random.sample(range(X.shape[0]), rows_smpl_cnt)
            xt = X[cols_idx].iloc[rows_idx]
            yt = y.iloc[rows_idx]
            tr = MyTreeReg(self.max_depth, self.min_samples_split, self.max_leafs, self.bins, sample_len=len(y))
            tr.fit(xt, yt)
            self.leafs_cnt +=tr.leafs_cnt
            self.trees.append(tr)
        for tr in self.trees:
            for col, val in tr.fi.items():
                self.fi[col] += val

    def predict(self, X:pd.DataFrame):
        preds = np.zeros(X.shape[0])
        for tr in self.trees:
            preds += np.array(tr.predict(X))/len(tr)
        
        return preds

