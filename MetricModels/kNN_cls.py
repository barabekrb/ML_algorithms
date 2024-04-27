import pandas as pd
import numpy as np


class kNN_cls():
    

    def __init__(self, k: int = 3) -> None:
        self.k = k
        self.train_size = None
        self.X = pd.DataFrame([])
        self.y = pd.Series([])
        
    

    def __str__(self) -> str:
        return f"kNN_cls class: k = {self.k}"
    

    def fit(self, X_, y_):
        self.X = X_.copy()
        self.y = y_.copy()
        self.train_size = self.X.shape

    
    def predict(self, X_):
        y_s = []
        for ind, row in X_.iterrows():
            cls_pred = []
            for j, rs in self.X.iterrows():
                cls_pred.append([j, np.linalg.norm(row - rs)])
            cls_pred = pd.DataFrame(cls_pred, columns=['ind', 'norm'])
            cls_pred.sort_values(by=["norm"], ascending=True, inplace=True, ignore_index=True)
            y_s.append(self.y[cls_pred[:self.k]['ind']].mode().sort_values(ascending=False, ignore_index=True)[0])
        return np.array(y_s)

    
    def predict_proba(self, X_):
        y_s = []
        for ind, row in X_.iterrows():
            cls_pred = []
            for j, rs in self.X.iterrows():
                cls_pred.append([j, np.linalg.norm(row - rs)])
            cls_pred = pd.DataFrame(cls_pred, columns=['ind', 'norm'])
            cls_pred.sort_values(by=["norm"], ascending=True, inplace=True, ignore_index=True)
            y_s.append(self.y[cls_pred[:self.k]['ind']].sum() / self.k)
        return np.array(y_s)