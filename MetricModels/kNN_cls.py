import pandas as pd
import numpy as np


class kNN_cls():


    def euclidean(x1, x2):
        return np.linalg.norm(x1 - x2)
    

    def chebyshev(x1, x2):
        return np.max(np.abs(x1 - x2))
    

    def manhattan(x1, x2):
        return np.sum(np.abs(x1 - x2))


    def cosine(x1, x2):
        return 1 - (x1 @ x2.T)/(np.linalg.norm(x1) * np.linalg.norm(x2))

    metrics = {
        'euclidean': euclidean,
        'chebyshev': chebyshev,
        'manhattan': manhattan,
        'cosine': cosine

    }
    

    def __init__(self, k: int = 3, metric: str = 'euclidean', weight: str = 'unoform') -> None:
        self.k = k
        self.train_size = None
        self.X = pd.DataFrame([])
        self.y = pd.Series([])
        self.metric = metric
        self.weight = weight
        
    

    def __str__(self) -> str:
        return f"kNN_cls class: k = {self.k}"
    

    def fit(self, X_, y_):
        self.X = X_.copy()
        self.y = y_.copy()
        self.train_size = self.X.shape

    
    def predict(self, X_):
        y_s = []
        if self.weight == 'uniform':
            for ind, row in X_.iterrows():
                cls_pred = []
                for j, rs in self.X.iterrows():
                    cls_pred.append([j, self.metrics[self.metric](row , rs)])
                cls_pred = pd.DataFrame(cls_pred, columns=['ind', 'norm'])
                cls_pred.sort_values(by=["norm"], ascending=True, inplace=True, ignore_index=True)
                y_s.append(self.y[cls_pred[:self.k]['ind']].mode().sort_values(ascending=False, ignore_index=True)[0])
        if self.weight == 'rank':
            pass
        if self.weight == 'distance':
            pass
        return np.array(y_s)

    
    def predict_proba(self, X_):
        y_s = []
        if self.weight == 'uniform':
            for ind, row in X_.iterrows():
                cls_pred = []
                for j, rs in self.X.iterrows():
                    cls_pred.append([j, self.metrics[self.metric](row , rs)])
                cls_pred = pd.DataFrame(cls_pred, columns=['ind', 'norm'])
                cls_pred.sort_values(by=["norm"], ascending=True, inplace=True, ignore_index=True)
                y_s.append(self.y[cls_pred[:self.k]['ind']].sum() / self.k)
        if self.weight == 'rank':
            for _, row in X_.iterrows():
                cls_pred = []
                for j, rs in self.X.iterrows():
                    cls_pred.append([j, self.metrics[self.metric](row , rs)])
                cls_pred = pd.DataFrame(cls_pred, columns=['ind', 'norm'])
                cls_pred.sort_values(by=["norm"], ascending=True, inplace=True, ignore_index=True)
                y_r = self.y[cls_pred['ind']]
                r1 = sum( 1 / (cls_pred['ind'][y_r==1] + 1))
                r2 = sum(1 / cls_pred['ind'] + 1)
                y_s.append(r1/r2)
        if self.weight == 'distance':
            pass
        return np.array(y_s)
    

    def rank(self, preds: pd.DataFrame):
        preds.insert(2,'class',self.y[0][preds['ind']],True)
        r1 = sum(1/(preds[preds['class']==1]['ind']+1))
        r2 = sum(1/(preds['ind']+1))
        return r1/r2
        
    