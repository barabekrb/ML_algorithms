import numpy as np
import pandas as pd

class MyKNNReg:
    def euclidean(x1_, x2_ : np.array):
        return np.linalg.norm(x1_ - x2_)

    
    def chebyshev(x1_, x2_ : np.array):
        return max(np.abs(x1_ - x2_))


    def manhattan(x1_, x2_ : np.array):
        return sum(np.abs(x1_ - x2_))


    def cosine(x1_, x2_ : np.array):
        return 1 - (x1_ @ x2_)/(np.linalg.norm(x1_)*np.linalg.norm(x2_))


    metric_func = {
        'euclidean': euclidean,
        'chebyshev': chebyshev,
        'manhattan': manhattan,
        'cosine':   cosine,
    }


    def rank(pred: pd.DataFrame):
        sumRank = sum(1/pred['rank'])
        w_ = np.array([(1/i)/sumRank for i in pred['rank']])


        return sum(w_ * pred['y'].to_numpy()) 


    def distance(pred: pd.DataFrame):
        sumRank = sum(1/pred['dist'])
        w_ = np.array([(1/i)/sumRank for i in pred['dist']])


        return sum(w_ * pred['y'].to_numpy()) 

    def uniform(pred: pd.DataFrame):
        return pred["y"].sum()/len(pred)

    weight_prob = {
        'rank': rank,
        'distance': distance,
        'uniform': uniform,
    }

    def __init__(self, k:int = 3, metric: str = 'euclidean', weight: str = 'uniform') -> None:
        self.k = k
        self.train_size = None
        self.x_ = pd.DataFrame
        self.y_ = pd.Series
        self.metric = self.metric_func[metric]
        self.weight = self.weight_prob[weight]


    def __str__(self) -> str:
        return f"MyKNNReg class: k={self.k}"
    

    def fit(self, x_ : pd.DataFrame, y_: pd.Series) -> None:
        self.x_ = x_
        self.y_ = y_
        self.train_size = x_.shape

    
    def predict(self, x_: pd.DataFrame)->np.array:
        predict_m = []
        for _, xf in x_.iterrows():
            k_features = np.array([self.metric(xf.to_numpy(), xt.to_numpy()) for _, xt in self.x_.iterrows()])
            pred_y = pd.DataFrame({'dist' : k_features, 'y' : self.y_.to_numpy()})
            pred_y = pred_y.sort_values(by=['dist'],ascending=True,ignore_index=True)[:self.k]
            pred_y.insert(1, 'rank', np.arange(1,self.k + 1))
            y_p = self.weight(pred_y)
            predict_m.append(y_p)
        return np.array(predict_m)
