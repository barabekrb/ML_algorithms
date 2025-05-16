import numpy as np
import pandas as pd 

class MyKNNClf:


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
        first_weght = sum(1/pred[pred["class"]==1]["rank"])/sum(1/pred["rank"])
        zero_weight = sum(1/pred[pred["class"]==0]["rank"])/sum(1/pred["rank"])

        return int(first_weght>=zero_weight), first_weght


    def distance(pred: pd.DataFrame):
        first_weght = sum(1/pred[pred["class"]==1]["dist"])/sum(1/pred["dist"])
        zero_weight = sum(1/pred[pred["class"]==0]["dist"])/sum(1/pred["dist"])

        return int(first_weght>=zero_weight), first_weght

    def uniform(pred: pd.DataFrame):
        first_prob = pred["class"].sum()/len(pred)

        return int(first_prob>=0.5), first_prob

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
        return f"MyKNNClf class: k={self.k}"
    
    def fit(self, x_:pd.DataFrame, y_:pd.Series) -> None:
        self.x_ = x_
        self.y_ = y_
        self.train_size = x_.shape


    def predict(self, x_features:pd.DataFrame) -> np.array:
        predict_m = []
        for _, xf in x_features.iterrows():
            k_features = np.array([self.metric(xf.to_numpy(), xt.to_numpy()) for _, xt in self.x_.iterrows()])
            pred_y = pd.DataFrame({'dist' : k_features, 'class' : self.y_.to_numpy()})
            pred_y = pred_y.sort_values(by=['dist'],ascending=True,ignore_index=True)[:self.k]
            pred_y.insert(1, 'rank', np.arange(1,self.k + 1))
            pred_cls, _ = self.weight(pred_y)
            predict_m.append(pred_cls)
        return np.array(predict_m)

    def predict_proba(self, x_features:pd.DataFrame) -> np.array:
        predict_m = []
        for _, xf in x_features.iterrows():
            k_features = np.array([self.metric(xf.to_numpy(), xt.to_numpy()) for _, xt in self.x_.iterrows()])
            pred_y = pd.DataFrame({'dist' : k_features, 'class' : self.y_.to_numpy()})
            pred_y = pred_y.sort_values(by=['dist'],ascending=True, ignore_index=True)[:self.k]
            pred_y.insert(1, 'rank', np.arange(1,self.k + 1))
            _, prob_frs = self.weight(pred_y)
            predict_m.append(prob_frs)
        return np.array(predict_m)