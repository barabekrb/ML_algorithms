import numpy as np
import pandas as pd 

class MyKNNClf:
    def __init__(self, k:int = 3) -> None:
        self.k = k
        self.train_size = None
        self.x_ = pd.DataFrame
        self.y_ = pd.Series

    def __str__(self) -> str:
        return f"MyKNNClf class: k={self.k}"
    
    def fit(self, x_:pd.DataFrame, y_:pd.Series):
        self.x_ = x_
        self.y_ = y_
        self.train_size = (x_.shape[0], x_.shape[1])


    def predict(self, x_features:pd.DataFrame):
        predict_m = []
        for _, xf in x_features.iterrows():
            k_features = np.array([np.linalg.norm(xf.to_numpy() - xt.to_numpy()) for _, xt in self.x_.iterrows()])
            pred_y = pd.DataFrame({'prob' : k_features, 'class' : self.y_.to_numpy()})
            pred_y = pred_y.sort_values(by=['prob'],ascending=True,ignore_index=True)[:self.k]
            cls, cont = np.unique(np.append(pred_y,1), return_counts=True)
            predict_m.append(int(cls[np.argmax(cont)]))
        return np.array(predict_m)

    def predict_proba(self, x_features:pd.DataFrame):
        predict_m = []
        for _, xf in x_features.iterrows():
            k_features = np.array([np.linalg.norm(xf.to_numpy() - xt.to_numpy()) for _, xt in self.x_.iterrows()])
            pred_y = pd.DataFrame({'prob' : k_features, 'class' : self.y_.to_numpy()})
            pred_y = pred_y.sort_values(by=['prob'],ascending=True, ignore_index=True)[:self.k]
            predict_m.append(pred_y["class"].sum()/self.k)
        return np.array(predict_m)