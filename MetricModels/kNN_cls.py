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