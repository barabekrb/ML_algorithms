import numpy as np
import pandas as pd

from typing import Optional, Union

class Leaf:
    def __init__(self, leaf_cnt, l_or_r, depth):
        self.l_or_r = l_or_r
        self.leaf_cnt = leaf_cnt
        self.depth = depth
        self.ig = 0
    

    def prnt(self):
        return self.depth * "\t" + f"{self.l_or_r} leaf score = {self.leaf_cnt}\n"
    

    def predict_proba(self, x):
        return self.leaf_cnt

class Tree:
    def __init__(self, root, depth, ig, leny, feature: str, left: Optional["Tree | Leaf"] = None, rigth: Optional["Tree | Leaf"] = None):
        self.depth = depth
        self.feature = feature
        self.root = root
        self.left = left
        self.right = rigth
        self.ig  = ig
        self.leny = leny

    def prnt(self)->str:
        tree =  self.depth*"\t" + str(self.feature) + ">" +  str(self.root) + f"    ig = {self.ig} |  len = {self.leny}" +"\n"
        if self.left != None:
            tree += self.left.prnt()
        if self.right != None:
            tree += self.right.prnt()

        return tree
    
    def sum_leaf(self):
        s = 0
        if self.left.__class__.__name__=="Leaf":
            s+=self.left.leaf_cnt
        else:
            s+=self.left.sum_leaf()
        if self.right.__class__.__name__=="Leaf":
            s+=self.right.leaf_cnt
        else:
            s+=self.right.sum_leaf()
        return s


    def predict_proba(self, x):
        if x[self.feature]<=self.root:
            return self.left.predict_proba(x)
        else:
            return self.right.predict_proba(x)
    


class MyTreeReg:
    def __init__(self, max_depth:int = 5, min_samples_split:int = 2, max_leafs:int = 20, bins: int = None, criterion: str = 'entropy'):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_leafs = max_leafs
        self.leafs_cnt = 0
        self.potential_leafs = 0
        self.tree = None
        self.bins = bins
        self.splits = None
        self.criterion = criterion
        self.fi = dict({})
        self.sample_len = 0


    def __str__(self):
        return f"MyTreeReg class: max_depth={self.max_depth}, min_samples_split={self.min_samples_split}, max_leafs={self.max_leafs}" 
    

    def make_splits(self, X:pd.DataFrame):
        splits = dict({})
        if (self.bins==None) or (len(X)<=self.bins-1):
            for col in X.columns:
                uniq_col_vals = np.sort(X[col].unique())
                splits[col] = ((np.append(uniq_col_vals[1:], 0) + uniq_col_vals)/2) [:-1]
            self.splits = splits
        else:
            for col in X.columns:
                _, borders = np.histogram(X[col], self.bins)
                splits[col] = borders[1:-1]
            self.splits =  splits

    def is_in(self, X:pd.DataFrame):
        for col, splts in self.splits.items():
            x_min = X[col].min()
            x_max = X[col].max()
            for splt in splts:
                if x_min<=splt<x_max:
                    return False
        return True

    def fit(self, X:pd.DataFrame, y:pd.Series):
        if self.max_leafs<2:
            self.max_leafs = 2
        if self.min_samples_split<2:
            self.min_samples_split = 2
        self.make_splits(X)
        self.tree = self.fitRecurtion(X, y, 1, "main")
        

    def fitRecurtion(self, x_:pd.DataFrame, y_:pd.Series, depth: int, l_or_r: str):
        if depth>self.max_depth:
            self.leafs_cnt+=1
            return Leaf(y_.mean(), l_or_r, depth)
        if self.bins!=None:
            if self.is_in(x_):
                self.leafs_cnt+=1
                return Leaf(y_.mean(), l_or_r, depth)
        self.potential_leafs+=2
        col, val, ig = self.get_best_split(x_, y_)
        xL_ = x_[x_[col]<=val]
        yL = y_[x_[col]<=val]
        xR_ = x_[x_[col]>val]
        yR = y_[x_[col]>val]
        lif = (len(yL)>=self.min_samples_split)
        rif = (len(yR)>=self.min_samples_split)

        if self.leafs_cnt + self.potential_leafs <= self.max_leafs:
            if lif:
                self.potential_leafs-=1
                lTree = self.fitRecurtion(xL_, yL, depth+1, "left")
            elif not lif:
                self.potential_leafs-=1
                self.leafs_cnt+=1
                if len(yL)==0:
                    lTree = Leaf(0, "left", depth+1)
                else:
                    lTree = Leaf(yL.mean(), "left", depth+1)
            if rif:
                self.potential_leafs-=1
                rTree = self.fitRecurtion(xR_, yR, depth+1, "right")
            elif not rif:
                self.potential_leafs-=1
                self.leafs_cnt+=1
                if len(yR)==0:
                    rTree = Leaf(0, "right", depth+1)
                else:
                    rTree = Leaf(yR.mean(), "right", depth+1)
        else:
            self.potential_leafs-=2
            self.leafs_cnt+=1
            return Leaf(y_.mean(), l_or_r, depth)
        

        # self.fi[col] = self.fi[col] + len(y_)/self.sample_len * ig
        return Tree(val, depth, ig, len(y_), col, lTree, rTree)


    def get_best_split(self, X:pd.DataFrame, y:pd.Series):

        def mse(ys:np.array):
            if len(ys)==0:
                return 0 
            return 1/len(ys) * sum((ys - ys.mean())**2)
        
        s0 = mse(y)
        bestIg = -np.inf
        bestCol = ""
        bestSplt = 0

        for col in X.columns:
            for splt in self.splits[col]:
                left_part = y[X[col]<=splt].to_numpy()
                right_part = y[X[col]>splt].to_numpy()
                ig = s0 - len(left_part)/len(y)*mse(left_part) - len(right_part)/len(y)*mse(right_part)
                if ig > bestIg:
                    bestSplt = splt
                    bestCol = col
                    bestIg = ig
        
        return bestCol, bestSplt, bestIg
    

    def print_tree(self):
        print(self.tree.prnt())

    
    def predict(self, X:pd.DataFrame):
        res = []
        for _, r in X.iterrows():
            avg = self.tree.predict_proba(r)
            res.append(avg)
        return res
