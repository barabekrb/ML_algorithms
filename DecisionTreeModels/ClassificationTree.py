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
        

class MyTreeClf:
    def __init__(self, max_depth: int = 5, min_samples_split: int = 2, max_leafs: int = 20, bins: int = None, criterion: str = 'entropy')->None:
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

    
    def __str__(self)->str:
        return f"MyTreeClf class: max_depth={self.max_depth}, min_samples_split={self.min_samples_split}, max_leafs={self.max_leafs}"
    

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


    def fit(self, x_:pd.DataFrame, y_:pd.Series):
        if self.max_leafs<2:
            self.max_leafs = 2
        if self.min_samples_split<2:
            self.min_samples_split = 2
        self.make_splits(x_)
        self.sample_len = len(y_)
        for col in x_.columns:
            self.fi[col] = 0
        self.tree = self.fitRecurtion(x_, y_, 1, "main")

    def fitRecurtion(self, x_:pd.DataFrame, y_:pd.Series, depth: int, l_or_r: str):
        if depth>self.max_depth:
            self.leafs_cnt+=1
            return Leaf(sum(y_)/len(y_), l_or_r, depth)
        if self.bins!=None:
            if self.is_in(x_):
                self.leafs_cnt+=1
                return Leaf(sum(y_)/len(y_), l_or_r, depth)
        self.potential_leafs+=2
        col, val, ig = self.get_best_split(x_, y_)
        xL_ = x_[x_[col]<=val]
        yL = y_[x_[col]<=val]
        xR_ = x_[x_[col]>val]
        yR = y_[x_[col]>val]
        lif = (len(yL.unique())>1) and (len(yL)>=self.min_samples_split)
        rif = (len(yR.unique())>1) and (len(yR)>=self.min_samples_split)

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
                    lTree = Leaf(sum(yL)/len(yL), "left", depth+1)
            if rif:
                self.potential_leafs-=1
                rTree = self.fitRecurtion(xR_, yR, depth+1, "right")
            elif not rif:
                self.potential_leafs-=1
                self.leafs_cnt+=1
                if len(yR)==0:
                    rTree = Leaf(0, "right", depth+1)
                else:
                    rTree = Leaf(sum(yR)/len(yR), "right", depth+1)
        else:
            self.potential_leafs-=2
            self.leafs_cnt+=1
            return Leaf(sum(y_)/len(y_), l_or_r, depth)
        

        self.fi[col] = self.fi[col] + len(y_)/self.sample_len * ig
        return Tree(val, depth, ig, len(y_), col, lTree, rTree)


    def is_in(self, X:pd.DataFrame):
        for col, splts in self.splits.items():
            x_min = X[col].min()
            x_max = X[col].max()
            for splt in splts:
                if x_min<=splt<x_max:
                    return False
        return True
                


    def get_best_split(self, x_: pd.DataFrame, y_: pd.Series):

        def schenon_enthropy(y_t, classes : np.array) -> float:
            s0 = 0.
            if len(y_t) == 0:
                return 1e-12
            for cls in classes:
                p_i = len(y_t[y_t == cls])/len(y_t)
                if p_i!=0:
                    s0 -= p_i * np.log2(p_i)  
                else:
                    s0 -= 1e-12
            return s0
        
        def gini(y_t, classes: np.array) -> float:
            s0 = 1.
            if len(y_t) == 0:
                return 1e-12
            for cls in classes:
                p_i = len(y_t[y_t == cls])/len(y_t)
                s0 -= p_i * p_i
            return s0
        
        funcs = {'entropy': schenon_enthropy, 'gini': gini}
        classes = y_.unique()
        s0 = funcs[self.criterion](y_, classes)
        bestIG = -np.inf
        bestCol = ""
        bestSplit = 0    
        for col in x_.columns:
            for splt in self.splits[col]:
                left_part = y_[x_[col]<=splt]
                right_part = y_[x_[col]>splt]
                ig = s0 - len(left_part)/len(y_)*funcs[self.criterion](left_part, classes) - len(right_part)/len(y_)*funcs[self.criterion](right_part, classes)
                if ig > bestIG:
                    bestIG = ig
                    bestCol = col
                    bestSplit = splt
        return bestCol, bestSplit, bestIG
    

    def print_tree(self):
        print(self.tree.prnt())


    def predict(self, X:pd.DataFrame):
        res = []
        for _, r in X.iterrows():
            prob = self.tree.predict_proba(r)
            res.append(int(prob>0.5))
        return res


    def predict_proba(self, X:pd.DataFrame):
        res = []
        for _, r in X.iterrows():
            prob = self.tree.predict_proba(r)
            res.append(prob)
        return res