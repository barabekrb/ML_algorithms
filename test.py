import numpy as np
import pandas as pd 
import sklearn.datasets as skl
from DecisionTreeModels.ClassificationTree import MyTreeClf


def rank(preds: pd.DataFrame):
        y = pd.DataFrame([[0],[1],[1],[1],[0]])
        preds.insert(2,'class',y[0][preds['ind']],True)
        r1 = sum(1/(preds[preds['class']==1]['ind']+1))
        r2 = sum(1/(preds['ind']+1))
        return r1/r2



def main():
    X, y = skl.make_classification(n_samples=150, n_features=5, n_informative=3, random_state=42)
    X = pd.DataFrame(X).round(2)
    y = pd.Series(y)
    X.columns = [f'col_{col}' for col in X.columns]

    trr = MyTreeClf(5,5,10)
    # c, v, ig = trr.get_best_split(X, y)
    # X1, y1 = X[X[c]<=v], y[X[c]<=v]
    # X2, y2 = X[X[c]>v], y[X[c]>v]

    # c, v, ig = trr.get_best_split(X1, y1)


    # X11, y11 = X1[X1[c]<=v], y1[X1[c]<=v]
    # X12, y12 = X1[X1[c]>v], y1[X1[c]>v]


    # c, v, ig = trr.get_best_split(X12, y12)

    # print(c, v)
    trr.fit(X, y)
    trr.print_tree()
    print(trr.leafs_cnt)

    
   

def schenon_enthropy(y_t, classes : np.array) -> float:
            s0 = 0.
            if y_t.size ==0:
                return 1e-12
            for cls in classes:
                p_i = len(y_t[y_t == cls])/len(y_t)
                if p_i!=0:
                    s0 -= p_i * np.log2(p_i)  
                else:
                    s0 -= 1e-12
            return s0

main()
# pr = pd.DataFrame([[0,3],[1,4],[2,6],[3,9],[4,10]], columns=['ind', 'norm'])
# y = pd.DataFrame([[0],[1],[1],[1],[0]])
# print(y[0][np.array([1,2,3])])