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
    df = pd.read_csv('banknote+authentication.zip', header=None)
    df.columns = ['variance', 'skewness', 'curtosis', 'entropy', 'target']
    X, y = df.iloc[:,:4], df['target']
    tr = MyTreeClf(3,2,5,None)
    tr.fit(X,y)
    # tr.print_tree()
    # print(tr.leafs_cnt)
    # print(tr.tree.sum_leaf())
    
          


def test_all(X):
    some_testes = [
          (1,1,2,8),
          (3,2,5,None),
          (5,200,10,4),
          (4,100,17,16),
          (10,40,21,10),
          (15, 20, 30, 6),
    ]

    some_results = [
          (2, 0.71033),
          (5, 2.916956),
          (10, 5.020575),
          (10, 5.85783),
          (19, 9.526468),
          (26, 12.025427),
    ]
    res_l = []
    res_sum_l = [] 
    for i, par in enumerate(some_testes):
          max_d, min_s, max_l, bins = par
          tr = MyTreeClf(max_d, min_s, max_l, bins)
          tr.fit(X, y)
          res_l.append([tr.leafs_cnt, some_results[i][0]])
          res_sum_l.append(tr.tree.sum_leaf() - some_results[i][1])
    

    print(res_l)
    print(res_sum_l)
    
   

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