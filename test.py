import numpy as np
import pandas as pd 
import sklearn.datasets as skl
from DecisionTreeModels.ClassificationTree import MyTreeClf
from DecisionTreeModels.RegressionTree import MyTreeReg

from sklearn.datasets import load_diabetes




def rank(preds: pd.DataFrame):
        y = pd.DataFrame([[0],[1],[1],[1],[0]])
        preds.insert(2,'class',y[0][preds['ind']],True)
        r1 = sum(1/(preds[preds['class']==1]['ind']+1))
        r2 = sum(1/(preds['ind']+1))
        return r1/r2



def main():
    # df = pd.read_csv('banknote+authentication.zip', header=None)
    # df.columns = ['variance', 'skewness', 'curtosis', 'entropy', 'target']
    # X, y = df.iloc[:,:4], df['target']
    # tr = MyTreeClf(15, 20, 30, bins=6)
    
    # # print(tr.tree.sum_leaf())
    # test_all(X, y)

    data = load_diabetes(as_frame=True)
    X, y = data['data'], data['target'] 
    # tr = MyTreeReg(1,1,2)
    # tr.fit(X, y)
    # tr.print_tree()

    test_all2(X, y)
          


def test_all2(X, y):
    some_testes = [
        (1,1,2),
        (3,2,5),
        (5,100,10),
        (4,50,17),
        (10,40,21),
        (15, 35, 30),
]

    some_results = [
        (2, 303.138024),
        (5, 813.992098),
        (7, 1143.916064),
        (11, 1808.268095),
        (21, 3303.816014),
        (27, 4352.894213),
    ]

    res_l = []
    res_sum_l = [] 
    for i, par in enumerate(some_testes):
          max_d, min_s, max_l = par
          tr = MyTreeReg(max_depth=max_d, min_samples_split=min_s, max_leafs=max_l)
          tr.fit(X, y)
          res_l.append([tr.leafs_cnt, some_results[i][0]])
          res_sum_l.append(tr.tree.sum_leaf() - some_results[i][1])
    

    print(res_l)
    print(res_sum_l)




def test_all(X, y):
    some_testes = [
          (1,1,2,8, 'gini'),
          (3,2,5,None, 'gini'),
          (5,200,10,4, 'entropy'),
          (4,100,17,16, 'gini'),
          (10,40,21,10, 'gini'),
          (15, 20, 30, 6, 'gini'),
    ]

    some_results = [
          (2, 0.981148),
          (5, 2.799994),
          (10, 5.020575),
          (11, 5.200813),
          (21, 10.198869),
          (27, 12.412269),
    ]
    res_l = []
    res_sum_l = [] 
    for i, par in enumerate(some_testes):
          max_d, min_s, max_l, bins, crit = par
          tr = MyTreeClf(max_depth=max_d, min_samples_split=min_s, max_leafs=max_l, bins=bins, criterion=crit)
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