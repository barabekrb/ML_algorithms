import pandas as pd
import numpy as np


def rank(preds: pd.DataFrame):
        y = pd.DataFrame([[0],[1],[1],[1],[0]])
        preds.insert(2,'class',y[0][preds['ind']],True)
        r1 = sum(1/(preds[preds['class']==1]['ind']+1))
        r2 = sum(1/(preds['ind']+1))
        return r1/r2



def main():
   
   cls, cont = np.unique(np.array([1,0,0,1]), return_counts=True)
   print(cls, cont, np.argmax(cont))



main()
# pr = pd.DataFrame([[0,3],[1,4],[2,6],[3,9],[4,10]], columns=['ind', 'norm'])
# y = pd.DataFrame([[0],[1],[1],[1],[0]])
# print(y[0][np.array([1,2,3])])