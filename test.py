import pandas as pd
import numpy as np


def rank(preds: pd.DataFrame):
        y = pd.DataFrame([[0],[1],[1],[1],[0]])
        preds.insert(2,'class',y[0][preds['ind']],True)
        r1 = sum(1/(preds[preds['class']==1]['ind']+1))
        r2 = sum(1/(preds['ind']+1))
        return r1/r2



def main():
   print(2**30)
   

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