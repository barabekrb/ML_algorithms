import numpy as np
import pandas as pd

class MyTreeClf:
    def __init__(self, max_depth: int = 5, min_samples_split: int = 2, max_leafs: int = 20)->None:
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_leafs = max_leafs

    
    def __str__(self)->str:
        return f"MyTreeClf class: max_depth={self.max_depth}, min_samples_split={self.min_samples_split}, max_leafs={self.max_leafs}"
    

    


    def get_best_split( x_: pd.DataFrame, y_: pd.Series):
        def schenon_enthropy(y_t, classes : np.array) -> float:
            s0 = 0.
            if y_t.size == 0:
                return 1e-12
            for cls in classes:
                p_i = len(y_t[y_t == cls])/len(y_t)
                if p_i!=0:
                    s0 -= p_i * np.log2(p_i)  
                else:
                    s0 -= 1e-12
            return s0
        splt_cols = pd.DataFrame({'col':[], 'splt':[], 'ig':[]})
        for col in x_.columns:
            uniq_col_vals = np.sort(x_[col].unique())
            split_vals = ((np.append(uniq_col_vals[1:], 0)+uniq_col_vals)/2) [:-1]
            classes = y_.unique()
            s0 = schenon_enthropy(y_, classes)
            IGs = pd.DataFrame({'splt':[], 'ig':[]})
            for splt in split_vals:
                left_part = y_[x_[col]<=splt]
                right_part = y_[x_[col]>splt]
                ig = s0 - len(left_part)/len(y_)*schenon_enthropy(left_part, classes) - len(right_part)/len(y_)*schenon_enthropy(right_part, classes)
                IGs = pd.concat([pd.DataFrame([[splt, ig]], columns=IGs.columns), IGs], ignore_index=True)
            splt_cols = pd.concat([pd.DataFrame([[col, IGs.iloc[IGs['ig'].idxmax()]['splt'], IGs['ig'].idxmax()]], columns=splt_cols.columns), splt_cols], ignore_index=True)
            splt_cols.add({'col':col, 'splt':IGs.iloc[IGs['ig'].idxmax()]['splt'], 'ig':IGs['ig'].max()})    
        return splt_cols['col'].iloc[splt_cols['ig'].idxmax()], splt_cols['splt'].iloc[splt_cols['ig'].idxmax()], splt_cols['ig'].max()