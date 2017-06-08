# create weighted ensemble for submission
import pandas as pd

df1 = pd.read_csv("../ensemble/XGB_leaky_v5.csv")    
df2 = pd.read_csv("../ensemble/xgb_magic_v6.csv")   
df3 = pd.read_csv("../ensemble/lstm_v2.csv")  

df4 = pd.read_csv("../ensemble/clf_xgb_tree.csv")

result = pd.DataFrame()
result['test_id'] = df1['test_id']
result['is_duplicate'] = 0.36 * df1['is_duplicate'] + 0.22 * df2['is_duplicate'] + 0.41 * df3['is_duplicate']
+ 0.01 * df4['is_duplicate']
result.to_csv('../predictions/final_5.csv', index = False)  


