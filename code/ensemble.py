# create weighted ensemble for submission
import pandas as pd

#tq = pd.read_csv("../ensemble/merged_v1_tq.csv")    # LB = 0.15451
#hw = pd.read_csv("../ensemble/ensemble_7_hw.csv")   # LB = 0.27795
#gaj = pd.read_csv("../ensemble/test.proba.[Feat@basic_nonlinear_20170526]_[Learner@clf_xgb_tree]_[Id@100]_gaj.csv") # LB = 0.30469
#wxk = pd.read_csv("../ensemble/sub_wxk.csv")        # LB = 0.35185

#result = pd.DataFrame()
#result['test_id'] = tq['test_id']
#result['is_duplicate'] = 0.7 * tq['is_duplicate'] + 0.1 * hw['is_duplicate'] + 0.1 * gaj['is_duplicate'] +  0.1 * wxk['is_duplicate'] 
#result.to_csv('../ensemble/result_20170530.csv', index = False)  

tq1 = pd.read_csv("../predictions/XGB_leaky_v5.csv")    
tq2 = pd.read_csv("../predictions/xgb_magic_v5.csv")   
tq3 = pd.read_csv("../predictions/lstm_v1.csv")  

result = pd.DataFrame()
result['test_id'] = tq1['test_id']
result['is_duplicate'] = 0.5 * tq1['is_duplicate'] + 0.4 * tq2['is_duplicate'] + 0.1 * tq3['is_duplicate'] 
result.to_csv('../predictions/ensemble_20170604.csv', index = False)  


