# convert magic features from online to the format used by Quora_HD
import pandas as pd
import pickle

def _save(fname, data, protocol=4):
    # use protocol=4 to save files larger than 4GB
    with open(fname, "wb") as f:
        pickle.dump(data, f, protocol)
# magic features from 0.158 kernel
X_train = pd.read_pickle("../features/X_train.pkl")
X_test = pd.read_pickle("../features/x_test.pkl")
magic_xgb = pd.concat([X_train, X_test], axis = 0)

# k_core
dfTrain = pd.read_csv("../data/train.csv", encoding="ISO-8859-1")
dfTest = pd.read_csv("../data/test.csv", encoding="ISO-8859-1")
dfAll = pd.concat([dfTrain, dfTest], axis = 0, ignore_index=True)
for col in "question1", "question2":
    dfAll[col] = dfAll[col].str.lower()
k_core = pd.read_csv("../features/question_kcores.csv")
k_core.index = k_core.question
magic_xgb['q1_k_core'] = dfAll["question1"].map(k_core['kcores']).values
magic_xgb['q1_k_core'].fillna(1, inplace = True)
magic_xgb['q2_k_core'] = dfAll["question2"].map(k_core['kcores']).values
magic_xgb['q2_k_core'].fillna(1, inplace = True)

# pagerank
pagerank_train = pd.read_csv("../features/pagerank_train.csv")
pagerank_test = pd.read_csv("../features/pagerank_test.csv")
pagerank_all = pd.concat([pagerank_train, pagerank_test], axis = 0)
magic_xgb['pagerank_q1'] = pagerank_all['q1_pr']
magic_xgb['pagerank_q2'] = pagerank_all['q2_pr']

X_train = magic_xgb.iloc[0:len(X_train)]
x_test = magic_xgb.iloc[len(X_train):]
X_train.to_pickle("../features/X_train_new.pkl")
x_test.to_pickle("../features/x_test_new.pkl")
fname = "../features/magic_xgb_{:d}D.pkl".format(magic_xgb.shape[1])
_save(fname, magic_xgb.values)




