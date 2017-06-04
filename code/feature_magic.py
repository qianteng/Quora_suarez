# convert magic features from online to the format used by Quora_HD
import pandas as pd
import pickle

def _save(fname, data, protocol=4):
    # use protocol=4 to save files larger than 4GB
    with open(fname, "wb") as f:
        pickle.dump(data, f, protocol)

X_train = pd.read_pickle("../features/X_train.pkl")
X_test = pd.read_pickle("../features/x_test.pkl")

dfTrain = pd.read_csv("../data/train.csv", encoding="ISO-8859-1")
dfTest = pd.read_csv("../data/test.csv", encoding="ISO-8859-1")
dfAll = pd.concat([dfTrain, dfTest], axis = 0, ignore_index=True)
for col in "question1", "question2":
    dfAll[col] = dfAll[col].str.lower()
k_core = pd.read_csv("../features/question_kcores.csv")
k_core.index = k_core.question
feature_k_core = dfAll["question1"].map(k_core['kcores']).fillna(-1, inplace = True)

magic_xgb = pd.concat([X_train, X_test], axis = 0)
fname = "../features/magic_xgb_{:d}D.pkl".format(magic_xgb.shape[1])
_save(fname, magic_xgb.values)



