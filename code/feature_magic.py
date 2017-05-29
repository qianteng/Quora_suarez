# convert magic features from online to the format used by Quora_HD
import pandas as pd

def _save(fname, data, protocol=4):
    # use protocol=4 to save files larger than 4GB
    with open(fname, "wb") as f:
        pickle.dump(data, f, protocol)

X_train = pd.read_pickle("../features/X_train.pkl")
X_test = pd.read_pickle("../features/x_test.pkl")
magic_xgb = pd.concat([X_train, X_test], axis = 0)
fname = "../features/magic_xgb_{:d}D.pkl".format(magic_xgb.shape[1])
_save(fname, magic_xgb.values)



