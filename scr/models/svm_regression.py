"""
This script ...
"""
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

if __name__ == '__main__':
    # define paths of data location
    INP_DATA_PATH = Path("../../data/processed/inp.csv")
    OUT_DATA_PATH = Path("../../data/processed/out.csv")

    inp = pd.read_csv(INP_DATA_PATH)
    out = pd.read_csv(OUT_DATA_PATH)

    inp_train, inp_test, out_train, out_test = train_test_split(inp, out, test_size=0.2,
                                                                random_state=0)

    regressors = []
    preds_test = []
    for col_out_train in out_train:
        regressor = SVR(kernel='rbf')
        regressor.fit(inp_train, out_train[col_out_train])
        regressors.append(regressor)
        preds_test.append(regressor.predict(inp_test))

    # TODO: visualize predicted positions vs real positions
    loss = mean_squared_error(np.array(preds_test).transpose(), out_test.to_numpy())
    abs_error = abs(out_test.to_numpy() - np.array(preds_test).transpose())
