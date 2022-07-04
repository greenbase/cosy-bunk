"""
This script does a support vector machine regression on the preprocessed joint data.
"""
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
import pickle
from scr.utility import DataScaler, draw_position_and_image, save_metrics


def gen_regressors(inp, out):
    regressors = []
    for col_out in out:
        regressor = SVR(kernel='rbf')
        regressor.fit(inp, out[col_out])
        regressors.append(regressor)
    return regressors


def apply_regressors(regressors, inp):
    preds_test = []
    for regressor in regressors:
        preds_test.append(regressor.predict(inp))
    return preds_test


def load_and_split_data(data_path):
    data = pd.read_csv(data_path)

    train, test = train_test_split(data, test_size=0.2, random_state=0)
    return train, test


def preprocess_out_data(out_train, out_test):
    timestamps_train = out_train['timestamp']
    timestamps_test = out_test['timestamp'].to_numpy()
    out_train = out_train.drop(['timestamp'], axis=1)
    out_test = out_test.drop(['timestamp'], axis=1)
    out_train = out_train.reset_index(drop=True)
    out_test = out_test.reset_index(drop=True)

    return out_train, timestamps_train, out_test, timestamps_test


if __name__ == '__main__':
    # define paths of data location
    INP_DATA_PATH = Path("../../../data/processed/inp.csv")
    OUT_DATA_PATH = Path("../../../data/processed/out.csv")
    OUT_SCALER_PATH = Path("../../../data/processed/out_scaler.pkl")
    IMAGE_PATH = Path('../../../data/raw/frames_kinect/')
    RESULT_PATH = Path("../../../reports/svm_regression/")

    OUT_SCALER = pickle.load(open(OUT_SCALER_PATH, "rb"))

    INP_TRAIN, INP_TEST = load_and_split_data(INP_DATA_PATH)
    OUT_TRAIN, OUT_TEST = load_and_split_data(OUT_DATA_PATH)
    OUT_TRAIN, TIMESTAMPS_TRAIN, OUT_TEST, TIMESTAMPS_TEST = preprocess_out_data(OUT_TRAIN,
                                                                                 OUT_TEST)

    REGRESSORS = gen_regressors(INP_TRAIN, OUT_TRAIN)
    PREDS_TEST = apply_regressors(REGRESSORS, INP_TEST)

    TARGETS_TEST_INV = OUT_SCALER.invert_transform(OUT_TEST.to_numpy())
    PREDS_INV = OUT_SCALER.invert_transform(np.array(PREDS_TEST).transpose())
    save_metrics(TARGETS_TEST_INV, PREDS_INV, RESULT_PATH)
    draw_position_and_image(TARGETS_TEST_INV, PREDS_INV, TIMESTAMPS_TEST, IMAGE_PATH, RESULT_PATH)
