"""
Load neural network model from 'models' directory and evaluate it on the test dataset
"""
import utils
import constants as const
from mlp import MLP
import pickle
import torch
from pathlib import Path
from scr.utility import draw_position_and_image
import io
import pandas as pd
from sklearn.model_selection import train_test_split


class CPUUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else:
            return super().find_class(module, name)


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
    INP_DATA_PATH = Path("../../../data/processed/inp.csv")
    OUT_DATA_PATH = Path("../../../data/processed/out.csv")
    IMAGE_PATH = Path('../../../data/raw/frames_kinect/')
    RESULT_PATH = Path("../../../reports/mlp_regression/")
    t, v, test_dataloader = utils.get_data_loaders(const.DATASET)

    # load model
    with open(const.PATH_NEURAL_NET, "rb") as mlp_file:
        if const.DEVICE == 'cpu':
            mlp = CPUUnpickler(mlp_file).load()
        else:
            mlp = pickle.load(mlp_file)

    # test mlp model
    distance_avg_mm, accuracy = utils.test(test_dataloader, mlp)
    print(distance_avg_mm, accuracy)

    # draw posture images
    mlp.eval()
    INP_TRAIN, INP_TEST = load_and_split_data(INP_DATA_PATH)
    OUT_TRAIN, OUT_TEST = load_and_split_data(OUT_DATA_PATH)
    OUT_TRAIN, TIMESTAMPS_TRAIN, OUT_TEST, TIMESTAMPS_TEST = preprocess_out_data(OUT_TRAIN,
                                                                                 OUT_TEST)
    if const.DEVICE == 'cpu':
        PREDS_TEST = mlp(torch.tensor(INP_TEST.to_numpy(), dtype=torch.float32)).detach().numpy()
    else:
        # put input to GPU
        INP_TEST_GPU = torch.tensor(INP_TEST.to_numpy(), dtype=torch.float32).to(const.DEVICE)

        PREDS_TEST = mlp(INP_TEST_GPU).detach().cpu().numpy()

    TARGETS_TEST_INV = const.POSITION_SCALER.inverse_transform(OUT_TEST.to_numpy())
    PREDS_INV = const.POSITION_SCALER.inverse_transform(PREDS_TEST)
    draw_position_and_image(TARGETS_TEST_INV, PREDS_INV, TIMESTAMPS_TEST, IMAGE_PATH, RESULT_PATH)
