"""
This script does a support vector machine regression on the preprocessed joint data.
"""
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt


def draw_position(targets, preds, path):
    """Generates plot of target and predicted joints such that coherent joints are connected and
    a human pose is displayed.

    Args:
        targets(:obj:`np.ndarray`): Target joint data of shape (<num samples>, 34,), where the
            second axis contains 17 joints of interest with one x and one y coordinate. Note that
            the order is important and that corresponding x and y coordinate have to be successive.
        preds(:obj:`np.ndarray`): Predicted joint data of shape (<num samples>, 34,), where the
            second axis contains 17 joints of interest with one x and one y coordinate. Note that
            the order is important and that corresponding x and y coordinate have to be successive.
    """
    plt.figure()
    connect_idxs = [[0, 1], [0, 10], [0, 13], [1, 2], [2, 3], [3, 4], [3, 7], [3, 16], [4, 5],
                    [5, 6], [7, 8], [8, 9], [10, 11], [11, 12], [13, 14], [14, 15]]
    for idx in range(0, len(preds) - 8, 8):
        plt.clf()
        plt.title("Target vs. Prediction")
        axes = [plt.subplot(241), plt.subplot(242), plt.subplot(243), plt.subplot(244),
                plt.subplot(245), plt.subplot(246), plt.subplot(247), plt.subplot(248)]
        for ax_idx, ax in enumerate(axes):
            ax.set_xlim([-1.2, 1.2])
            ax.set_ylim([-1.2, 1.2])
            pred_in = preds[idx + ax_idx][0::2]
            pred_out = preds[idx + ax_idx][1::2]
            target_in = targets[idx + ax_idx][0::2]
            target_out = targets[idx + ax_idx][1::2]
            # plot connections between joints
            for con_idx in connect_idxs:
                ax.plot(pred_in[con_idx], pred_out[con_idx], color="red")
                ax.plot(target_in[con_idx], target_out[con_idx], color="blue")
            # plot head
            ax.plot(pred_in[16], pred_out[16], marker="o", markersize=20, markeredgecolor="red",
                    markerfacecolor="red")
            ax.plot(target_in[16], target_out[16], marker="o", markersize=20,
                    markeredgecolor="blue", markerfacecolor="blue")

        plt.legend(["Prediction", "Target"], loc=(0, 0.8))
        plt.savefig(path.joinpath("predictions_{0}-{1}.png".format(idx, idx + 7)))


def save_metrics(targets, preds, path):
    # TODO: revert normalization + add metric according to 'model_quality_criteria.md'
    mse = mean_squared_error(preds, targets)
    mae = mean_absolute_error(preds, targets)
    with open(path.joinpath('metrics.txt'), 'w') as f:
        f.write("Mean Sqared Error: {}\n".format(mse))
        f.write("Mean Absolute Error: {}\n".format(mae))


if __name__ == '__main__':
    # define paths of data location
    INP_DATA_PATH = Path("../../data/processed/inp.csv")
    OUT_DATA_PATH = Path("../../data/processed/out.csv")
    RESULT_PATH = Path("../../reports/svm_regression/")

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

    TARGETS = out_test.to_numpy()
    PREDS = np.array(preds_test).transpose()

    save_metrics(TARGETS, PREDS, RESULT_PATH)
    draw_position(TARGETS, PREDS, RESULT_PATH)
