"""
Abbreviations:
mm  Millimeter
avg Average
"""
from turtle import distance
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt


class DataScaler:
    def __init__(self, data):
        """Class for scaler which is used before application of model to scale input correctly.

        Args:
            data (:obj:`np.ndarray`): Input to transform of shape (number samples, features,).
        """
        self._normalizer = self._get_fitted_scalers(data)

    @staticmethod
    def _get_fitted_scalers(data):
        # Returns normalizer which is fitted to the data.
        normalizer = MinMaxScaler(feature_range=(-1, 1))
        normalizer.fit(data)

        return normalizer

    def transform(self, data):
        """Executes normalisation.

        Args:
            data(:obj:`np.ndarray`): Data to transform.

        Returns:
            :obj:`np.ndarray` transformed data
        """
        return self._normalizer.transform(data)

    def inverse_transform(self, data):
        """Reverts normalisation.

        Args:
            data(:obj:`np.ndarray`): Data to invert transform.

        Returns:
            :obj:`np.ndarray` invert transformed data
        """
        return self._normalizer.inverse_transform(np.copy(data))


def draw_position_and_image(targets, preds, time_stamps, image_path, result_path):
    """Generates plot of target and predicted joints such that coherent joints are connected and
    a human pose is displayed.

    Args:
        targets(:obj:`np.ndarray`): Target joint data of shape (<num samples>, 34,), where the
            second axis contains 17 joints of interest with one x and one y coordinate. Note that
            the order is important and that corresponding x and y coordinate have to be successive.
        preds(:obj:`np.ndarray`): Predicted joint data of shape (<num samples>, 34,), where the
            second axis contains 17 joints of interest with one x and one y coordinate. Note that
            the order is important and that corresponding x and y coordinate have to be successive.
        time_stamps(:obj:`np.ndarray`): Array of shape (<num samples>,) which contains the
            timestamp for every target/predicted sample.
        image_path(Path): Absolute or relative path to image data folder.
        result_path(Path): Absolute or relative path for saving result plots.
    """
    plt.figure()
    connect_idxs = [[0, 1], [0, 10], [0, 13], [1, 2], [2, 3], [3, 4], [3, 7], [3, 16], [4, 5],
                    [5, 6], [7, 8], [8, 9], [10, 11], [11, 12], [13, 14], [14, 15]]
    for idx in range(0, len(preds) - 4, 4):
        plt.clf()
        plt.title("Target vs. Prediction")
        axes = [plt.subplot(241), plt.subplot(242), plt.subplot(243), plt.subplot(244),
                plt.subplot(245), plt.subplot(246), plt.subplot(247), plt.subplot(248)]
        for ax_idx, ax in enumerate(axes):
            if ax_idx % 2 == 0:
                pred_in = preds[int((idx + ax_idx) / 2)][0::2]
                pred_out = preds[int((idx + ax_idx) / 2)][1::2]
                target_in = targets[int((idx + ax_idx) / 2)][0::2]
                target_out = targets[int((idx + ax_idx) / 2)][1::2]

                # plot connections between joints
                for con_idx in connect_idxs:
                    ax.plot(pred_in[con_idx], pred_out[con_idx], color="red")
                    ax.plot(target_in[con_idx], target_out[con_idx], color="blue")
                ax.tick_params(axis='both', which='major', labelsize=8)
                ax.tick_params(axis='both', which='minor', labelsize=7)

                # plot head
                ax.plot(pred_in[16], pred_out[16], marker="o", markersize=20,
                        markeredgecolor="red", markerfacecolor="red")
                ax.plot(target_in[16], target_out[16], marker="o", markersize=20,
                        markeredgecolor="blue", markerfacecolor="blue")
            else:
                time_stamp = time_stamps[int((idx + ax_idx - 1) / 2)]
                file_name = 'frame-body-{}.png'.format(time_stamp)
                im = plt.imread(image_path.joinpath(file_name))
                upper_start = int(0.07*np.shape(im)[0])
                left_start = int(0.35*np.shape(im)[1])
                right_end = int(0.65*np.shape(im)[1])
                ax.imshow(np.flip(im[upper_start:, left_start:right_end], axis=1))
                ax.axis('off')
                ax.set_title(time_stamp, size=9)

        # add legend
        prediction_proxy = plt.Rectangle((0, 0), 1, 1, fc="red")
        target_proxy = plt.Rectangle((0, 0), 1, 1, fc="blue")
        plt.legend([prediction_proxy, target_proxy], ["Prediction", "Target"], loc=(-4.1, 2.1))
        plt.savefig(result_path.joinpath("predictions_{0}-{1}.png".format(idx, idx + 7)))


def save_metrics(distance_avg_mm, accuracy, path):
    """Calculates relevant metrics and saves them to txt-file under specified path.

    Args:
        targets(:obj:`np.ndarray`): Target joint data of shape (<num samples>, 34,), where the
            second axis contains 17 joints of interest with one x and one y coordinate. Note that
            the order is important and that corresponding x and y coordinate have to be successive.
        preds(:obj:`np.ndarray`): Predicted joint data of shape (<num samples>, 34,), where the
            second axis contains 17 joints of interest with one x and one y coordinate. Note that
            the order is important and that corresponding x and y coordinate have to be successive.
        path(Path): Absolute or relative path for saving result txt-file.
    """
    with open(path.joinpath('metrics.txt'), 'w') as f:
        f.write(f"Average joint distance in mm: {distance_avg_mm}\n")
        f.write(f"Accuracy: {accuracy}\n")

def get_metrics(predictions, targets, scaler):
    """
    Calculates the proportion of sleeping positions predicted correctly within tolerance.
    
    Parameters
    ----------
    predictions : array
        Array of predicted and scaled joint coordinates. Shape: (Testsamples x 34)
    targets : array
        Array of measured and scaled joint coordinates.
    scaler : Instance of DataScaler
        Scaler previously used to scale data. Used here to perform inverse scaling.
    
    Returns
    -------
    loss_avg : float
        Average error for a single coordinate prediction
    accuracy : float
        Proportion of samples for which all predicted joint positions fall into specified radial tolerance area around the target joint.
    """
    distance_sum_mm = 0
    positions_correct_count = 0
    SAMPLES_TOTAL = len(predictions)

    for sample_prediction, sample_target in zip(predictions,targets):
        # rescale values to Millimeters
        sample_prediction_mm=scaler.inverse_transform(sample_prediction.reshape((1,-1)))
        sample_target_mm = scaler.inverse_transform(sample_target.reshape((1,-1)))

        # reshape sample arrays. New shape: (Joints x joint-coordinates)
        sample_prediction_mm=sample_prediction_mm.reshape((17,2))
        sample_target_mm=sample_target_mm.reshape((17,2))

        # calculate euclidean distance between predicted joint positions and
        # target positions
        sample_distances_mm=np.linalg.norm(sample_prediction_mm - sample_target_mm,axis=1)
        distance_sum_mm+=np.sum(sample_distances_mm)

        # check if any joint is out of tolerance and count No. of sleeping
        # positions for which all joint fall within tolerance area
        distance_tolerance_mm = 50
        if np.all(sample_distances_mm < distance_tolerance_mm):
            positions_correct_count+=1

    accuracy = positions_correct_count / SAMPLES_TOTAL
    distance_mm_avg = distance_sum_mm / (len(sample_distances_mm)*SAMPLES_TOTAL)

    return distance_mm_avg, accuracy