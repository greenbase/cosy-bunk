"""
This script loads the raw json files from specified path, preprocesses them and writes input and
output data to a seperate csv-files. Additionally the used Datascalers are also saved.

The proper assembly of this script relies on the correct mapping of files in the input folder and
the respective files in the output folder. I.e. the amount of json-files in both folders must be
equal as well as the order of files within each folder must match the order in the other folder.

For details on the table of data build see references>data>explanation-column-names.md
"""

import os
import json
import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler


class DataScaler:
    def __init__(self, data):
        """Class for scaler which is used before application of model to scale input correctly.

        Args:
            data (:obj:`np.ndarray`): Input to transform of shape (number samples, features,)
.
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
        return pd.DataFrame(self._normalizer.transform(data), columns=data.columns)


# HANDLE FORCE SENSOR DATA
def load_inp(path):
    # define column names for force sensor readings
    # Pattern: "force.<slat_index>.<sensor_index>"
    # The slat index reaches from 1 to 32
    # The sensor index reaches from 1 to 2 for the left and right sensor respectively
    force_column_names = [f"force.{latte_index}.{sensor_index}" for latte_index in range(1, 33) for
                          sensor_index in range(1, 3)]
    df_forces = pd.DataFrame(columns=force_column_names)  # collects readings for all datapoints

    for inp_file in os.listdir(path):
        if inp_file.endswith(".json"):
            with open(path.joinpath(inp_file)) as json_file:
                forces = json.load(json_file)
            assert isinstance(forces, list)

            forces_flat = np.expand_dims(np.array(forces).flatten(), axis=0)
            df_forces = df_forces.append(pd.DataFrame(forces_flat, columns=force_column_names))

    return df_forces


# HANDLE JOINT MEASUREMENTS
def load_out(path):
    # define column names for force sensor readings
    # Pattern: "position.<joint_index>.<dimension>"
    important_joints = [0, 1, 2, 3, 6, 7, 12, 13, 14, 18, 19, 20, 22, 23, 24, 26]
    joint_column_names = [f"position.{joint_index}.{dimension}" for joint_index in important_joints
                          for dimension in ['x', 'y', 'z']]
    df_joints = pd.DataFrame(columns=joint_column_names)  # collects readings for all datapoints
    for out_file in os.listdir(path):
        if out_file.endswith(".json"):
            with open(path.joinpath(out_file)) as json_file:
                data = json.load(json_file)
                joints_data = data["joints"]
            assert isinstance(joints_data, list)

            positions = []
            for joint_index in important_joints:
                joint_dict = joints_data[joint_index]

                positions.append(joint_dict['position']['v'])
            positions_flat = np.expand_dims(np.array(positions).flatten(), axis=0)
            df_joints = df_joints.append(pd.DataFrame(positions_flat, columns=joint_column_names))

    return df_joints


def scale_and_safe_data(data, dest_path, prefix):
    scaler = DataScaler(data)
    scaled_data = scaler.transform(data)
    pickle.dump(scaler, open(dest_path.joinpath('{}_scaler.pkl'.format(prefix)), 'wb'))
    scaled_data.reset_index(drop=True).to_csv(dest_path.joinpath("{}.csv".format(prefix)))


if __name__ == '__main__':
    # define paths of data location
    INP_PATH = Path("../../data/raw/in")
    OUT_PATH = Path("../../data/raw/out")
    RESULT_PATH = Path("../../data/processed")

    scale_and_safe_data(load_inp(INP_PATH), RESULT_PATH, "inp")
    scale_and_safe_data(load_out(OUT_PATH), RESULT_PATH, "out")
