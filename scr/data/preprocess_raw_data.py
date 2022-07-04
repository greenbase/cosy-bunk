"""
This script loads the raw json files from specified path, preprocesses them and writes input and
output data to a separate csv-files. Additionally the used data scalers are also saved.

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
from scr.utility import DataScaler


def load_inp(path):
    """Loads relevant force sensor data from multiple json-files.

    Args:
        path(Path): Absolute or relative path to json-files with force measurements.

    Returns:
        :obj:`pd.Dataframe` Force measurements where each column header is of pattern
            'force.<slat_index>.<sensor_index>'. The slat index reaches from 1 to 32 and the sensor
            index reaches from 1 to 2 for the left and right sensor respectively.
    """
    force_column_names = [f"force.{latte_index}.{sensor_index}" for latte_index in range(1, 33) for
                          sensor_index in range(1, 3)]
    df_forces = pd.DataFrame(columns=force_column_names)  # collects readings for all datapoints

    for inp_file in os.listdir(path):
        if inp_file.endswith(".json"):
            with open(path.joinpath(inp_file)) as json_file:
                forces = json.load(json_file)
            if isinstance(forces, dict):
                forces = forces['data']
            assert isinstance(forces, list)

            forces_flat = np.expand_dims(np.array(forces).flatten(), axis=0)
            df_forces = df_forces.append(pd.DataFrame(forces_flat, columns=force_column_names))

    return df_forces


def load_out(path):
    """Loads relevant joint-measurements from multiple json-files.

    Args:
        path(Path): Absolute or relative path to json-files with joint data.

    Returns:
        :obj:`pd.Dataframe` Joint data where each column header is of pattern
            'position.<joint_index>.<dimension>'. The joint index corresponds to the official
            numbering of Microsoft azure kinect. The dimension is either 'x' or 'y'.
        :obj:`pd.Dataframe` separate dataframe containing corresponding time-stamps
    """
    important_joints = [0, 1, 2, 3, 5, 6, 7, 12, 13, 14, 18, 19, 20, 22, 23, 24, 26]
    joint_column_names = [f"position.{joint_index}.{dimension}" for joint_index in
                          important_joints for dimension in ['x', 'y']]
    df_joints = pd.DataFrame(columns=joint_column_names)  # collects readings for all datapoints
    df_time_stamps = pd.DataFrame(columns=['timestamp'])
    for out_file in os.listdir(path):
        if out_file.endswith(".json"):
            with open(path.joinpath(out_file)) as json_file:
                data = json.load(json_file)
                joints_data = data["joints"]
            assert isinstance(joints_data, list)

            positions = []
            for joint_index in important_joints:
                joint_dict = joints_data[joint_index]

                positions.append(joint_dict['position']['v'][:2])
            positions_flat = np.expand_dims(np.array(positions).flatten(), axis=0)
            df_joints = df_joints.append(pd.DataFrame(positions_flat, columns=joint_column_names))
            df_time_stamps = df_time_stamps.append(
                pd.DataFrame(np.array([[data['timestamp']]]), columns=['timestamp']))

    return df_joints, df_time_stamps


def scale_and_save_data(data, dest_path, prefix, time_stamps=None):
    """Scales and saves given data as csv-file. The fitted scaler is saved as pkl-file.

    Args:
        data(:obj:`pd.Dataframe`): Data to transform.
        dest_path(Path): Absolute or relative path for saving data and scaler.
        prefix(str): File prefix for saving specific csv- and pkl-file
        time_stamps(:obj:`pd.Dataframe`): Optional dataframe which contains the timestamp for every
            sample from given data. When passed the timestamps are added as an additional column.
    """
    scaler = DataScaler(data)
    pickle.dump(scaler, open(dest_path.joinpath('{}_scaler.pkl'.format(prefix)), 'wb'))
    scaled_data = pd.DataFrame(scaler.transform(data), columns=data.columns)
    scaled_data = scaled_data.reset_index(drop=True)
    if time_stamps is not None:
        time_stamps = time_stamps.reset_index(drop=True)
        scaled_data = scaled_data.join(time_stamps['timestamp'])
    scaled_data.to_csv(dest_path.joinpath("{}.csv".format(prefix)), index=False)


if __name__ == '__main__':
    # define paths of data location
    INP_PATH = Path("../../data/raw/in")
    OUT_PATH = Path("../../data/raw/out")
    RESULT_PATH = Path("../../data/processed")

    df_forces = load_inp(INP_PATH)
    scale_and_save_data(df_forces, RESULT_PATH, "inp")

    df_joints, df_time_stamps = load_out(OUT_PATH)
    scale_and_save_data(df_joints, RESULT_PATH, "out", time_stamps=df_time_stamps)
