"""
This module reads all json files from the specified directories, parses them and writes all the
data into a pandas dataframe which then is saved in csv-format.

The proper assembly of the dataframe relies on the correct mapping of files in one folder
(input data) and the respective files in another folder (output data). I.e. the amount of
json-files in both folders must be equal as well as the order of files within each folder must
match the order in the other folder.

For details on the table of data build see references>data>explanation-column-names.md

To run this script change to it directory first or execute it as a module.
"""

import sys
import os
import json
import pandas as pd
import numpy as np

sys.path.append(os.path.realpath("..\config"))
from definitions import ROOT_DIR

# define paths of data location
path_slatted_frame_measurements = ROOT_DIR / "data/raw/in"
path_joints_coordinates = ROOT_DIR / "data/raw/out"
path_interim_data = ROOT_DIR / "data/interim"

df_raw_data = pd.DataFrame()

# HANDLE JOINT MEASUREMENTS ==>

for joints_file in path_joints_coordinates.glob("*.json"):
    row = pd.DataFrame()
    with open(joints_file) as json_file:
        data = json.load(json_file)
        joints_data = data["joints"]
    assert isinstance(joints_data, list)

    # append orientation and position data for all joints to the new
    # datapoint
    for joint_index, joint_dict in enumerate(joints_data):
        df_single_joint = pd.json_normalize(joint_dict)
        # add prefix to column names to identify joints
        df_single_joint = df_single_joint.add_prefix(f"j{joint_index}.")
        row = pd.concat([row, df_single_joint], axis=1)
    assert row.shape[0] == 1, "Data of all joints should be in the same row."

    # add joint data for current observation aka joint file to dataframe
    df_raw_data = pd.concat([df_raw_data, row], axis=0)

# HANDLE FORCE SENSOR DATA ==>

# define column names for force sensor readings
# Pattern: "force.<slat_index>.<sensor_index>"
# The slat index reaches from 1 to 32
# The sensor index reaches from 1 to 2 for the left and right sensor
# respectively
column_names = [f"force.{latte_index}.{sensor_index}" for latte_index in range(1, 33) for
                sensor_index in range(1, 3)]

df_forces = pd.DataFrame(columns=column_names)  # collects readings for all datapoints

for slatted_frame_file in path_slatted_frame_measurements.glob("*.json"):
    row = pd.DataFrame(columns=column_names)
    with open(slatted_frame_file) as json_file:
        forces = json.load(json_file)
    assert isinstance(forces, list)
    forces = np.array(forces)
    assert forces.shape == (32, 2)

    forces_flat = forces.reshape((1, 64))
    row = pd.DataFrame(forces_flat, columns=column_names)
    df_forces = pd.concat([df_forces, row], axis=0)

assert df_forces.shape[1] == 64

# merge force sensor values with joint measurements
df_raw_data = pd.concat([df_raw_data, df_forces], axis=1)

assert df_raw_data.shape[1] == 9 * 32 + 64

df_raw_data = df_raw_data.reset_index(drop=True)

df_raw_data.to_csv(path_interim_data / "joint-and-force-values.csv")
