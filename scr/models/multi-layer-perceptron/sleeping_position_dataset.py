"""
Defines class providing access to dataset applicable to pytorch.
"""
import numpy as np
import pandas as pd
from torch.utils.data import Dataset

class SleepingPositionDataset(Dataset):
    """
    Subclass of a pytorch dataset.
    
    Parameters
    ----------
    slat_forces_file : str or path obj
        Path to csv file containing measured slat forces (predictor values)
    joint_positions_file : str or path obj
        Path to csv file containing body joint coordinates (target values)
    """
    def __init__(self, slat_forces_file, joint_positions_file):
        self.joint_positions=pd.read_csv(joint_positions_file,index_col="timestamp")
        self.slat_forces=pd.read_csv(slat_forces_file)

    def __len__(self):
        length_axis_0=self.joint_positions.shape[0]
        return length_axis_0

    def __getitem__(self,index):
        joint_positions=self.joint_positions.iloc[index].to_numpy(dtype=np.float32)
        slat_forces=self.slat_forces.iloc[index].to_numpy(dtype=np.float32)
        return slat_forces, joint_positions