import numpy as np
import pandas as pd
from torch.utils.data import Dataset

class SleepingPositionDataset(Dataset):
    def __init__(self, slat_forces_file, joint_positions_file):
        self.joint_positions=pd.read_csv(joint_positions_file,index_col=[0])
        self.slat_forces=pd.read_csv(slat_forces_file,index_col=[0])

    def __len__(self):
        length_axis_0=self.joint_positions.shape[0]
        length_axis_1=self.joint_positions.shape[1]+self.slat_forces.shape[1]
        return length_axis_0

    def __getitem__(self,index):
        joint_positions=self.joint_positions.iloc[index].to_numpy(dtype=np.float32)
        slat_forces=self.slat_forces.iloc[index].to_numpy(dtype=np.float32)
        return slat_forces, joint_positions