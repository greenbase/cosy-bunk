import torch.cuda
import torch.utils.data as tud
from sleeping_position_dataset import SleepingPositionDataset
import pickle
import os
import sys
from pathlib import Path

os.chdir(Path(__file__).parent)
sys.path.append(os.path.realpath("..\..\config"))
from definitions import ROOT_DIR



DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# load scaler for joint position data
_PATH_POSITION_DATA_SCALER=ROOT_DIR / "data/processed/out_scaler.pkl"
#SCALER=pickle.load(_PATH_POSITION_DATA_SCALER)

# load dataset
_PATH_FORCES_DATA=ROOT_DIR / "data/processed/inp.csv"
_PATH_POSITION_DATA=ROOT_DIR / "data/processed/out.csv"
DATASET=SleepingPositionDataset(_PATH_FORCES_DATA, _PATH_POSITION_DATA)

PATH_NEURAL_NET= ROOT_DIR / "models/neural_net.pkl"
PATH_NEURAL_NET_PARAMETERS=ROOT_DIR / "models/neural_net_parameters.json"