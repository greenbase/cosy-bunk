"""
Load neural network model from 'models' directory and evaluate it on the test dataset
"""
from cgi import test
import utils
import constants as const
from mlp import MLP
import pickle
import torch
import numpy as np
from pathlib import Path
from utility import draw_position_and_image

IMAGE_PATH = Path('../../../data/raw/frames_kinect/')
RESULT_PATH = Path("../../../reports/mlp_regression/")
t,v, test_dataloader=utils.get_data_loaders(const.DATASET)

# load model
with open(const.PATH_NEURAL_NET,"rb") as mlp_file:
    mlp=pickle.load(mlp_file)

# test mlp model
distance_avg_mm, accuracy=utils.test(test_dataloader,mlp)

print(distance_avg_mm,accuracy)

# draw posture images
mlp.eval()
predictions_array=np.zeros((len(test_dataloader),34))
joint_coordinates_array=np.zeros((len(test_dataloader),34))
timestamps=test_dataloader.dataset.dataset.joint_positions.index

# make predictions for each test sample
with torch.no_grad():
    for sample_index, (slat_forces, joint_coordinates) in enumerate(test_dataloader):
        slat_forces = slat_forces.to(const.DEVICE)
        joint_coordinates = joint_coordinates.to(const.DEVICE)
        predictions=mlp(slat_forces)  # predict single sample

        # convert prediction and target tensors to numpy arrays
        predictions=predictions.detach().cpu().numpy()
        joint_coordinates=joint_coordinates.detach().cpu().numpy()

        predictions_array[sample_index]=predictions
        joint_coordinates_array[sample_index]=joint_coordinates

predictions_array=const.POSITION_SCALER.inverse_transform(predictions_array)
joint_coordinates_array=const.POSITION_SCALER.inverse_transform(joint_coordinates_array)
draw_position_and_image(joint_coordinates_array,predictions_array,timestamps,IMAGE_PATH,RESULT_PATH)