"""
Load neural network model from 'models' directory and evaluate it on the test dataset
"""
import utils
import constants as const
from mlp import MLP
import pickle
t,v, test_dataloader=utils.get_data_loaders(const.DATASET)

# load model
with open(const.PATH_NEURAL_NET,"rb") as mlp_file:
    mlp=pickle.load(mlp_file)

# test mlp model
distance_avg_mm, accuracy=utils.test(test_dataloader,mlp)

print(distance_avg_mm,accuracy)