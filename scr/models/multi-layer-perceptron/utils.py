import torch
from torch.nn import L1Loss
import torch.utils.data as tud
import constants as const
import numpy as np

# make modules from scr importable
import os
import sys
from pathlib import Path
os.chdir(Path(__file__).parent)
sys.path.append(os.path.realpath("..\.."))

from utility import get_metrics

loss_fn = L1Loss(reduction="mean")

def train(dataloader, model, optimizer):

    size = len(dataloader.dataset)
    model.train()
    for batch, (X, Y) in enumerate(dataloader, start=1):
        X, Y = X.to(const.DEVICE), Y.to(const.DEVICE)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, Y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
        if batch % 10 == 0:
            loss, current = loss.item(), batch * len(X)
            #print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test(dataloader,model):
    """
    Use dataloader validate or test model.

    Returns
    -------
    tuple
        (average loss per sample, accuracy of correct positions)
    """
    positions_correct_count=0
    samples_total = len(dataloader)
    model.eval()
    loss_sum = 0
    predictions_array=np.zeros((len(dataloader),34))
    joint_coordinates_array=np.zeros((len(dataloader),34))
    # make predictions for each test sample
    with torch.no_grad():
        for sample_index, (slat_forces, joint_coordinates) in enumerate(dataloader):
            slat_forces = slat_forces.to(const.DEVICE)
            joint_coordinates = joint_coordinates.to(const.DEVICE)
            predictions=model(slat_forces)  # predict single sample


            #loss_sum += loss_fn(predictions, joint_coordinates).item()

            # convert prediction and target tensors to numpy arrays
            predictions=predictions.detach().cpu().numpy()
            joint_coordinates=joint_coordinates.detach().cpu().numpy()

            predictions_array[sample_index]=predictions
            joint_coordinates_array[sample_index]=joint_coordinates
            
        distance_avg_mm, accuracy=get_metrics(predictions_array,joint_coordinates_array,const.POSITION_SCALER)

            # rescale prediction values back to mm
    #         predictions_mm=const.POSITION_SCALER.inverse_transform(predictions_array)
    #         joint_coordinates_mm=const.POSITION_SCALER.inverse_transform(joint_coordinates_array)
            
    #         # reshape predictions accordingly to coordinate sets
    #         # Joints x num_of_coordinates
    #         predictions=predictions.reshape((17,2))  
    #         joint_coordinates=joint_coordinates.reshape((17,2))

    #         # calculate euclidean distance between predicted joint positions
    #         # and target positions
    #         distances=np.linalg.norm(predictions_mm - joint_coordinates_mm,axis=1)

    #         # check if any joint is out of tolerance, i.e. sleeping pos. incorrect
    #         if np.all(distances<50):  # Unit: mm
    #             positions_correct_count+=1
    
    # accuracy = positions_correct_count / samples_total
    # loss_avg = loss_sum / samples_total
    #print(f"Test Error:\nAvg loss: {loss_avg:>8f} Accuracy: {accuracy*100:>.2f}%\n")
    return distance_avg_mm, accuracy

def get_data_loaders(dataset,batch_size_train=32):
    """
    Returns tuple of dataloaders: (train,valid,test)
    
    Parameters
    ----------
    dataset : torch.utils.data.Dataset
        Set of data to be distributed to the dataloaders
    batch_size_train : int
        Number of samples per batch from the training dataloader. Note: validation and test dataloader use batch_size=1

    Returns
    -------
    tuple
        Tuple of dataloader objects (train,valid,test)
    """
    # set sizes for data split
    # training data 70% of all data; validation and test data 15% each
    # TODO make calulation of lenght robust for even and odd numbers
    training_data_length    = int(len(dataset)*0.7)+1  
    validation_data_length  = int((len(dataset)-training_data_length)/2)
    test_data_length        = validation_data_length

    # split data
    training_dataset,validation_dataset,test_dataset=tud.random_split(
        dataset=dataset,
        lengths=[training_data_length,validation_data_length,test_data_length],
        generator=torch.Generator().manual_seed(42))

    # instantiate dataloaders
    training_dataloader=tud.DataLoader(
        training_dataset,
        batch_size=batch_size_train,
        shuffle=True)
    validation_dataloader=tud.DataLoader(validation_dataset)
    test_dataloader=tud.DataLoader(test_dataset)

    return (training_dataloader,validation_dataloader,test_dataloader)