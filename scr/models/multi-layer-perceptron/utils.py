import imp
import torch
from torch.nn import L1Loss
import torch.utils.data as tud
import constants as const
import numpy as np

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
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test(dataloader,model):
    """
    Use dataloader validate or test model.

    Returns
    -------
    tuple
        (average loss per sample, accuracy of correct positions)
    """
    euclidean_distance=torch.nn.PairwiseDistance(p=2)
    positions_correct_count=0
    samples_total = len(dataloader)
    model.eval()
    loss_sum = 0
    # make predictions for each test sample
    with torch.no_grad():
        for slat_forces, joint_coordinates in dataloader:
            slat_forces = slat_forces.to(const.DEVICE)
            joint_coordinates = joint_coordinates.to(const.DEVICE)

            predictions=model(slat_forces)  # predict single sample

            loss_sum += loss_fn(predictions, joint_coordinates).item()

            # convert prediction and target tensors to numpy arrays
            predictions_array=predictions.detach().cpu().numpy()
            joint_coordinates_array=joint_coordinates.detach().cpu().numpy()
            
            # rescale prediction values back to mm
            predictions_mm=const.SCALER.inverse_transform(predictions_array)
            joint_coordinates_mm=const.SCALER.inverse_transform(joint_coordinates_array)
            
            # reshape predictions accordingly to coordinate sets
            # Joints x num_of_coordinates
            predictions=predictions.reshape((16,3))  
            joint_coordinates=joint_coordinates.reshape((16,3))

            # calculate euclidean distance between predicted joint positions
            # and target positions
            distances=np.linalg.norm(predictions_mm - joint_coordinates_mm,axis=1)

            # check if any joint is out of tolerance, i.e. sleeping pos. incorrect
            if np.all(distances<15):  # Unit: mm
                positions_correct_count+=1
    
    accuracy = positions_correct_count / samples_total
    loss_avg = loss_sum / samples_total
    print(f"Test Error:\nAvg loss: {loss_avg:>8f} Accuracy: {accuracy*100:>.2f}%\n")
    return loss_avg, accuracy

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
    training_data_length    = int(len(dataset)*0.7)  
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