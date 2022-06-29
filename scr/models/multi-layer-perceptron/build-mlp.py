"""

"""

import torch
import mlp
import pandas as pd
import plotly.express as px
import sys
import os
from sleeping_position_dataset import SleepingPositionDataset
import torch.utils.data as tud
from pathlib import Path

os.chdir(Path(__file__).parent)
sys.path.append(os.path.realpath("..\..\config"))
from definitions import ROOT_DIR

def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader, start=1):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    # TODO implement learning rate scheduler

        if batch % 1 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
            return loss


## TODO define test function according to usecase metrics
# def test(dataloader, model, loss_fn):
#     size = len(dataloader.dataset)
#     num_batches = len(dataloader)
#     model.eval()
#     test_loss, correct = 0, 0
#     with torch.no_grad():
#         for X, y in dataloader:
#             X, y = X.to(device), y.to(device)
#             pred = model(X)
#             test_loss += loss_fn(pred, y).item()
#             correct += (pred.argmax(1) == y).type(torch.float).sum().item()
#     test_loss /= num_batches
#     correct /= size
#     print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


if __name__=="__main__":

    # get data paths
    path_input_data=ROOT_DIR / "data/processed/inp.csv"
    path_output_data=ROOT_DIR / "data/processed/out.csv"

    # set up training and test dataset
    dataset=SleepingPositionDataset(path_input_data, path_output_data)
    training_data_length=int(len(dataset)*0.7)  # training data 70% of all data
    training_dataset,test_dataset=tud.random_split(
        dataset=dataset,
        lengths=[training_data_length, len(dataset)-training_data_length],
        generator=torch.Generator().manual_seed(42))
    
    # set up dataloader
    batch_size=32
    train_dataloader=tud.DataLoader(training_dataset,batch_size=batch_size,shuffle=True)
    test_dataloader=tud.DataLoader(test_dataset,batch_size=batch_size,shuffle=True)

    # Get cpu or gpu device for training.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")

    # instantiate MLP
    input_size=64
    # TODO outputsize auf 32 ändern (nur x,y Position)
    output_size=48
    mlp_model=mlp.MLP(input_size,output_size).to(device)
    print(mlp_model)

    # setup loss function optimizer
    #TODO ist die loss function geeignet für multi regression?
    loss_fn = torch.nn.L1Loss(reduction="mean")
    optimizer = torch.optim.SGD(mlp_model.parameters(), lr=1e-1)

    # initialize hyperparameter
    # structure parameter
    hidden_layer_total=
    neurons_per_layer_total=
    activation_fn_hidden_layer=
    # training parameters
    epochs_total=
    momentum=

    # training process
    losses=[]
    epochs=10
    for epoch_count in range(1,epochs+1):
        print(f"Epoch {epoch_count}\n----------------------------")
        loss=train(train_dataloader, mlp_model, loss_fn,  optimizer=optimizer)
        if epoch_count % 100==0:
            losses.append((epoch_count,loss))

    # plot losses over epochs
    losses=pd.DataFrame(losses,columns=["Epochen","Absolute Mean Loss"])
    fig=px.line(losses,x="Epochen",y="Absolute Mean Loss", title="Training of Neural Networks")
    fig.show()
    print("Done!")