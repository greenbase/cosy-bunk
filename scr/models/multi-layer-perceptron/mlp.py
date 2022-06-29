from torch import relu
import torch.nn as nn

class MLP(nn.Module):
    """
    Sets up a multi-layer perceptron that approximates a multi-output regression function.

    Parameters
    ----------
    input_size : int
        Number of input parameters
    output_size : int
        Number of 

    Returns
    -------
    model : torch.Sequential
        Pytorch model of sequentialy stacked layers and activation functions
    """

    def __init__(self,input_size:int,output_size:int) -> None:
        super().__init__()
        self.linear_relu_stack=nn.Sequential(
            nn.Linear(input_size,16),
            nn.ReLU(),
            nn.Linear(16,output_size),
            nn.Tanh(),
        )

    def forward(self,x):
        """
        Specifies how the data will be passed forward through the network.
        
        Parameters
        ----------
        x : array-like
            Data feed to the model
            
        Returns
        -------
        logits : array-like
            Array of logits as computed for one hidden layer.
        """
        logits=self.linear_relu_stack(x)
        return logits