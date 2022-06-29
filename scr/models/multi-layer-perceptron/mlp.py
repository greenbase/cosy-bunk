from collections import OrderedDict
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
        Number of output values
    hidden_layer : int
        Number of hidden layer in neural network
    activation_fn : object of torch.nn
        Activation function used in hidden layer neurons
    neurons_per_layer : int
        Number of neuron in each hidden layer

    Returns
    -------
    model : torch.Sequential
        Pytorch model of sequentialy stacked layers and activation functions
    """

    def __init__(self,input_size:int,output_size:int, hidden_layer_total=2, activation_fn=nn.ReLU(), neurons_per_layer=16) -> None:
        super().__init__()
        layers=OrderedDict()

        # collect layers
        for layer_index in range(1,hidden_layer_total+1):
            if layer_index==1:
                # add first hidden layer
                layers[f"hl_{layer_index}"]=nn.Linear(input_size,neurons_per_layer)
                layers[f"hl_{layer_index}_activation"]=activation_fn
                continue
            # add additional layers
            layers[f"hl_{layer_index}"]=nn.Linear(neurons_per_layer,neurons_per_layer)
            layers[f"hl_{layer_index}_activation"]=activation_fn

        # add output layer
        layers["output_layer"]=nn.Linear(neurons_per_layer,output_size)
        layers["output_layer_activation"]=nn.Tanh()

        # build mlp
        self.linear_relu_stack=nn.Sequential(layers)


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