import pickle
import torch
from mlp import MLP
from utils import train, test, get_data_loaders
import constants as const

class Objective(object):
    def __init__(self):
        self.accuracy_old=0

    def __call__(self, trial):
        # suggest hyperparameters
        structure_parameters={
            "hidden_layer_total" : trial.suggest_int("hidden_layer_total",2,6),
            #"activation_fn" : trial.suggest_categorical("activation_fn",[torch.nn.ReLU(),torch.nn.Tanh()0]),
            "neurons_per_layer" : trial.suggest_int("neurons_per_layer",16,128)
        }
        training_parameters={
            "epochs_total": trial.suggest_int("epochs_total",5,10, step=5),
            "batch_size_train": trial.suggest_int("batch_size_train",50,400),
            "learning_rate": trial.suggest_float("learning_rate",0.001,0.1)
        }

        # set up model
        # TODO set outputsize to 32
        model=MLP(64,48,**structure_parameters)
        model.to(const.DEVICE)
        print(model)

        optimizer = torch.optim.SGD(model.parameters(), lr=training_parameters["learning_rate"])
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

        # set up data loaders
        training_dataloader, validation_dataloader, test_dataloader=get_data_loaders(const.DATASET,training_parameters["batch_size_train"])

        # train model
        epochs=training_parameters["epochs_total"]
        losses=[]
        accuracies=[]
        for epoch_count in range(1,epochs+1):
            print(f"Epoch {epoch_count}\n----------------------------")
            train(training_dataloader, model, optimizer)
            lr_scheduler.step()
            if epoch_count % 50==0:
                # TODO track training data loss and accuracy
                loss,accuracy = test(validation_dataloader,model)
                losses.append((epoch_count,loss))
                accuracies.append((epoch_count,accuracy))

        # evaluate fully trained model
        loss, accuracy = test(validation_dataloader,model)

        # save new model if accuracy has improved
        if accuracy>self.accuracy_old:
            with open(const.PATH_NEURAL_NET,"wb") as model_file:
                pickle.dump(model,model_file)

        self.accuracy_old=accuracy

        return accuracy