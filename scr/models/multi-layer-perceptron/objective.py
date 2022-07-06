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
            "epochs_total": trial.suggest_int("epochs_total",500,1000, step=500),
            "batch_size_train": trial.suggest_int("batch_size_train",50,400),
            "learning_rate": trial.suggest_float("learning_rate",0.001,0.1)
        }

        # set up model
        model=MLP(64,34,**structure_parameters)
        model.to(const.DEVICE)
        print(model)

        optimizer = torch.optim.SGD(model.parameters(), lr=training_parameters["learning_rate"])
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

        # set up data loaders
        training_dataloader, validation_dataloader, test_dataloader=get_data_loaders(const.DATASET,training_parameters["batch_size_train"])

        # train model
        epochs_total=training_parameters["epochs_total"]
        for epoch_count in range(1,epochs_total+1):
            print(f"Epoch {epoch_count}\n----------------------------")
            train(training_dataloader, model, optimizer)
            lr_scheduler.step()
            # evaluate model every x epochs or after the last training epoch
            if epoch_count % 50==0 | epoch_count==epochs_total:
                # TODO track training data loss and accuracy
                loss_training, accuracy_training= test(training_dataloader, model)
                loss_validation,accuracy_validation = test(validation_dataloader,model)
                # losses.append((epoch_count,loss))
                # accuracies.append((epoch_count,accuracy))

                # write metrics to csv
        # save new model if accuracy has improved
        if accuracy_validation>self.accuracy_old:
            with open(const.PATH_NEURAL_NET,"wb") as model_file:
                pickle.dump(model,model_file)

        self.accuracy_old=accuracy_validation

        return accuracy_validation