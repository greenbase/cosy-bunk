import pickle
import optuna
import torch
from mlp import MLP
from utils import train, test, get_data_loaders
import constants as const
import csv

class Objective(object):
    def __init__(self):
        self.accuracy_max=0

    def __call__(self, trial):
        # suggest hyperparameters
        structure_parameters={
            "hidden_layer_total" : trial.suggest_int("hidden_layer_total",1,6),
            #"activation_fn" : trial.suggest_categorical("activation_fn",[torch.nn.ReLU(),torch.nn.Tanh()0]),
            "neurons_per_layer" : trial.suggest_int("neurons_per_layer",20,200,20)
        }
        training_parameters={
            "epochs_total": trial.suggest_int("epochs_total",4000,4000,5000),
            "batch_size_train": trial.suggest_int("batch_size_train",10,150,20),
            "learning_rate": trial.suggest_float("learning_rate",0.01,0.2)
        }

        # set up model
        model=MLP(64,34,**structure_parameters)
        model.to(const.DEVICE)
        print(model)

        optimizer = torch.optim.SGD(model.parameters(), lr=training_parameters["learning_rate"])
        # lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)

        # set up data loaders
        training_dataloader, validation_dataloader, test_dataloader=get_data_loaders(const.DATASET,training_parameters["batch_size_train"])

        # set up csv file for metrics of new model to be trained
        fieldnames=["epoch","loss","accuracy"]
        with open("model_metrics.csv","w",newline="") as csv_file:
            csv_writer=csv.DictWriter(csv_file,fieldnames)
            csv_writer.writeheader()

        # train model
        epochs_total=training_parameters["epochs_total"]
        for epoch_count in range(1,epochs_total+1):
            #print(f"Epoch {epoch_count}\n----------------------------")
            train(training_dataloader, model, optimizer)
            # lr_scheduler.step()
            # evaluate model every x epochs or after the last training epoch
            if epoch_count % 100==0 or epoch_count==epochs_total:
                metrics={"epoch": epoch_count}
                metrics["loss"], metrics["accuracy"] = test(validation_dataloader,model)

                # prune trial if loss 
                trial.report(metrics["loss"],epoch_count-1)
                if trial.should_prune():
                    raise optuna.TrialPruned()

                # write metrics to csv
                with open("model_metrics.csv","a",newline="") as csv_file:
                    csv_writer=csv.DictWriter(csv_file,fieldnames)
                    csv_writer.writerow(metrics)       

        # save new model if accuracy has improved
        if metrics["accuracy"]>self.accuracy_max:
            self.accuracy_max=metrics["accuracy"]
            with open(const.PATH_NEURAL_NET,"wb") as model_file:
                pickle.dump(model,model_file)

        return metrics["accuracy"]