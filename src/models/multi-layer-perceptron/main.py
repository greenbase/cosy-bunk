"""
Execute hyperparameter tuning for neural network.

Optionally defines a starting point for the hyperparameter tuning and saves Parameters of the best
model to the 'models' directory.
"""
import os
import sys
from pathlib import Path
import json
import optuna
from optuna.pruners import ThresholdPruner
import constants as const
from objective import Objective

# Datascaler class must be available via __main__.Datascaler, therefore import it in main.py
os.chdir(Path(__file__).parent)
sys.path.append(os.path.realpath("..\..\..\src"))
from utility import DataScaler

starting_point = {
    "hidden_layer_total": 4,
    "neurons_per_layer": 60,
    "batch_size_train": 10,
    "learning_rate": 0.3,
}


def main():
    # perform hyperparameter tuning
    study = optuna.create_study(direction="maximize",
                                pruner=ThresholdPruner(upper=24, n_warmup_steps=999))
    study.enqueue_trial(starting_point)
    study.optimize(Objective(), n_trials=1)

    # save best hyperparameters and best model
    with open(const.PATH_NEURAL_NET_PARAMETERS, "w", encoding="utf-8") as file:
        json.dump(study.best_params, file, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    main()
