"""

"""
import os
import sys
from pathlib import Path
os.chdir(Path(__file__).parent)
sys.path.append(os.path.realpath("..\.."))

import json

import optuna
from optuna.pruners import ThresholdPruner

# Datascaler class must be available via __main__.Datascaler
# Therefor import it in main.py
from utility import DataScaler  

import constants as const
from objective import Objective

starting_point={
    "hidden_layer_total":4,
    "neurons_per_layer":60,
    "batch_size_train":10,
    "learning_rate":1.2,
}

def main():
    # perform hyperparameter tuning
    study=optuna.create_study(direction="maximize",pruner=ThresholdPruner(upper=24,n_warmup_steps=999))
    study.enqueue_trial(starting_point)
    study.optimize(Objective(), n_trials=10)

    # save best hyperparameters and best model
    with open(const.PATH_NEURAL_NET_PARAMETERS,"w",encoding="utf-8") as file:
        json.dump(study.best_params, file, indent=4, ensure_ascii=False)

if __name__=="__main__":
    main()



    # # plot losses over epochs
    # losses=pd.DataFrame(losses,columns=["Epochen","Absolute Mean Loss"])
    # fig=px.line(losses,x="Epochen",y="Absolute Mean Loss", title="Training of Neural Networks")
    # fig.show()
    # print("Done!")

    # study=optuna.create_study(direction="maximize")
    # study.optimize(objective,n_trials=100)
