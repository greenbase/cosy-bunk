"""

"""
import os
import sys
from pathlib import Path
os.chdir(Path(__file__).parent)
sys.path.append(os.path.realpath("..\.."))

import json

import optuna

# Datascaler class must be available via __main__.Datascaler
# Therefor import it in main.py
from utility import DataScaler  

import constants as const
from objective import Objective

def main():
    # perform hyperparameter tuning
    study=optuna.create_study()
    study.optimize(Objective(), n_trials=2)

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
