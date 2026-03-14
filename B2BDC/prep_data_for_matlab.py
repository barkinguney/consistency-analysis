# i need one csv of active reactions, nominal value, ub, lb 
import pickle
from pathlib import Path
from pprint import pprint
import numpy as np 
import pandas as pd

uncertainty_factors = pd.read_csv("RRC_uncertainty/results/uncertainty_factors.csv")
conditions = pd.read_csv("sensitivity/results/operating_conditions.csv")
#surrogate_data = pd.read_csv("")

uncertainty_factors["nominal"] = 1.0
uncertainty_factors["lb"] = 1.0 / np.pow(10, uncertainty_factors["Uncertainty Factor"])
uncertainty_factors["ub"] = np.pow(10, uncertainty_factors["Uncertainty Factor"])
uncertainty_factors = uncertainty_factors.drop(columns=["Uncertainty Factor"])
uncertainty_factors = uncertainty_factors.rename(columns={"Reaction": "equation"})
print(uncertainty_factors)


conditions = conditions[["composition","filename", "T5", "P5", "phi", "tau", "exp_unc"]]
conditions["exp_unc"] = conditions["exp_unc"] / 100.0

conditions["exp_id"] = (
    conditions["filename"].astype(str).str.replace(".xml", "", regex=False) + 
    "." + 
    conditions["T5"].astype(int).astype(str)
)
conditions["exp_id"] = conditions["exp_id"] + "." + (conditions.groupby("exp_id").cumcount() + 1).astype(str)
conditions = conditions.drop(columns=["filename", "composition"])
conditions = conditions.rename(columns={"T5": "T", "P5": "P", "tau":"exp_idt"})
print(conditions)


uncertainty_factors.to_csv("B2BDC/input/active_reactions.csv", index=False)
conditions.to_csv("B2BDC/input/experiment_units.csv", index=False)




