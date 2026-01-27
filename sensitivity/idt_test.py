import pandas as pd
import numpy as np
import cantera as ct
import matplotlib.pyplot as plt
from scipy.stats import qmc
from scipy import linalg
import time
import copy

import read_data
import cantera_related_functions



#Setup mechanism
#gas = ct.Solution('Supplementary-3_syngas.yaml')
gas = ct.Solution('C2H4_2021.yaml')
no_reactions = len(gas.reactions())
prior_uncertainty_factor = 0.3

rxns_df = pd.DataFrame()
rxns_df["equation"] = gas.reaction_equations()
rxns_df["id"] = rxns_df["equation"].apply(lambda eq: cantera_related_functions.find_reaction_index_by_equation(gas, eq))
rxns_df["f_value"] = prior_uncertainty_factor

#idt_data_folders = ["data\\idt_data\\hydrogen", "data\\idt_data\\syngas"]
idt_data_folders = ["data\\idt_data\\ethylene"]
idt_data_df = pd.DataFrame()
for idt_data_folder in idt_data_folders:
    idt_data_df = pd.concat([idt_data_df, read_data.extract_idt_data_to_dataframe(idt_data_folder)], ignore_index=True)



# n=0
# for idx,operating_condition in enumerate(idt_data_df.itertuples(index=False)):
#     if (operating_condition[8] == "max") and n<5:
#         print(operating_condition)
#         tau = cantera_related_functions.calc_IDT_constV(gas=gas, operating_condition=operating_condition, t_max=2, save_time_history_plot=True)
#         n+=1
        
# for idx,operating_condition in enumerate(idt_data_df.itertuples(index=False)):
#     if (operating_condition[8] == "max"):
#         print(operating_condition)
#         tau = cantera_related_functions.calc_IDT_constV(gas=gas, operating_condition=operating_condition, t_max=2, save_time_history_plot=True)

for key, group in idt_data_df.groupby('filename'):
    # if group["ignition_type"].iloc[0] == "max":
    #     cantera_related_functions.plot_IDT_vs_T(gas, group)
    cantera_related_functions.plot_IDT_vs_T(gas, group)
