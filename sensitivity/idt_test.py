import pandas as pd
import numpy as np
import cantera as ct
import matplotlib.pyplot as plt
from scipy.stats import qmc
from scipy import linalg
import time
import copy
from pathlib import Path

import read_data
import cantera_related_functions



#Setup mechanism
gas = ct.Solution('Supplementary-3_syngas.yaml')
# gas = ct.Solution('C2H4_2021.yaml')
no_reactions = len(gas.reactions())
prior_uncertainty_factor = 0.3

rxns_df = pd.DataFrame()
rxns_df["equation"] = gas.reaction_equations()
rxns_df["id"] = rxns_df["equation"].apply(lambda eq: cantera_related_functions.find_reaction_index_by_equation(gas, eq))
rxns_df["f_value"] = prior_uncertainty_factor

idt_data_folders = ["data\\idt_data\\hydrogen", "data\\idt_data\\syngas"]
# idt_data_folders = ["data\\idt_data\\ethylene"]
idt_data_df = pd.DataFrame()
for idt_data_folder in idt_data_folders:
    df = read_data.extract_idt_data_to_dataframe(idt_data_folder)
    df.fillna(value= "", inplace=True)
    idt_data_df = pd.concat([idt_data_df, df], ignore_index=True)



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

# data_df = pd.read_csv("idt_test_data.csv", na_filter=False)
# print(data_df)
# for key, group in idt_data_df.groupby('filename'):
#     if group["ignition_type"].iloc[0] == "d/dt max":
#         cantera_related_functions.plot_IDT_vs_T(gas, group, save_time_history_plots=False)
        


# valid_filenames = {f"x100000{a}.xml" for a in range(33, 42)}
# idt_data_df = idt_data_df[idt_data_df["filename"].apply(lambda x: Path(x).name in valid_filenames)]
# for key, group in idt_data_df.groupby('filename'):
#     print(key)
#     ignition_type = group["ignition_type"].iloc[0]
#     xml_name = group["filename"].iloc[0].split(".")[0]
#     fuel_name = group["composition"].iloc[0].split(":")[0]
#     results_path = f"results/idt/plots/{fuel_name}/{ignition_type}/{xml_name}"
#     cantera_related_functions.plot_IDT_vs_T(
#         gas,
#         group,
#         output_path=results_path,
#         interactive=False,
#         save_time_history_plots=False,
#     )


# idt_data_df = pd.read_csv("sensitivity/results/operating_conditions.csv")
# idt_data_df = idt_data_df.fillna(value= "", inplace=False)
# for key, group in idt_data_df.groupby('filename'):
#     print(key)
#     ignition_type = group["ignition_type"].iloc[0]
#     xml_name = group["filename"].iloc[0].split(".")[0]
#     fuel_name = group["composition"].iloc[0].split(":")[0]
#     results_path = f"results/idt/plots/{fuel_name}/{ignition_type}/{xml_name}/bad"
#     cantera_related_functions.plot_IDT_vs_T(
#         gas,
#         group,
#         output_path=results_path,
#         interactive=False,
#         save_time_history_plots=False,
    # )
    
for key, group in idt_data_df.groupby('filename'):
    print(key)
    if group["filename"].iloc[0] == "x10000020.xml":
        ignition_type = group["ignition_type"].iloc[0]
        xml_name = group["filename"].iloc[0].split(".")[0]
        fuel_name = group["composition"].iloc[0].split(":")[0]
        results_path = f"results/idt/plots/{fuel_name}/{ignition_type}/{xml_name}"
        cantera_related_functions.plot_IDT_vs_T(
            gas,
            group,
            output_path=results_path,
            interactive=False,
            save_time_history_plots=True,
        )
    


# for key, group in idt_data_df.groupby('filename'):
#     if group["ignition_type"].iloc[0] == "d/dt max":
#         cantera_related_functions.plot_IDT_vs_T(gas, group, save_time_history_plots=False)
    
