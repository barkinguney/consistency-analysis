import numpy as np
import pandas as pd
import cantera as ct
import matplotlib.pyplot as plt


if __name__ == "__main__":
    # #Setup mechanism
    # mech_file = 'C2H4_2021.yaml'
    # mech_name = mech_file.split('.')[0]
    # gas = ct.Solution(mech_file)
    # no_reactions = len(gas.reactions())

    # operating_conditions = [[1683, 0.98, 'C2H4:0.01, O2:0.03, AR:0.96'],
    #                     [1286, 1.17, 'C2H4:0.01, O2:0.03, AR:0.96']]  # T5 in K, P5 in atm, 
    # operating_condition = operating_conditions[0]   


    if_df = pd.read_csv("surrogate/impact_factors_ignition_delay.csv")
    #print(if_df)
    operating_consitions =if_df["operating_condition"].unique()
    for condition in operating_consitions:
        print(f"Operating condition: {condition}")
        cond_if_df = if_df[if_df["operating_condition"] == condition]
        max_if = cond_if_df["if"].max()
        significant_reactions = cond_if_df[cond_if_df["if"] >= 0.08 * max_if]
        print(significant_reactions)
        
        
        
    print(if_df[if_df["id"] == 8])