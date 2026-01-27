import numpy as np
import pandas as pd
import cantera as ct
import matplotlib.pyplot as plt
import time
import copy

import sensitivity


if __name__ == "__main__":
    #Setup mechanism
    mech_file = 'C2H4_2021.yaml'
    mech_name = mech_file.split('.')[0]
    gas = ct.Solution(mech_file)
    no_reactions = len(gas.reactions())

    operating_conditions = [[1683, 0.98, 'C2H4:0.01, O2:0.03, AR:0.96'],
                        [1286, 1.17, 'C2H4:0.01, O2:0.03, AR:0.96']]  # T5 in K, P5 in atm, 
    operating_condition = operating_conditions[0]   


    si_df = pd.read_csv("surrogate/analysis_results/morris_sensitivity_ignition_delay_C2H4_2021.csv")
    active_reactions, inactive_reactions = sensitivity.reduce_mech_after_morris(si_df, mu_star_treshld=0.05, mu_star_conf_treshld=0.9)
    sensitivity.sanity_check_morris_screening(gas, active_reactions, inactive_reactions, operating_condition)

