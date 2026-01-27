import json
import sys
import SALib.sample.morris as morris_sample
import SALib.analyze.morris as morris_analyze
import numpy as np
import pandas as pd
import cantera as ct
import matplotlib.pyplot as plt
from scipy.stats import qmc
from scipy import linalg
import time
import copy

import log_runs
import cantera_related_functions
# for a given mecahnsim, operating conditions range and QOI data. 
# find which reactions are inactive for the given QOI across operating range


def reduce_mech_after_morris(df, mu_star_treshld=0.05, mu_star_conf_treshld=0.9):
    """_summary_

    Args:
        df (_type_): morris analysis sensitivity dataframe
        mu_star_treshld (float, optional): Keep reactions with relative mu_star above this threshold. Defaults to 0.02.
        mu_star_conf_treshld (float, optional): Keep reactions with relative mu_star_conf above this threshold. Defaults to 0.8.
    """
    max_mu_star = df["mu_star"].max()
    print(f"Maximum mu_star value is {max_mu_star:.4f}")
    active_reactions = df[(df["mu_star"] >= max_mu_star * mu_star_treshld)]
    uncertain_reactions = df[(df["mu_star_conf"] >= df["mu_star"] * mu_star_conf_treshld)]
    print("number of Possibly Active reactions based on Morris sensitivity analysis:")
    print(len(active_reactions))
    print("number of Possibly Uncertain reactions based on Morris sensitivity analysis:")
    print(len(uncertain_reactions))
    df_filtered = pd.concat([active_reactions, uncertain_reactions]).drop_duplicates().reset_index(drop=True)
    print("Number of reactions that are active or uncertain:")
    print(len(df_filtered))
    print("Rest of the reactions can be considered inactive and certain.")
    inactive_reactions = df[~df.index.isin(df_filtered.index)]
    # print(inactive_reactions)
    return df_filtered, inactive_reactions

def sanity_check_morris_screening(gas, active_reactions_df, inactive_reactions_df, operating_condition):
    """_summary_

    Args:
        active_reactions_df (_type_): morris analysis sensitivity dataframe for active reactions
        inactive_reactions_df (_type_): morris analysis sensitivity dataframe for inactive reactions
    """
    
    print("Sanity check of Morris screening by perturbing reaction rates and checking IDT changes.")
    calc_IDT_constV(gas=gas, operating_condition=operating_condition, t_max=0.001, ignition_target = "T", ignition_type="d/dt max")
    for reaction_name in inactive_reactions_df["reaction_equation"]:
        rxn_index = find_reaction_index_by_equation(gas, reaction_name)
        if rxn_index is not None:
             gas.set_multiplier(0.3, i_reaction=rxn_index)
    print("IDT After perturbing inactive reactions:")
    calc_IDT_constV(gas=gas, operating_condition=operating_condition, t_max=0.001, ignition_target = "T", ignition_type="d/dt max")
    reset_rates(gas)
    for reaction_name in active_reactions_df["reaction_equation"]:
        rxn_index = find_reaction_index_by_equation(gas, reaction_name)
        if rxn_index is not None:
             gas.set_multiplier(0.3, i_reaction=rxn_index)
    print("IDT After perturbing active reactions:")
    calc_IDT_constV(gas=gas, operating_condition=operating_condition, t_max=0.001, ignition_target = "T", ignition_type="d/dt max")
    reset_rates(gas)



if __name__ == "__main__":
    
    config = {
        "seed": 42,
        "note": "morris_sensitivity_analysis",
    }
    run_dir = log_runs.setup_run(config, prefix="sensitivity_results/exp_")
    
    
    #Setup mechanism
    mech_file = 'C2H4_2021.yaml'
    mech_name = mech_file.split('.')[0]
    gas = ct.Solution(mech_file)
    no_reactions = len(gas.reactions())

    # Define the model inputs
    problem = {
        'num_vars': no_reactions,
        'names': [f'{gas.reaction_equations()[i]}' for i in range(no_reactions)],
        'bounds': [[-0.1, 0.1]] * no_reactions
    }


    # Generate samples
    param_values = morris_sample.sample(problem, N=10, num_levels=4, optimal_trajectories=None)
    print(param_values.shape)
    print(param_values)

    operating_conditions = [[1683, 0.98, 'C2H4:0.01, O2:0.03, AR:0.96'],
                        [1286, 1.17, 'C2H4:0.01, O2:0.03, AR:0.96']]  # T5 in K, P5 in atm, 
    operating_condition = operating_conditions[0]   

    Y = np.zeros([param_values.shape[0]])

    # Run model (example)
    for i, param_set in enumerate(param_values):
        exp_params = np.power(10, param_set)  # convert from log10 space
        cantera_related_functions.multiply_rates(gas, exp_params, rxn_ids=None, method="cantera_built_in")
        Y[i] = cantera_related_functions.calc_IDT_constV(gas=gas, operating_condition=operating_condition, t_max=0.001, ignition_target = "T", ignition_type="d/dt max")
        if i % 20 == 0:
            print(f"Completed {i} of {param_values.shape[0]} simulations.")
            
    #dummy run to generate Y
    for i, param_set in enumerate(param_values):
        if i != 5:
            Y[i] = 0.01 * i  # dummy data
        else:
            Y[i] = 0.5  # outlier data point

    Si = morris_analyze.analyze(problem, param_values, Y, print_to_console=True)

    sens_df = Si.to_df()
    sens_df["reaction_equation"] = sens_df.index
    print(sens_df)
    
    active_reactions, inactive_reactions = reduce_mech_after_morris(sens_df, mu_star_treshld=0.05, mu_star_conf_treshld=0.9)   
    #sanity_check_morris_screening(gas, active_reactions, inactive_reactions, operating_condition)
    
    # Write outputs wherever you like under run_dir
    results_path = run_dir / f"morris_sensitivity_ignition_delay_{mech_name}.csv"
    sens_df.to_csv(results_path)
    print("done")


