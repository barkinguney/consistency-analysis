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



def lhs_sampling(factors_list, n):
    dim = len(factors_list)

    seed = 1327
    sampler = qmc.LatinHypercube(d=dim, seed=seed)
    sample = sampler.random(n=n)
    l_bounds = 1/np.pow(10, factors_list)
    u_bounds = 1*np.pow(10, factors_list)
    qmc.scale(sample, l_bounds, u_bounds)
    return sample

def get_A(ct_rxn, operating_conditions=None):
    if ct_rxn.reaction_type == 'Arrhenius' or ct_rxn.reaction_type == 'three-body-Arrhenius':
        A = [ct_rxn.rate.pre_exponential_factor]
    elif ct_rxn.reaction_type == 'falloff-Troe' or ct_rxn.reaction_type == 'falloff-Lindemann':
        A_low = ct_rxn.rate.low_rate.pre_exponential_factor
        A_high = ct_rxn.rate.high_rate.pre_exponential_factor
        A = [A_low, A_high]
    elif ct_rxn.reaction_type == 'pressure-dependent-Arrhenius':
        A = [p[1].pre_exponential_factor for p in ct_rxn.rate.rates]
    else:
        raise ValueError(f"Reaction has unhandled reaction type: {ct_rxn.reaction_type}")
    return A
   

def multiply_A_in_dict(d, base_A, m):
    rtype = d.get("type", None)

    # Elementary (no "type") or explicit elementary
    if rtype is None or rtype == "elementary":
        d["rate-constant"]["A"] = base_A[0]* m
        return d

    # Three-body
    if rtype == "three-body":
        d["rate-constant"]["A"] = base_A[0] * m
        return d

    # Falloff (Troe or Lindemann): multiply both limits
    if rtype == "falloff":
        d["low-P-rate-constant"]["A"]  = base_A[0] * m
        d["high-P-rate-constant"]["A"] = base_A[1] * m
        return d

    # Pressure-dependent Arrhenius (PLOG)
    if rtype == "pressure-dependent-Arrhenius":
        for i, entry in enumerate(d["rate-constants"]):
            entry["A"] = base_A[i] * m
        return d

    raise NotImplementedError(f"Unsupported reaction dict type={rtype}")

def apply_multiplier_to_reaction(gas, base_A, i, m):
    old = gas.reaction(i)
    d = dict(old.input_data)

    d2 = multiply_A_in_dict(d, base_A, m)
    new = ct.Reaction.from_dict(d2, gas)
    gas.modify_reaction(i, new)
    #print(f"Applied multiplier {m} to reaction {i}: {old.equation}")
    
def multiply_all_A(gas, rxns_df, m_sample, method="cantera_built_in"):
    if method == "cantera_built_in":
        for idx, rxn in enumerate(rxns_df.itertuples()):
            gas.set_multiplier(value=m_sample[idx], i_reaction=rxn.id)
            
    if method == "manual_A_modification":
        for idx, rxn in enumerate(rxns_df.itertuples()):
            apply_multiplier_to_reaction(gas, rxn.base_A, rxn.id, m_sample[idx])

def calculate_ignition_delay_constV_batch_reactor(gas, T5_list, P5_list, reactants, meas_IDT = None, plot=False, t_max = 0.1, ignition_type="d/dtmax"):
    # batch reactor const volume
    ignition_delay_times = []
    for T5, P5 in zip(T5_list, P5_list):
        tau = calculate_igintion_delay_at_one_condition(gas, T5, P5, reactants, t_max , ignition_type)
        ignition_delay_times.append(tau)
        
    if plot:
        scaled_inverse_T5 = [1000.0 / T for T in T5_list]

        plt.figure(figsize=(8, 6))
        if meas_IDT is not None:
            plt.scatter(scaled_inverse_T5, meas_IDT, label='real experiment', color='red')
        plt.scatter(scaled_inverse_T5, ignition_delay_times, label='Cantera', color='blue')
        plt.xlabel('1000/T5 (1/K)')
        plt.ylabel('Ignition Delay Time (s)')
        plt.yscale('log')
        plt.legend()
        plt.title('Ignition Delay Times Comparison')
        plt.grid(True)
        plt.show()
        
    return ignition_delay_times

def fit_linear_least_squares(X: np.ndarray, y: np.ndarray):
    """
    Fit y ≈ c0 + sum_i c_i x_i via least squares.

    Parameters
    ----------
    X : (N, d) array
        Design matrix of sampled parameters (e.g., log10 multipliers).
        Each row is one LHS sample x^(k).
    y : (N,) or (N, 1) array
        Model outputs at ONE fixed operating condition (e.g., log(IDT)).

    Returns
    -------
    c0 : float
        Intercept.
    c : (d,) array
        Coefficients.
    y_hat : (N,) array
        Fitted values.
    residuals : (N,) array
        y - y_hat.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)

    if X.ndim != 2:
        raise ValueError("X must be 2D with shape (N, d)")
    if y.shape[0] != X.shape[0]:
        raise ValueError(f"y length ({y.shape[0]}) must match X rows ({X.shape[0]})")

    N, d = X.shape

    # Add intercept column
    A = np.hstack([np.ones((N, 1)), X])  # (N, d+1)

    # Solve min ||A beta - y||_2
    beta, *_ = linalg.lstsq(A, y)  # beta shape: (d+1,)

    c0 = float(beta[0])
    c = beta[1:].copy()

    y_hat = A @ beta
    residuals = y - y_hat
    return c0, c, y_hat, residuals

def stratified_sample_operating_conditions(
    df,
    n_T_bins=5,
    n_logP_bins=4,
    n_phi_bins=4,
    cap_per_bin=1,
    random_state=0
):
    """
    Stratified sampling of operating conditions from experimental data.

    Parameters
    ----------
    df : pandas.DataFrame
        Must contain columns ['T5', 'P5', 'phi']
    n_T_bins : int
        Number of bins for T (linear)
    n_logP_bins : int
        Number of bins for log(P)
    n_phi_bins : int
        Number of bins for phi (linear)
    cap_per_bin : int
        Max number of samples per occupied bin
    random_state : int
        RNG seed

    Returns
    -------
    sampled_df : pandas.DataFrame
        Subset of df selected via stratified binning
    """

    rng = np.random.default_rng(random_state)

    df = df.copy()

    # ---- compute transformed coordinates ----
    df["logP"] = np.log(df["P5"])

    # ---- bin edges from data min/max ----
    T_edges = np.linspace(df["T5"].min(), df["T5"].max(), n_T_bins + 1)
    logP_edges = np.linspace(df["logP"].min(), df["logP"].max(), n_logP_bins + 1)
    phi_edges = np.linspace(df["phi"].min(), df["phi"].max(), n_phi_bins + 1)

    # ---- assign bins ----
    df["T_bin"] = pd.cut(df["T5"], bins=T_edges, include_lowest=True, labels=False)
    df["logP_bin"] = pd.cut(df["logP"], bins=logP_edges, include_lowest=True, labels=False)
    df["phi_bin"] = pd.cut(df["phi"], bins=phi_edges, include_lowest=True, labels=False)

    # ---- group by bins ----
    grouped = df.groupby(["T_bin", "logP_bin", "phi_bin"])

    sampled_indices = []

    for _, group in grouped:
        if len(group) == 0:
            continue

        # pick up to cap_per_bin samples
        if len(group) <= cap_per_bin:
            sampled_indices.extend(group.index.tolist())
        else:
            sampled_indices.extend(
                rng.choice(group.index, size=cap_per_bin, replace=False).tolist()
            )

    sampled_df = df.loc[sampled_indices].drop(
        columns=["logP", "T_bin", "logP_bin", "phi_bin"]
    )

    return sampled_df

#Setup mechanism
gas = ct.Solution('Supplementary-3_syngas.yaml')
no_reactions = len(gas.reactions())
prior_uncertainty_factor = 0.3

rxns_df = pd.DataFrame()
rxns_df["equation"] = gas.reaction_equations()
rxns_df["id"] = rxns_df["equation"].apply(lambda eq: cantera_related_functions.find_reaction_index_by_equation(gas, eq))
rxns_df["f_value"] = prior_uncertainty_factor



# ok that works. NOw
# get operationg conditions from real data to make it meaningful 
idt_data_folders = ["data\\idt_data\\hydrogen", "data\\idt_data\\syngas"]
idt_data_df = pd.DataFrame()
for idt_data_folder in idt_data_folders:
    idt_data_df = pd.concat([idt_data_df, read_data.extract_idt_data_to_dataframe(idt_data_folder)], ignore_index=True)
    
operating_conditions_df = stratified_sample_operating_conditions(idt_data_df, n_T_bins=4, n_logP_bins=3, n_phi_bins=3, cap_per_bin=1, random_state=42)
print(operating_conditions_df.columns.tolist())
print(operating_conditions_df)

uncertainty_factors = np.full(no_reactions, prior_uncertainty_factor)
param_multipliers_samples = lhs_sampling(factors_list=uncertainty_factors, n=no_reactions*3)
if_df = pd.DataFrame()

for condition_idx, operating_condition in enumerate(operating_conditions_df.itertuples(index=False)):
    ls_data = []
    for sample_idx, multipliers_sample in enumerate(param_multipliers_samples):
        cantera_related_functions.multiply_rates(gas, multipliers_sample, rxn_ids=None, method="cantera_built_in")
        tau = cantera_related_functions.calc_IDT_constV(gas=gas, operating_condition=operating_condition, t_max=0.01)
        ls_data.append([multipliers_sample, tau])
        if sample_idx % 20 == 0:
            print(f"Completed {sample_idx} of {param_multipliers_samples.shape[0]} simulations for operating condition {condition_idx} of {len(operating_conditions_df)}.")

    # now we do ls fit 
    X = np.array([row[0] for row in ls_data])
    y = np.log10(np.array([row[1] for row in ls_data])) # fit log(IDT) because we sample log multipliers for parameters
    c0, c, y_hat, residuals = fit_linear_least_squares(X, y)
    print("Intercept:", c0)
    print("Coefficients:", c)
    
    temp_df = rxns_df.copy(deep=True)
    
    #now we can calculate impact factors
    # If I vary parameter 𝑖 over its full uncertainty range, how much could IDT change?”
    temp_df["operating_condition"] = str(operating_condition)
    temp_df["if"] = np.abs(c) * temp_df["f_value"] *2 * np.log(10)  # factor of 2 because we consider +/-f_value range
    temp_df["low_impact"] = temp_df["if"] < temp_df["if"].max() * 0.1 # below 10% within a sample mechanism is low impact for that operating conditoin, 
    
    # store impact factors for this operating condition
    if_df = pd.concat([if_df, temp_df]).applymap(copy.deepcopy)

if_df.reset_index(drop=True, inplace=True)
# now we aggragete impact factors over operating conditions
#if a reaction is low impact in all operating conditoions its inactive
inactive_eqs = if_df.groupby("equation")["low_impact"].sum().eq(len(operating_conditions_df))
if_df["inactive"] = if_df["equation"].map(inactive_eqs).fillna(False)

if_df.to_csv("surrogate/impact_factors_ignition_delay.csv", index=False)
print(if_df)

  
  
  
