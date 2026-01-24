import pandas as pd
import numpy as np
import cantera as ct
import matplotlib.pyplot as plt
from scipy.stats import qmc
from scipy import linalg
import time
import copy

import read_data

mechanism_path= "C:\\Users\\barki\\Desktop\\master thesis\\papers\\mechanism\\ScienceDirect_files_18Dec2025_16-58-12.343\\C2H4_2021.yaml"

uncertain_rxns = [
    {"equation": "C2H4 + O <=> CH3 + HCO", "f_range": [0.135, 0.142]},
    {"equation": "C2H4 + O <=> CH2CO + H2", "f_range": [0.135, 0.142]},
    {"equation": "C2H4 + O <=> CH2CHO + H", "f_range": [0.135, 0.142]},
    {"equation": "C2H4 + OH <=> C2H3 + H2O", "f_range": [0.301, 0.318]},
    {"equation": "C2H4 + OH <=> CH2CH2OH", "f_range": [0.143, 0.149]},
    {"equation": "C2H4 + H (+M) <=> C2H5 (+M)", "f_range": [0.22, 0.22]},
    {"equation": "C2H4 + H <=> C2H3 + H2", "f_range": [0.847, 0.898]},
    {"equation": "C2H5 + O2 <=> C2H4 + HO2", "f_range": [0.231, 0.241]},
    {"equation": "C2H5 + O2 <=> C2H5O2", "f_range": [0.491, 0.511]}
]


def lhs_sampling(rxns, n):
    factors_list = rxns["f_value"].tolist()
    dim = len(factors_list)

    seed = 1327
    sampler = qmc.LatinHypercube(d=dim, seed=seed)
    sample = sampler.random(n=n)
    l_bounds = 1/np.pow(10, factors_list)
    u_bounds = 1*np.pow(10, factors_list)
    qmc.scale(sample, l_bounds, u_bounds)
    return sample

def find_reaction_index_by_equation(gas, equation: str) -> int:
    for i, rxn in enumerate(gas.reactions()):
        if rxn.equation == equation:
            return i
    raise ValueError(f"Reaction not found (exact match): {equation}")

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

def calculate_igintion_delay_at_one_condition(gas, T5, P5, reactants, t_max = 0.1 , ignition_type="d/dtmax"):
    # batch reactor const volume
    
    # Define the reactor temperature and pressure
    reactor_temperature = T5  # Kelvin
    reactor_pressure_atm = P5  # atm
    reactor_pressure = reactor_pressure_atm * 101325  # Pascals

    gas.TPX = reactor_temperature, reactor_pressure, reactants

    reactor = ct.IdealGasReactor(gas, energy='on')  # constant volume if volume not changed
    reactor_network = ct.ReactorNet([reactor])
    
    time_history = ct.SolutionArray(gas, extra="t")

    t0 = time.time()
    t = 0

    counter = 1
    while t < t_max:
        t = reactor_network.step()
        if not counter % 10:
            time_history.append(reactor.thermo.state, t=t)
        counter += 1

    # Ignition delay time defined as the time corresponding to the maximum temperature rise rate
    if ignition_type == "d/dtmax":
        tau = time_history.t[np.argmax(np.gradient(time_history.T, time_history.t))]
    
    t1 = time.time()

    print(f"T5 = {T5} K, P5 = {P5} atm, Computed Ignition Delay: {tau:.3e} seconds. Took {t1-t0:3.2f}s to compute")
    
    return tau

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
gas = ct.Solution('C2H4_2021.yaml')
rxns_df = pd.DataFrame.from_dict(uncertain_rxns)
rxns_df["id"] = rxns_df["equation"].apply(lambda eq: find_reaction_index_by_equation(gas, eq))
#rxns_df["base_A"] = rxns_df["id"].apply(lambda i: get_A(gas.reaction(i)))
rxns_df["f_value"] = rxns_df["f_range"].apply(lambda fr: np.mean(fr))
print("Uncertain reactions with IDs and base A values:")
print(rxns_df)
    

# Now gas has modified reaction rates according to the sampled multipliers
# now we simulate a simple ignition delay
T5_list = [1683, 1621, 1488, 1481, 1475, 1455, 1452, 1439, 1426, 1423,
    1416, 1414, 1412, 1400, 1397, 1388, 1388, 1386, 1357, 1286, 1286]
P5_list = [0.98, 1.16, 1.21, 1.08, 1.19, 1.17, 1.17, 1.14, 1.13, 1.12,
    1.13, 1.11, 1.11, 1.09, 1.10, 1.09, 1.10, 1.08, 1.03, 1.27, 1.17]
measured_IDT = [7.30e-05, 7.80e-05, 1.23e-04, 1.54e-04, 1.49e-04, 1.52e-04, 1.57e-04,
    1.80e-04, 1.85e-04, 2.04e-04, 2.01e-04, 2.17e-04, 1.96e-04, 2.27e-04,
    2.20e-04, 2.36e-04, 2.47e-04, 2.54e-04, 3.18e-04, 5.37e-04, 4.99e-04]
reactants = 'C2H4:0.01, O2:0.03, AR:0.96'
# T5_list = [
#     1228, 1276, 1295, 1331, 1341, 1388, 1416, 1468, 1487,
#     1507, 1530, 1554, 1625, 1647, 1652, 1732, 1746
# ]
# P5_list = [
#     3.26, 3.16, 2.90, 3.05, 2.91, 2.88, 2.99, 2.77, 3.02,
#     2.77, 2.96, 2.73, 2.79, 2.61, 2.72, 2.61, 2.61
# ]
# measured_IDT = [
#     6.90e-04, 3.65e-04, 3.06e-04, 2.01e-04, 1.85e-04, 1.19e-04,
#     9.23e-05, 6.69e-05, 5.63e-05, 5.29e-05, 4.41e-05,
#     4.29e-05, 3.09e-05, 3.06e-05, 2.90e-05, 2.36e-05, 2.28e-05
# ]

#ignition_delay_times = calculate_ignition_delay_constV_batch_reactor(gas, T5_list, P5_list, reactants, meas_IDT=measured_IDT, plot=True)




# ok that works. NOw
# get operationg conditions from real data to make it meaningful 
operating_condition_range = [[700,1600], [1,40], [0.5,3]]  # T5 in K, P5 in atm, phi

idt_data_folder = "C:\\Users\\barki\\Desktop\\master thesis\\reaction-data-consistency-thesis\\surrogate\\idt_data\\xmls"
idt_data_df = read_data.extract_idt_data_to_dataframe(idt_data_folder)

operating_conditions_df = stratified_sample_operating_conditions(idt_data_df, n_T_bins=5, n_logP_bins=4, n_phi_bins=4, cap_per_bin=1, random_state=42)
print(operating_conditions_df)
operating_conditions = [[1683, 0.98, 'C2H4:0.01, O2:0.03, AR:0.96'],
                       [1286, 1.17, 'C2H4:0.01, O2:0.03, AR:0.96']]  # T5 in K, P5 in atm, 
param_multipliers_samples = lhs_sampling(rxns_df, len(rxns_df)*3)
if_df = pd.DataFrame()
for operating_condition in operating_conditions:
    ls_data = []
    for multipliers_sample in param_multipliers_samples:
        multiply_all_A(gas, rxns_df, multipliers_sample, method="cantera_built_in")
        tau = calculate_igintion_delay_at_one_condition(gas=gas, T5=operating_condition[0], P5=operating_condition[1], reactants=operating_condition[2], t_max=0.5, ignition_type="d/dtmax")
        ls_data.append([multipliers_sample, tau])

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
inactive_eqs = if_df.groupby("equation")["low_impact"].sum().eq(len(operating_conditions))
if_df["inactive"] = if_df["equation"].map(inactive_eqs).fillna(False)

if_df.to_csv("surrogate/impact_factors_ignition_delay.csv", index=False)
print(if_df)

  
  
  
