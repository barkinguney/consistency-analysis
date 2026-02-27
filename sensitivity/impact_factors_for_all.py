import pandas as pd
import numpy as np
import cantera as ct
import matplotlib.pyplot as plt
from scipy.stats import qmc
from scipy import linalg
import time
import copy
from pathlib import Path
import re

import read_data
import cantera_related_functions



def lhs_sampling(factors_list, n):
    # log uniform samples
    dim = len(factors_list)

    seed = 1327
    sampler = qmc.LatinHypercube(d=dim, seed=seed)
    sample = sampler.random(n=n)
    z = qmc.scale(sample, -factors_list, factors_list)
    m = 10.0 ** z
    return m
    
    # l_bounds = 1/np.pow(10, factors_list)
    # u_bounds = 1*np.pow(10, factors_list)
    # scaled_sample = qmc.scale(sample, l_bounds, u_bounds)
    # return scaled_sample

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


def _safe_condition_tag(condition_idx, operating_condition):
    t5 = getattr(operating_condition, "T5", None)
    p5 = getattr(operating_condition, "P5", None)
    phi = getattr(operating_condition, "phi", None)

    if t5 is not None and p5 is not None and phi is not None:
        tag = f"cond_{condition_idx:03d}_T{float(t5):.1f}_P{float(p5):.3f}_phi{float(phi):.3f}"
    else:
        tag = f"cond_{condition_idx:03d}"

    return re.sub(r"[^A-Za-z0-9._-]", "_", tag)


def save_sensitivity_outputs_for_condition(temp_df, condition_idx, operating_condition, output_dir):
    """
    Save sensitivity values for one operating condition as:
    - CSV table listing reactions and sensitivity values
    - Horizontal bar plot with reaction equations and sensitivity values
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    condition_tag = _safe_condition_tag(condition_idx, operating_condition)

    condition_df = (
        temp_df[["equation", "if", "low_impact"]]
        .sort_values("if", ascending=False)
        .reset_index(drop=True)
    )
    condition_df["rank"] = np.arange(1, len(condition_df) + 1)

    csv_path = output_dir / f"{condition_tag}_sensitivity.csv"
    condition_df.to_csv(csv_path, index=False)

    top_n = 20
    plot_df = condition_df.head(top_n)

    fig_height = max(6, 0.35 * len(plot_df))
    plt.figure(figsize=(12, fig_height))
    y_labels = plot_df["equation"].tolist()
    x_values = plot_df["if"].to_numpy(dtype=float)
    y_pos = np.arange(len(plot_df))

    plt.barh(y_pos, x_values, color="tab:blue")
    plt.yticks(y_pos, y_labels, fontsize=8)
    plt.gca().invert_yaxis()
    plt.xlabel("Sensitivity value (impact factor)")

    if hasattr(operating_condition, "T5") and hasattr(operating_condition, "P5") and hasattr(operating_condition, "phi"):
        plt.title(
            f"Top {min(top_n, len(condition_df))} sensitivity reactions | Condition {condition_idx} "
            f"(T5={operating_condition.T5:.1f} K, P5={operating_condition.P5:.3f}, phi={operating_condition.phi:.3f})"
        )
    else:
        plt.title(f"Top {min(top_n, len(condition_df))} sensitivity reactions | Condition {condition_idx}")

    x_offset = max(np.max(x_values) * 0.01, 1e-12)
    for y, x in zip(y_pos, x_values):
        plt.text(x + x_offset, y, f"{x:.3e}", va="center", fontsize=7)

    plt.grid(axis="x", alpha=0.25)
    plt.tight_layout()

    fig_path = output_dir / f"{condition_tag}_sensitivity.png"
    plt.savefig(fig_path, dpi=300)
    plt.close()



if __name__ == "__main__":

    #Setup mechanism
    gas = ct.Solution('Supplementary-3_syngas.yaml')
    # get operationg conditions from real data to make it meaningful
    idt_data_folders = ["data\\idt_data\\hydrogen", "data\\idt_data\\syngas"]
    idt_unc_data_folders = ["results\\idt_unc\\hydrogen", "results\\idt_unc\\syngas"]
    # gas = ct.Solution('C2H4_2021.yaml')
    # # get operationg conditions from real data to make it meaningful
    # idt_data_folders = ["data\\idt_data\\ethylene"]
    
    no_reactions = len(gas.reactions())
    prior_uncertainty_factor = 0.1

    rxns_df = pd.DataFrame()
    rxns_df["equation"] = gas.reaction_equations()
    print(rxns_df["equation"].tolist())
    rxns_df["id"] = rxns_df["equation"].apply(lambda eq: cantera_related_functions.find_reaction_index_by_equation(gas, eq))
    rxns_df["f_value"] = prior_uncertainty_factor


    
    idt_data_df = pd.DataFrame()
    for idt_data_folder in idt_data_folders:
        idt_data_df = pd.concat([idt_data_df, read_data.extract_idt_data_to_dataframe(idt_data_folder)], ignore_index=True)
    
    idt_data_df.to_csv("sensitivity/results/idt_data_for_sensitivity_analysis.csv", index=False)
    operating_conditions_df = stratified_sample_operating_conditions(idt_data_df, n_T_bins=3, n_logP_bins=3, n_phi_bins=3, cap_per_bin=1, random_state=42)
    for condition in operating_conditions_df.itertuples(index=False):
        unc_file_name = Path(condition.filename).stem + f".csv"
        for idt_unc_data_folder in idt_unc_data_folders:
            unc_file_path = Path(idt_unc_data_folder) / unc_file_name
            if unc_file_path.exists():
                unc_df = pd.read_csv(unc_file_path)
                
                if (unc_df["expanded_relative"] > 80.0).any():
                    unc = 80.0
                else:
                    cond =  unc_df[unc_df['T5_K'] == condition.T5]
                    unc = cond["expanded_relative"].values[0]
                operating_conditions_df.loc[operating_conditions_df["filename"] == condition.filename, "exp_unc"] = unc  
                break
    operating_conditions_df["exp_unc"] = operating_conditions_df["exp_unc"].fillna(80.0)         
    operating_conditions_df.to_csv("sensitivity/results/operating_conditions.csv", index=False)
    print(operating_conditions_df.columns.tolist())

    uncertainty_factors = np.full(no_reactions, prior_uncertainty_factor)
    param_multipliers_samples = lhs_sampling(factors_list=uncertainty_factors, n=no_reactions*3)
    if_df = pd.DataFrame()

    per_condition_output_dir = Path("sensitivity/results/per_condition_sensitivity")

    for condition_idx, operating_condition in enumerate(operating_conditions_df.itertuples(index=False)):
        
        # t_max = 0.1
        # try:
        #     tau = cantera_related_functions.calc_IDT_constV(gas=gas, operating_condition=operating_condition, t_max=t_max, debug=True)
        # except ValueError as e:
        #     print(f"Error calculating IDT for operating condition {operating_condition}: {e}")
        #     tau = t_max
        ls_data = []
        for sample_idx, multipliers_sample in enumerate(param_multipliers_samples):
            cantera_related_functions.multiply_rates(gas, multipliers_sample, rxn_ids=None, method="cantera_built_in")
            t_max = 0.1
            try:
                tau = cantera_related_functions.calc_IDT_constV(gas=gas, operating_condition=operating_condition, t_max=t_max)
            except ValueError as e:
                print(f"Error calculating IDT for operating condition {operating_condition}: {e}")
                tau = t_max
            ls_data.append([multipliers_sample, tau])
            if sample_idx % 20 == 0:
                print(f"Completed {sample_idx} of {param_multipliers_samples.shape[0]} simulations for operating condition {condition_idx} of {len(operating_conditions_df)}.")

        # now we do ls fit 
        # X = np.array([row[0] for row in ls_data])
        # y = np.log10(np.array([row[1] for row in ls_data])) # fit log(IDT) because we sample log multipliers for parameters
        X = np.log10(np.array([row[0] for row in ls_data]))  # log10 multipliers
        y = np.log10(np.array([row[1] for row in ls_data]))  # log10 IDT
        c0, c, y_hat, residuals = fit_linear_least_squares(X, y)
        # print("Intercept:", c0)
        # print("Coefficients:", c)
        
        temp_df = rxns_df.copy(deep=True)
        
        #now we can calculate impact factors
        # If I vary parameter 𝑖 over its full uncertainty range, how much could IDT change?”
        impact_threshold = 0.1 # 
        temp_df["operating_condition"] = str(operating_condition)
        temp_df["if"] = np.abs(c) * temp_df["f_value"] * 2  #* np.log(10)  # factor of 2 because we consider +/-f_value range
        temp_df["rank"] = temp_df["if"].rank(ascending=False, method="first")
        temp_df["low_impact"] = temp_df["if"] < (temp_df["if"].max() * impact_threshold)

        save_sensitivity_outputs_for_condition(
            temp_df=temp_df,
            condition_idx=condition_idx,
            operating_condition=operating_condition,
            output_dir=per_condition_output_dir,
        )
        
        # store impact factors for this operating condition
        if_df = pd.concat([if_df, temp_df], ignore_index=True, copy=True)

    if_df.reset_index(drop=True, inplace=True)
    # now we aggragete impact factors over operating conditions
    #if a reaction is low impact in all operating conditoions its inactive
    inactive_type = "not_top5_in_any"
    if inactive_type == "low_impact_in_all":
        inactive_eqs = if_df.groupby("equation")["low_impact"].sum().eq(len(operating_conditions_df))
    if inactive_type == "not_top5_in_any":
        inactive_eqs = if_df.groupby("equation")["rank"].min().gt(5)
    if_df["inactive"] = if_df["equation"].map(inactive_eqs).fillna(False)
    active_df = if_df[~if_df["inactive"]].copy()
    #active_df['rank'] = active_df.groupby("operating_condition")["if"].rank(ascending=False, method="first")
    
    unique_active_eqs = active_df["equation"].unique()
    pd.DataFrame(unique_active_eqs, columns=["equation"]).to_csv("sensitivity/results/unique_active_eqs.csv", index=False)
    
    print(f"Identified {len(unique_active_eqs)} active reactions out of {len(rxns_df)} across all operating conditions:")
    for eq in unique_active_eqs:
        print(f" - {eq}")
         
    active_df.to_csv("sensitivity/results/impact_factors_ignition_delay_active_reactions.csv", index=False)
    if_df.to_csv("sensitivity/results/impact_factors_ignition_delay.csv", index=False)
    
    

  
  
  
