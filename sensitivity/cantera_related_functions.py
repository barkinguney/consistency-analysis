
import json
import sys
import numpy as np
import pandas as pd
import cantera as ct
import matplotlib.pyplot as plt
from scipy.stats import qmc
from scipy import linalg
from scipy.signal import find_peaks
import time
import copy
import os
import plotly.graph_objects as go
from pathlib import Path

def multiply_rates(gas, m_sample, rxn_ids = None, method="cantera_built_in"):
    if rxn_ids is None:
        rxn_ids = range(len(gas.reactions()))

    if method == "cantera_built_in":
        for idx in rxn_ids:
            gas.set_multiplier(value=m_sample[idx], i_reaction=idx)
            
def reset_rates(gas, rxn_ids = None):
    if rxn_ids is None:
        rxn_ids = range(len(gas.reactions()))
    for idx in rxn_ids:
        gas.set_multiplier(1.0, i_reaction=idx)

def find_reaction_index_by_equation(gas, equation: str) -> int:
    for i, rxn in enumerate(gas.reactions()):
        if rxn.equation == equation:
            return i   
    print(f"Warning: Reaction not found (exact match): {equation}", file=sys.stderr) 



def calc_IDT_constV(gas, operating_condition, t_max = 0.001, save_time_history_plot = False):
    """Calculate ignition delay time at given operating condition.
    batch reactor const volume"""
    
    def parse_ignition_target(ignition_target: str):
        ignition_targets = ignition_target.split(";", 1)
        target = ignition_targets[0]
        if len(ignition_targets) > 1 and ignition_targets[1] != "":
            print(f"Warning: Multiple ignition targets specified ({ignition_target}). Using the first one: {target}")
        if "EX" in target:
            target = target.replace("EX", "*")
        return target
    
    def get_species_indices(gas):
        species_indices = {name: i for i, name in enumerate(gas.species_names)}
        return species_indices
    
    def convert_units(P5, P5_unit,ignition_amount, ignition_units): 
        """convert pressure to Pa and ignition concentration to kmol/m3"""
        if P5_unit.lower() in ['kpa']:
            P5_pa = P5 * 1000
        elif P5_unit.lower() in ['atm']:
            P5_pa = P5 * 101325
        elif P5_unit.lower() in ['bar']:
            P5_pa = P5 * 100000
        elif P5_unit.lower() in ["torr"]:
            P5_pa = P5 * 133.322
        else:
            raise ValueError(f"Unknown pressure unit: {P5_unit}")

        # Ignition amount conversion can be added here if needed
        if ignition_units.lower() in ["mol/cm3"]:
            ignition_amount = ignition_amount * 1e-3
        if ignition_units.lower() in ["kmol/m3", "unitless", ""]:
            pass
        else:
            raise ValueError(f"Unknown ignition amount unit: {ignition_units}")

        return P5_pa, ignition_amount
    
    def save_time_history_plot_to_file(time_history, ignition_target, operating_condition, tau):
        xml_name = operating_condition[12].split(".")[0]
        fuel_name = operating_condition[4].split(":")[0]
        results_path = f"results/idt/plots/{fuel_name}/{xml_name}"
        os.makedirs(results_path, exist_ok=True)
        if False:
            plt.figure()
            plt.plot(time_history.t, getattr(time_history, ignition_target))
            plt.xlabel('Time (s)')
            plt.ylabel(ignition_target)
            plt.xlim(0, tau*1.5)
            plt.axvline(x=tau, color='r', linestyle='--', label=f'Ignition Delay Time: {tau:.6f} s')
            plt.title(f"Time History of {ignition_target} for Operating Condition: {operating_condition}")
            plt.savefig(f"{results_path}/time_history_{operating_condition[0]}{operating_condition[1]}_{operating_condition[2]}{operating_condition[3]}.svg")
        
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=time_history.t,
                y=getattr(time_history, ignition_target),
                mode="lines",
                name=ignition_target
            )
        )
        fig.add_vline(
            x=tau,
            line=dict(color="red", dash="dash"),
            annotation_text=f"IDT = {tau:.6f} s",
            annotation_position="top"
        )
        fig.update_layout(
            title=f"Time History of {ignition_target} for Operating Condition: {operating_condition}",
            xaxis_title="Time (s)",
            yaxis_title=ignition_target,
            xaxis=dict(range=[0, tau * 1.5]),
            template="plotly_white"
        )
        fig.write_html(f"{results_path}/time_history_{operating_condition[0]}{operating_condition[1]}_{operating_condition[2]}{operating_condition[3]}.html")
        
        
        
    T5 = operating_condition.T5  # Kelvin
    P5, ignition_amount = convert_units(operating_condition.P5, operating_condition.P5_units, operating_condition.ignition_amount, operating_condition.ignition_units)
    if ignition_amount != "":   
        ignition_amount = float(ignition_amount)
    composition = operating_condition.composition  # Cantera composition string
    ignition_target = parse_ignition_target(operating_condition.ignition_target)
    ignition_type = operating_condition.ignition_type

    gas.TPX = T5, P5, composition
    # constant volume if volume not changed
    reactor = ct.IdealGasReactor(gas, energy='on', clone=True)  
    reactor_network = ct.ReactorNet([reactor])
    
    time_history = ct.SolutionArray(reactor.phase, extra="t")

    t0 = time.time()
    t = 0

    counter = 1
    while t < t_max:
        t = reactor_network.step()
        if not counter % 10:
            time_history.append(reactor.phase.state, t=t)
        counter += 1

    # Ignition delay time defined as the time corresponding to the maximum temperature rise rate

    
    # TARGET_MAP = {
    #     "T": lambda th: th.T,
    #     "P": lambda th: th.P,
    # }
    # TYPE_MAP = {
    #     "max": lambda y, t: np.nanargmax(y),
    #     "d/dt max": lambda y, t: np.nanargmax(np.gradient(y, t)),
    # }
    
    # tau = time_history.t[np.argmax(np.gradient(time_history.T, time_history.t))]
    #target_conc = ignition_type[1]
    #tau = time_history.t[np.nanargmin(np.abs(target_conc - time_history[ignition_target]))]
    
    # try:
    #     y = TARGET_MAP[ignition_target](time_history)
    # except KeyError:
    #     raise ValueError(f"Unknown ignition_target={ignition_target!r}")
    # try:
    #     idx = TYPE_MAP[ignition_type[0]](y, time_history.t)
    # except KeyError:
    #     raise ValueError(f"Unknown ignition_type={ignition_type!r}")   
    
    
    print(f"Calculating IDT using target: {ignition_target}, type: {ignition_type}")
    
    species_idxs = get_species_indices(reactor.phase)
    
    if ignition_target == "T" :
        "TODO: reuse species code"
        if ignition_type == "max":
            tau = time_history.t[np.nanargmax(time_history.T)]
        elif ignition_type== "d/dt max":
            tau = time_history.t[np.nanargmax(np.gradient(time_history.T, time_history.t))]
        else:
            raise ValueError(f"Warning: Unknown ignition_type {ignition_type}")
    elif ignition_target == "P" :
        "TODO: reuse species code"
        if ignition_type == "max":
            tau = time_history.t[np.nanargmax(time_history.P)]
        elif ignition_type == "d/dt max":
            tau = time_history.t[np.nanargmax(np.gradient(time_history.P, time_history.t))]
        else:         
            raise ValueError(f"Warning: Unknown ignition_type {ignition_type}")
    else:
        if ignition_type == "max":
            "TODO: looks good but test to make sure IDT calc good"
            #sometimes first and highest peak is not idt.
            # of all local maxima, that is at least 0.8*global_max, take the latest as idt
            species_conc_history = time_history.concentrations[:, species_idxs[ignition_target]]
            # print(time_history.t)
            # print(time_history.concentrations)
            # print(species_conc_history)
            # global_max = np.max(species_conc_history)
            # threshold = 0.8 * global_max
            # peaks, _ = find_peaks(species_conc_history, height=threshold, distance=3)
            # print(peaks)
            # latest_peak_index = peaks[-1] if peaks.size > 0 else None
            # print(latest_peak_index)
            # tau = time_history.t[latest_peak_index]
            # max_tau = time_history.t[np.nanargmax(species_conc_history)]
            
            # print(f"Computed max-based IDT for species {ignition_target}. Global max conc: {global_max}, Max-based tau: {max_tau}, Selected tau: {tau}")
            tau = time_history.t[np.nanargmax(time_history.concentrations[:, species_idxs[ignition_target]])]
            print(f"Computed max-based IDT for species {ignition_target}. Max-based tau: {tau}")
        elif ignition_type == "d/dt max":
            "TODO: looks good but test to make sure IDT calc good"
            # print(reactor.phase.species_names)
            # print(time_history.concentrations[:, species_idxs[ignition_target]])
            # print(time_history.t)
            tau = time_history.t[np.nanargmax(np.gradient(time_history.concentrations[:, species_idxs[ignition_target]], time_history.t))]
        elif ignition_type =="concentration":
            "TODO: target conc equivalent to sthereshold conc in next logic. combine"
            target_conc = ignition_amount
            tau = time_history.t[np.nanargmin(np.abs(float(target_conc) - time_history.concentrations[:, species_idxs[ignition_target]]))]
            print(f"Computed concentration-based IDT at {target_conc} for species {ignition_target}")
        elif ignition_type =="relative concentration":
            "TODO: looks good but test to make sure IDT calc good"
            # ignition when target species conc > ((target species conc at t=0) + ignition_amount * (max conc - target species conc at t=0))
            species_conc_history = time_history.concentrations[:, species_idxs[ignition_target]]
            baseline_conc = species_conc_history[0]
            max_conc = np.max(species_conc_history)
            threshold_conc = baseline_conc + (max_conc - baseline_conc) * ignition_amount
            idxs = np.where(species_conc_history >= threshold_conc)[0]
            if idxs.size == 0:
                tau = np.nan
                print(f"Warning: No ignition within integration time for {operating_condition}. Consider increasing t_max.")
                return tau
            idx = idxs[0]
            if idx == 0:
                tau = time_history.t[0]
                return tau
            # linear interpolation to get more accurate tau
            ratio = (species_conc_history[idx]- threshold_conc)/(species_conc_history[idx]-species_conc_history[idx-1])
            tau = time_history.t[idx] - ((time_history.t[idx]-time_history.t[idx-1])*ratio)
            noninterpolated_tau = time_history.t[idx]
            print(f"Computed relative concentration-based IDT at {ignition_amount*100:.1f}% increase for species {ignition_target}. Non-interpolated tau: {noninterpolated_tau:.3e} s, Interpolated tau: {tau:.3e} s")
        elif ignition_type =="baseline min intercept from d/dt":
            "TODO: ask what this is"
            raise NotImplementedError("baseline min intercept from d/dt not implemented yet")
        else:
            raise ValueError(f"Warning: Unknown ignition_type {ignition_type}")
        
        
    
    
    # tau = time_history.t[idx]
    
    #convert tau to microseconds
    tau = tau * 1e6
      
    if tau > t_max*1e6*0.8:
        print(f"Warning: Ignition delay time ({tau:.3e} μs) is close to the maximum simulation time ({t_max*1e6:.3e} μs). Consider increasing t_max.")
        
    t1 = time.time()
    print(f"T5 = {T5} K, P5 = {P5} pa, Computed Ignition Delay: {tau:.3e} μs. Took {t1-t0:3.2f}s to compute")
    
    if save_time_history_plot:
        save_time_history_plot_to_file(time_history, ignition_target, operating_condition, tau)
    
    return tau
    
def plot_IDT_vs_T(gas, operating_conditions_df, output_path=None, interactive = True, save_time_history_plots=False):
    T5_list = []
    P5_list = []
    exp_tau_list = []
    sim_tau_list = []
    for operating_condition in operating_conditions_df.itertuples(index=False):
        T5_list.append(operating_condition[0])
        P5_list.append(operating_condition[2])
        exp_tau_list.append(operating_condition[6])
        sim_tau = calc_IDT_constV(gas, operating_condition, t_max=2, save_time_history_plot=save_time_history_plots)
        sim_tau_list.append(sim_tau)
    
    if output_path is None:
        xml_name = operating_conditions_df.iloc[0, 12].split(".")[0]
        fuel_name = operating_conditions_df.iloc[0, 4].split(":")[0]
        results_path = f"results/idt/plots/{fuel_name}/{xml_name}"
    else:
        results_path = output_path
    
    os.makedirs(results_path, exist_ok=True)
    
    scaled_inverse_T5 = [1000.0 / T for T in T5_list]
    
    if interactive:
        fig = go.Figure()

        # Experimental data
        fig.add_trace(
            go.Scatter(
                x=scaled_inverse_T5,
                y=exp_tau_list,
                mode="markers",
                name="real experiment",
                marker=dict(color="red", size=8)
            )
        )

        # Cantera simulation data
        fig.add_trace(
            go.Scatter(
                x=scaled_inverse_T5,
                y=sim_tau_list,
                mode="markers",
                name="Cantera",
                marker=dict(color="blue", size=8)
            )
        )

        # Layout and axes
        fig.update_layout(
            title=f"IDT_type:{operating_conditions_df.iloc[0,9]}{operating_conditions_df.iloc[0,8]}, {operating_conditions_df.iloc[0,4]}",
            xaxis_title="1000/T5 (1/K)",
            yaxis_title="Ignition Delay Time (μs)",
            yaxis_type="log",
            template="plotly_white",
            width=800,
            height=600
        )

        # Gridlines (Plotly-style)
        fig.update_xaxes(showgrid=True)
        fig.update_yaxes(showgrid=True)

        # Save interactive HTML (recommended)
        fig.write_html(f"{results_path}/IDT_plot_experiment_vs_cantera.html")
    else:
        plt.figure(figsize=(8, 6))
        plt.scatter(scaled_inverse_T5, exp_tau_list, label='real experiment', color='red')
        plt.scatter(scaled_inverse_T5, sim_tau_list, label='Cantera', color='blue')
        plt.xlabel('1000/T5 (1/K)')
        plt.ylabel('Ignition Delay Time (μs)')
        plt.yscale('log')
        plt.legend()
        plt.title('Ignition Delay Times Comparison')
        plt.grid(True)
        plt.savefig(f"{results_path}/IDT_plot_experiment_vs_cantera.svg")