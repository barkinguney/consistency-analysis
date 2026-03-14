
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
from datetime import datetime

def apply_reference_plot_style() -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.labelsize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "axes.spines.top": True,
            "axes.spines.right": True,
            "axes.linewidth": 0.9,
            "grid.color": "#D9D9D9",
            "grid.linestyle": "-",
            "grid.linewidth": 0.6,
            "figure.dpi": 120,
            "savefig.dpi": 400,
            "savefig.bbox": "tight",
        }
    )

def multiply_rates(gas, m_sample, rxn_ids = None, method="cantera_built_in"):
    if rxn_ids is None:
        rxn_ids = range(len(gas.reactions()))

    assert m_sample.shape[0] == len(rxn_ids), f"Length of m_sample ({m_sample.shape[0]}) must match number of reactions being modified ({len(rxn_ids)})"
    
    if method == "cantera_built_in":
        for i, rxn_idx in enumerate(rxn_ids):
            #print(gas.multiplier(rxn_idx), "*************")
            gas.set_multiplier(value=m_sample[i], i_reaction=rxn_idx)
            #print(gas.multiplier(rxn_idx), "*************")
            
            
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

def modify_reaction(gas, target_equation: str, new_params):
    rxn_idx = gas.reaction_equations().index(target_equation)
    rxn = gas.reaction(rxn_idx)
    A_old = rxn.rate.pre_exponential_factor
    b_old = rxn.rate.temperature_exponent
    Ea_old = rxn.rate.activation_energy
    #print(f"Old A-factor: {A_old}", f"Old temperature exponent: {b_old}", f"Old activation energy: {Ea_old}")

    rxn.rate = ct.ArrheniusRate(new_params[0], new_params[1], new_params[2])
    gas.modify_reaction(rxn_idx, rxn)
    #print(f"New A-factor applied: {gas.reaction(rxn_idx).rate.pre_exponential_factor}, New temperature exponent applied: {gas.reaction(rxn_idx).rate.temperature_exponent}, New activation energy applied: {gas.reaction(rxn_idx).rate.activation_energy}")
    
def convert_units(P5=None, P5_unit=None, ignition_amount=None, ignition_units=None, tau=None, tau_units=None): 
        """convert pressure to Pa, ignition concentration to kmol/m3, tau to seconds"""
        if P5 is not None:
            if P5_unit.lower() in ['pa']:
                pass
            elif P5_unit.lower() in ['kpa']:
                P5 = P5 * 1000
            elif P5_unit.lower() in ['atm']:
                P5 = P5 * 101325
            elif P5_unit.lower() in ['bar']:
                P5 = P5 * 100000
            elif P5_unit.lower() in ["torr"]:
                P5 = P5 * 133.322
            else:
                raise ValueError(f"Unknown pressure unit: {P5_unit}")
            P5_unit = "Pa"

        if ignition_amount is not None:
            if ignition_units.lower().strip() in ["mol/cm3"]:
                ignition_amount = float(ignition_amount) * 1e-3
                ignition_units = "kmol/m3"
            elif ignition_units.lower().strip() in ["kmol/m3", "unitless", ""]:
                pass
            else:
                raise ValueError(f"Unknown ignition amount unit: {ignition_units}")
            
        
        if tau is not None:
            if tau_units.lower().strip() in ["s"]:
                tau = float(tau)
            elif tau_units.lower().strip() in ["ms"]:
                tau = float(tau) * 1e-3
            elif tau_units.lower().strip() in ["us", "μs"]:
                tau = float(tau) * 1e-6
            else:
                raise ValueError(f"Unknown tau unit: {tau_units}")
            tau_units = "s"
        
        return P5, P5_unit, ignition_amount,ignition_units, tau, tau_units
    


def calc_IDT_constV(gas, operating_condition, t_max = 0.1, save_time_history_plot = False, debug=False):
    
    # TODO:pressure profile affect idt at low temps manually compensate, lower volume with time. i think that means to set different constv for each time.
    # two peaks, or one mini peak might happen. take the first peak. multi fuels cause 2 peaks
    # for sensitivity pressur profile dont matter. 
    # pressure profile can be between 2% to 10% 
    # do implement all igniton types present in data.
    # this is IDT just shock tube. 
    # const v assumption dont hold for low temp, long time, due to boundary layer growth.
    # base,ine min intercept from d/dt means idt is the interept time of d(x)/dt with  baseline x
    """Calculate ignition delay time at given operating condition.
    batch reactor const volume"""
    
    def parse_ignition_target(ignition_target: str):
        ignition_targets = ignition_target.split(";", 1)
        target = ignition_targets[0]
        if len(ignition_targets) > 1 and ignition_targets[1] != "":
            print(f"Warning: Multiple ignition targets specified ({ignition_target}). Using the first one: {target}")
        if "EX" in target:
            target = target.replace("EX", "")
        if "*" in target:
            target = target.replace("*", "")
        return target
    
    def get_species_indices(gas):
        species_indices = {name: i for i, name in enumerate(gas.species_names)}
        return species_indices
    
    
    def save_time_history_plot_to_file(time_history, ignition_target, operating_condition, tau):
        xml_name = operating_condition[12].split(".")[0]
        fuel_name = operating_condition[4].split(":")[0]
        results_path = f"results/idt/plots/{fuel_name}/{xml_name}"
        os.makedirs(results_path, exist_ok=True)
        species_idxs = get_species_indices(reactor.phase)

        if ignition_target == "T":
            y_values = time_history.T
            y_label = "Temperature [K]"
        elif ignition_target == "P":
            y_values = time_history.P / 101325.0
            y_label = "Pressure [atm]"
        else:
            y_values = time_history.concentrations[:, species_idxs[ignition_target]]
            y_label = f"{ignition_target} Concentration [kmol/m3]"

        apply_reference_plot_style()
        fig, ax = plt.subplots(figsize=(7.2, 5.2))

        series_color = "#1f77b4"
        idt_color = "#d62728"
        ax.plot(time_history.t, y_values, color=series_color, linewidth=1.8, label=ignition_target)
        ax.axvline(
            x=tau,
            color=idt_color,
            linestyle="--",
            linewidth=1.2,
            label=f"IDT = {tau:.3e} s",
        )

        max_x = tau * 1.5 if np.isfinite(tau) and tau > 0 else float(time_history.t[-1])
        phi_value = getattr(operating_condition, "phi", None)
        pressure_atm = float(P5) / 101325.0
        if phi_value is not None and phi_value != "":
            title_text = f"{composition}, phi = {float(phi_value):.1f}, P ~ {pressure_atm:.1f} atm, T = {float(T5):.0f} K"
        else:
            title_text = f"{composition}, P ~ {pressure_atm:.1f} atm, T = {float(T5):.0f} K"

        ax.set_xlim(0, max_x)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel(y_label)
        ax.set_title(title_text, pad=10)

        legend = ax.legend(loc="best", frameon=True)
        legend.get_frame().set_alpha(0.92)
        legend.get_frame().set_edgecolor("#DDDDDD")

        ax.grid(True, which="major", axis="both")
        ax.minorticks_on()
        ax.grid(True, which="minor", linestyle=":", linewidth=0.45, alpha=0.6)
        ax.margins(x=0.02, y=0.08)
        fig.tight_layout()

        output_stem = f"{results_path}/time_history_{operating_condition[0]}{operating_condition[1]}_{operating_condition[2]}{operating_condition[3]}"
        fig.savefig(f"{output_stem}.png")
        fig.savefig(f"{output_stem}.svg")
        plt.close(fig)
        
    def calc_idt_from_type(time_history, ignition_target, ignition_type, simple=True):
        #problems
        #sometimes doubled for OHmax
        #1000x for concetrations
        #2-3x for relative concentrations
        def calc_baseline_min_intercept_tau(signal_history, t_history, target_name):
            signal_history = np.asarray(signal_history, dtype=float)
            t_history = np.asarray(t_history, dtype=float)

            grad = np.gradient(signal_history, t_history)
            idx_ddt_max = int(np.nanargmax(grad))
            slope = float(grad[idx_ddt_max])
            t_ddt_max = float(t_history[idx_ddt_max])
            y_ddt_max = float(signal_history[idx_ddt_max])
            baseline_min = float(np.nanmin(signal_history))

            if not np.isfinite(slope) or slope <= 0:
                print(
                    f"Warning: Non-positive or invalid max d/dt slope for {target_name}. "
                    "Using d/dt max time directly."
                )
                return t_ddt_max

            tau_intercept = t_ddt_max + (baseline_min - y_ddt_max) / slope
            tau_intercept = max(float(t_history[0]), min(t_ddt_max, float(tau_intercept)))
            return tau_intercept
        
        if simple:
            species_idxs = get_species_indices(reactor.phase)
            #tau = time_history.t[np.argmax(np.gradient(time_history.T, time_history.t))]
            if ignition_target == "T" :
                "TODO: reuse species code"
                if ignition_type == "max":
                    tau = time_history.t[np.nanargmax(time_history.T)]
                elif ignition_type== "d/dt max":
                    tau = time_history.t[np.nanargmax(np.gradient(time_history.T, time_history.t))]
                elif ignition_type == "baseline min intercept from d/dt":
                    tau = calc_baseline_min_intercept_tau(time_history.T, time_history.t, "T")
                else:
                    raise ValueError(f"Warning: Unknown ignition_type {ignition_type}")
            elif ignition_target == "P" :
                "TODO: reuse species code"
                if ignition_type == "max":
                    tau = time_history.t[np.nanargmax(time_history.P)]
                elif ignition_type == "d/dt max":
                    tau = time_history.t[np.nanargmax(np.gradient(time_history.P, time_history.t))]
                elif ignition_type == "baseline min intercept from d/dt":
                    tau = calc_baseline_min_intercept_tau(time_history.P, time_history.t, "P")
                else:         
                    raise ValueError(f"Warning: Unknown ignition_type {ignition_type}")
            # elif ignition targaet contains "OH"    
            elif "OH" in ignition_target:
                tau = time_history.t[np.argmax(np.gradient(time_history.T, time_history.t))]
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
                    #print(f"Computed max-based IDT for species {ignition_target}. Max-based tau: {tau}")
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
                    tau = tau * 1000 # x1000 here fixes it. it means somethig is wron with units
                    #print(f"Computed concentration-based IDT at {target_conc} for species {ignition_target}")
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
                    # # linear interpolation to get more accurate tau
                    # ratio = (species_conc_history[idx]- threshold_conc)/(species_conc_history[idx]-species_conc_history[idx-1])
                    # tau = time_history.t[idx] - ((time_history.t[idx]-time_history.t[idx-1])*ratio)
                    tau = time_history.t[idx]
                    #print(f"Computed relative concentration-based IDT at {ignition_amount*100:.1f}% increase for species {ignition_target}., tau: {tau:.3e} s")
                elif ignition_type =="baseline min intercept from d/dt":
                    species_conc_history = time_history.concentrations[:, species_idxs[ignition_target]]
                    tau = calc_baseline_min_intercept_tau(species_conc_history, time_history.t, ignition_target)
                else:
                    raise ValueError(f"Warning: Unknown ignition_type {ignition_type}")
        else:
            print(f"Calculating IDT using target: {ignition_target}, type: {ignition_type}")
    
            species_idxs = get_species_indices(reactor.phase)
            
            if ignition_target == "T" :
                "TODO: reuse species code"
                if ignition_type == "max":
                    tau = time_history.t[np.nanargmax(time_history.T)]
                elif ignition_type== "d/dt max":
                    tau = time_history.t[np.nanargmax(np.gradient(time_history.T, time_history.t))]
                elif ignition_type == "baseline min intercept from d/dt":
                    tau = calc_baseline_min_intercept_tau(time_history.T, time_history.t, "T")
                else:
                    raise ValueError(f"Warning: Unknown ignition_type {ignition_type}")
            elif ignition_target == "P" :
                "TODO: reuse species code"
                if ignition_type == "max":
                    tau = time_history.t[np.nanargmax(time_history.P)]
                elif ignition_type == "d/dt max":
                    tau = time_history.t[np.nanargmax(np.gradient(time_history.P, time_history.t))]
                elif ignition_type == "baseline min intercept from d/dt":
                    tau = calc_baseline_min_intercept_tau(time_history.P, time_history.t, "P")
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
                    species_conc_history = time_history.concentrations[:, species_idxs[ignition_target]]
                    tau = calc_baseline_min_intercept_tau(species_conc_history, time_history.t, ignition_target)
                else:
                    raise ValueError(f"Warning: Unknown ignition_type {ignition_type}")
        
        return tau
        
        
    T5 = operating_condition.T5  # Kelvin
    P5, P5_units, ignition_amount, ignition_units, exp_tau, tau_units = convert_units(operating_condition.P5, operating_condition.P5_units, operating_condition.ignition_amount, operating_condition.ignition_units, operating_condition.tau, operating_condition.tau_units)
    if ignition_amount != "":   
        ignition_amount = float(ignition_amount)
    composition = operating_condition.composition  # Cantera composition string
    ignition_target = parse_ignition_target(operating_condition.ignition_target)
    ignition_type = operating_condition.ignition_type

    gas.TPX = T5, P5, composition
    if operating_condition.pressure_rise is not None and operating_condition.pressure_rise != "" and False:
        # 2. Calculate specific heat ratio (gamma) at initial conditions
        gamma = gas.cp_mass / gas.cv_mass
        # 3. Convert your relative dP/dt from ms^-1 to s^-1
        dpdt_rel_ms = operating_condition.pressure_rise  # Assuming this is in ms^-1
        r_sec = dpdt_rel_ms * 1000.0  # Now 21.4 s^-1
        # 4. Create the Reactor an      d a dummy Environment (Reservoir)
        reactor = ct.IdealGasReactor(gas)
        env = ct.Reservoir(ct.Solution('air.yaml'))
        # 5. Connect them with a Wall
        wall = ct.Wall(reactor, env)
        wall.area = 1.0  # m^2 (so velocity mathematically equals dV/dt)
        # 6. Set the Wall Velocity to compress the reactor
        V0 = reactor.volume
        constant_velocity = -(V0 * r_sec) / gamma
        # In Cantera, setting the velocity to a float uses a constant speed.
        # The negative sign ensures the volume is decreasing (compression).
        wall.velocity = constant_velocity 
        # 7. Set up the Reactor Network and run as usual
        reactor_network = ct.ReactorNet([reactor])
    else:  
        # constant volume if volume not changed
        reactor = ct.IdealGasReactor(gas, energy='on', clone=True)  
        reactor_network = ct.ReactorNet([reactor])
    
    time_history = ct.SolutionArray(reactor.phase, extra="t")

    t0 = time.time()
    t = 0

    counter = 1
    while t < t_max:
        t = reactor_network.step()
        if not counter % 2:
            time_history.append(reactor.phase.state, t=t)
        counter += 1
        

    tau = calc_idt_from_type(time_history=time_history, ignition_target=ignition_target, ignition_type=ignition_type, simple=True) 
          
    if tau > t_max:
        #raise ValueError(f"Warning: Ignition delay time ({tau:.3e} s) exceeds maximum simulation time ({t_max:.3e} s). Consider increasing t_max.")
        #print(f"Warning: Ignition delay time ({tau*1e6:.3e} μs) is close to the maximum simulation time ({t_max*1e6:.3e} μs). Consider increasing t_max.")
        print(f"Warning: Ignition delay time ({tau:.3e} s) is close to the maximum simulation time ({t_max:.3e} s). Consider increasing t_max.")
        
    t1 = time.time()
    if debug:
        print(f"T5 = {T5} K, P5 = {P5:.3e} pa, Computed Ignition Delay: {tau*1e6:.3e} μs. Took {t1-t0:3.2f}s to compute")
    
    if save_time_history_plot:
        save_time_history_plot_to_file(time_history, ignition_target, operating_condition, tau)
    
    return tau
    
    

def plot_IDT_vs_T(gas, operating_conditions_df, output_path=None, interactive = False, save_time_history_plots=False):
    
    
    T5_list = []
    P5_list = []
    exp_tau_list = []
    sim_tau_list = []
    for operating_condition in operating_conditions_df.itertuples(index=False):
        T5_list.append(operating_condition[0])
        P5_list.append(operating_condition[2])
        _, _, _, _, exp_tau, _ = convert_units(tau=operating_condition[6], tau_units=operating_condition[7])
        exp_tau_list.append(exp_tau)
        sim_tau = calc_IDT_constV(gas, operating_condition, t_max=2, save_time_history_plot=save_time_history_plots)
        sim_tau_list.append(sim_tau)
        
    sim_tau_list = [tau*1e6 for tau in sim_tau_list]  # convert to microseconds
    exp_tau_list = [tau*1e6 for tau in exp_tau_list]  # convert to microseconds
    
    
    if output_path is None:
        xml_name = operating_conditions_df.iloc[0, 12].split(".")[0]
        fuel_name = operating_conditions_df.iloc[0, 4].split(":")[0]
        results_path = f"results/idt/plots/{fuel_name}/{xml_name}"
    else:
        results_path = output_path
    
    os.makedirs(results_path, exist_ok=True)
    
    scaled_inverse_T5 = [1000.0 / T for T in T5_list]
    
    first_row = operating_conditions_df.iloc[0]
    ignition_type = first_row["ignition_type"] if "ignition_type" in operating_conditions_df.columns else first_row.iloc[8]
    ignition_target = first_row["ignition_target"] if "ignition_target" in operating_conditions_df.columns else first_row.iloc[9]
    composition = first_row["composition"] if "composition" in operating_conditions_df.columns else first_row.iloc[4]
    phi = first_row["phi"] if "phi" in operating_conditions_df.columns else None
    pressure = operating_conditions_df["P5"].mean()
    author_year = first_row["author_year"] if "author_year" in operating_conditions_df.columns else None
    # convert pascal to atm 
    pressure_atm = pressure / 101325
    plot_title = f"{composition}, φ = {phi:.1f}, P ~ {pressure_atm:.1f} atm"
    experiment_label = author_year
    
    if interactive:
        fig = go.Figure()

        # Experimental data
        fig.add_trace(
            go.Scatter(
                x=scaled_inverse_T5,
                y=exp_tau_list,
                mode="markers",
                name=experiment_label,
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
            title=plot_title,
            xaxis_title="1000/T5 [1000/K]",
            yaxis_title="Ignition Delay [μs]",
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
        apply_reference_plot_style()
        fig, ax = plt.subplots(figsize=(7.2, 5.2))

        exp_color = "#d62728"
        sim_color = "#1f77b4"

        ax.scatter(
            scaled_inverse_T5,
            exp_tau_list,
            label=experiment_label,
            color=exp_color,
            marker="o",
            s=34,
            edgecolors=exp_color,
            linewidths=0.9,
            alpha=0.95,
        )
        ax.scatter(
            scaled_inverse_T5,
            sim_tau_list,
            label="Cantera",
            color=sim_color,
            marker="s",
            s=34,
            edgecolors=sim_color,
            linewidths=0.9,
            alpha=0.95,
        )

        ax.set_xlabel("1000/T5 [1000/K]")
        ax.set_ylabel("Ignition Delay [μs]")
        ax.set_yscale("log")
        ax.set_title(plot_title, pad=10)

        legend = ax.legend(loc="best", frameon=True)
        legend.get_frame().set_alpha(0.92)
        legend.get_frame().set_edgecolor("#DDDDDD")

        ax.grid(True, which="major", axis="both")
        ax.minorticks_on()
        ax.grid(True, which="minor", linestyle=":", linewidth=0.45, alpha=0.6)
        ax.margins(x=0.03, y=0.08)
        fig.tight_layout()

        fig.savefig(f"{results_path}/IDT_plot_experiment_vs_cantera.png")
        fig.savefig(f"{results_path}/IDT_plot_experiment_vs_cantera.svg")
        plt.close(fig)