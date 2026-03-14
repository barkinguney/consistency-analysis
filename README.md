# Data Consistency Evaluation for Reaction Model Optimization

Collection of uncertainty evaluation, consistency analysis and similation tools to support reaction mechanism optimization.

## Table of Contents

- [Modules](#modules)
  - [Experimental Shock Tube Ignition Delay Time Uncertainty Evaluation](#experimental-shock-tube-ignition-delay-time-uncertainty-evaluation)
  - [Reaction Rate Coefficient Uncertainty Evaluation](#reaction-rate-coefficient-uncertainty-evaluation)
  - [Cantera Ignition Delay Time Simulation](#cantera-ignition-delay-time-simulation)
  - [B2BDC Consistency Analysis](#b2bdc-consistency-analysis)
- [Installation](#installation)
- [Known Issues](#known-issues)


## Modules

### Experimental Shock Tube Ignition Delay Time Uncertainty Evaluation

Performs uncertainty evaluation of experimental shock tube Ignition Delay Time (IDT) data. Total uncertainty is combination of systematic and random parts. Systematic uncertainties are evaluated by assessing nonidealities in experimental facility conditions. Random uncertainties are evaluated by a linear regression model for the given dataset. For detailed description, see:
[`IDT_uncertainty_cpp/IDT_UQ.pdf`](IDT_uncertainty_cpp/IDT_UQ.pdf)
[`IDT_uncertainty_cpp/README.md`](IDT_uncertainty_cpp/README.md)

### Reaction Rate Coefficient Uncertainty Evaluation

For a given reaction, collects rate expressions from the Reaction Mechanism Generator (RMG) Database, from all available literature sources. Performs linear regression to evalute the uncertainty interval of the reaction rate coeffient k(t). For detailed description, see:
[`RRC_uncertainty/README.md`](RRC_uncertainty/README.md)




### Cantera Ignition Delay Time Simulation

Simulates constant-volume ignition in shock tubes for a given reaction mechanism and operating conditions, computing IDT via configurable targets (e.g. dT/dt maximum, species thresholds). For detailed description, see:[`sensitivity/README.md`](sensitivity/README.md)




### B2BDC Consistency Analysis

Performs Model-Data consistency analysis using the Bound-to-Bound Data Collaboration (B2BDC) method. For a dataset of experimental points with associated uncertainties, and a model with uncertainty intervals on model parameters, B2BDC tries to find a set of model parameters within their uncertainty intervals that explain all data points within their uncertianty intervals. MATLAB Toolbox provided at 
[B2BDC](https://github.com/B2BDC/B2BDC).
 For detailed description, see:[`B2BDC/README.md`](B2BDC/README.md)


## Installation

See respective module README.

## Known Issues

- There needs to be something handling units, or all input/output values need to be accompanied by their units.
- Needs unit tests
