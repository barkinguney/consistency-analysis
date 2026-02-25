Collection of codes to support reaction mechanism optimization

# Installation

- You need to have MATLAB installed. And a working MATLAB licence whenever executing MATLAB code from python
- python -m pip install matlabengine          requires you to have latest matlab version installed if not you can specify your matlab version like  matlabengine-25.2.2
- Following MATALAB toolboxes need to be installed manually inside the correct matlab version:
Statistics and Machine Learning Toolbox
Optimization Toolbox
https://github.com/B2BDC/B2BDC_v1.0
https://cvxr.com/cvx/download
https://github.com/sqlp/sedumi


# Issues
* UncertaintyQuantification.cpp 
    * doesnt work when all pressures in the dataset are the same. For that case need to fit just 2 parameters A and B. C need to be removed from model, jacobian, lsfit, and uncertainty propagation.
    * regularizing with Tref, and Pref do make the ls fit better, but make the uncertainties kinda meaningless. since in the propagating A,B,C,T5,P5,phi errors to tau error step, T5,P5 that is closer to Tref,Pref give smaller uncertainty results. it doesnt make sense to have smaller unceertanty just by being close to an arbitrary reference point
    * reads from the old json format. We need a more comprehensive xml input format based on RESPECTH, with the addition of FixedIFS data, and T5,P5,phi uncertainty data. 
* impact_factors_for_all.py 
    * likely is not too good at doing good sensitivity analysis. its better than nothing, but needs to be tested, fail cases need to be better understood. maybe replaced by a smarter method/better implementation?
* get_params_from_rmg.py 
    * cant find sources for usually around 1/3 of reactions. 
    * reaction/species name parsing and matching with rmg need to improved. 
    * I didnt have time to implement troe/lindemann for pressure dependent equations. they are currenty just skipped. 
    * efficiencies array is necessayry for third body reactions. I didnt have time to implement combining efficiencies from multiple sources. I dont know what makes sense phsically, put them all together and average multiple sources maybe? 
    * we dont know yet if TUMKIN arrhenius parameter data source will be rmg in the end. Ideally it would be NIST, but i dont know how to access NIST data. 
    * Only around 1/30 RMG data contain temperature range information. Which is bad. Temp range is important which makes the data worse.
* cantera_functions.py
    * for "bad" operationg conditions (we get bad conditions from random sampling) idt doesnt happen, so they are skipped. but sometimes that results in skipping too many points which likely lowers accuracy of stuff. needs to be investiged handled better possibly. mybe reject bad operating conditions in the very beginning. 
    * I didnt have time to implement pressure rise correction. It is relevant for multi species/mixed fuels. wouldn't effect sensitivity too much, maybe rankings. but still should be implemented
    * I implemented idt for differnt targets and types but a simple dT/dt max seem to work better. that means likely something with my implementation is wrong. Needs to be thoroughly tested.
    * currently any multiplicative factor is on the cantera net_rate. It may be useful and more powerful to modify A,n,Ea directly but someone would need to do that. 
* there needs to be somethign handling units, or only input/output values with units.
* everything needs to be tested. it is likely i have bugs everywhere. 


## Arrhenius parameter uncertainty calculation
Arrhenius parameters are collected from the Reaction Mechanisim Generator (RMG) database
Installation: Follow instructions on the RMG website
https://reactionmechanismgenerator.github.io/RMG-Py/users/rmg/installation/index.html

## Sensitivity analysis

For a given reaction mechansim (.yaml), reaction rate uncertainty factors, a set of operatingm conditions, and QOI (Ignition Delay Time). Ranks the impact of each reaction on the QOI for the operating conditions. Currently only implemented QOI is Ignition Delay Time.

## Surrogate generation

Builds a quadratic surrog

