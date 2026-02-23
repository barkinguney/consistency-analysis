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
* UncertaintyQuantification.cpp doesnt work when all pressures in the dataset are the same. For that case need to fit just 2 parameters A and B. C need to be removed from model, jacobian, lsfit, and uncertainty propagation.  
* UncertaintyQuantification.cpp reads from the old json format. We need a more comprehensive xml input format based on RESPECTH, with the addition of FixedIFS data, and T5,P5,phi uncertainty data. 
 

## Arrhenius parameter uncertainty calculation
Arrhenius parameters are collected from the Reaction Mechanisim Generator (RMG) database
Installation: Follow instructions on the RMG website
https://reactionmechanismgenerator.github.io/RMG-Py/users/rmg/installation/index.html

## Sensitivity analysis

For a given reaction mechansim (.yaml), reaction rate uncertainty factors, a set of operatingm conditions, and QOI (Ignition Delay Time). Ranks the impact of each reaction on the QOI for the operating conditions. Currently only implemented QOI is Ignition Delay Time.

## Surrogate generation

Builds a quadratic surrog

