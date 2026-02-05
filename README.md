Collection of codes to support reaction mechanism optimization
## Arrhenius parameter uncertainty calculation
Arrhenius parameters are collected from the Reaction Mechanisim Generator (RMG) database
Installation: Follow instructions on the RMG website
https://reactionmechanismgenerator.github.io/RMG-Py/users/rmg/installation/index.html

## Sensitivity analysis

For a given reaction mechansim (.yaml), reaction rate uncertainty factors, a set of operatingm conditions, and QOI (Ignition Delay Time). Ranks the impact of each reaction on the QOI for the operating conditions. Currently only implemented QOI is Ignition Delay Time.

## Surrogate generation

Builds a quadratic surrog