## Arrhenius parameters uncertainty calculation 

Adapted from FORTRAN code UQ_RRC based on FUMILI from Dr. Nadezda Slavinskaya



Works with reactions of type
- Arrhenius
- MultiArrhenius
- PDepArrenius
- ThirdBodyArrhenius


**Installation:** Follow the instructions on the RMG website:
https://reactionmechanismgenerator.github.io/RMG-Py/users/rmg/installation/index.html


## Known Issues

### `get_params_from_rmg.py`

- Don't run this every time. Run it once for all reactions in a given mechanism.
- Can't find sources for usually around 1/3 of reactions.
- Reaction/species name parsing and matching with RMG needs to be improved.
- Didn't have time to implement Troe/Lindemann for pressure-dependent equations. They are currently just skipped.
- The efficiencies array is necessary for third-body reactions. Didn't have time to implement combining efficiencies from multiple sources. It's unclear what makes sense physically — perhaps collecting all sources and averaging them?
- It's not yet known whether TUMKIN will be the final Arrhenius parameter data source. Ideally it would be NIST, but it's unclear how to access NIST data programmatically.
- Only around 1/30 RMG data entries contain temperature range information, which is problematic since temperature range is important and its absence degrades data quality.
- Add weights to the Arrhenius fit, weighted by uncertainty (e.g. 50%).











[ k(T) = A \cdot T^{\beta} \cdot \exp\left(-\frac{E_a}{RT}\right) ]

Where:

( k(T) ): Reaction rate coefficient at temperature ( T ).
( A ): Pre-exponential factor (frequency factor).
( \beta ): Temperature exponent.
( E_a ): Activation energy.
( R ): Universal gas constant (( 8.314 , \text{J/mol·K} )).
( T ): Absolute temperature in Kelvin.

Details from the Code
Functional Form:

In the ARITHM subroutine, the functional ( Y ) (logarithm of the reaction rate) is calculated as: [ Y = \ln(k(T)) = \ln(A) + \beta \cdot \ln(T) - \frac{E_a}{RT} ]
This is the logarithmic transformation of the Arrhenius equation, which is commonly used in least squares fitting to linearize the equation.
Derivatives:

The derivatives of ( Y ) with respect to the parameters ( A ), ( \beta ), and ( E_a ) are:
( \frac{\partial Y}{\partial A} = \frac{1}{A} )
( \frac{\partial Y}{\partial \beta} = \ln(T) )
( \frac{\partial Y}{\partial E_a} = -\frac{1}{RT} )
Implementation:

The FUMILI subroutine uses these derivatives to compute the gradient and Hessian matrix for the least squares optimization.