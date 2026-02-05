## Arrhenius parameters uncertainty calculation 

Adapted from FORTRAN code UQ_RRC based on FUMILI from Dr. Nadezda Slavinskaya



Works with reactions of type
- Arrhenius
- MultiArrhenius
- PDepArrenius
- ThirdBodyArrhenius











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