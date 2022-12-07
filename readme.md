# BFGS for black-box optimization problems

This is a Python implementation of the Broyden–Fletcher–Goldfarb–Shanno (BFGS) algorithm.

Derivatives are estimated using the standard finite difference approximations.

The stepsize along the gradient direction is defined using Bisection algorithm for weak Wolfe conditions.

The STOP conditions are the maximum number of iterations or the accuracy $\| \nabla f \| <= \varepsilon$.

The 3rd returned value is the number of FEs spent by the algorithm.