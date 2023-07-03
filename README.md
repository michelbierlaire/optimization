# biogeme_optimization
Various optimization algorithms used for teaching and research

The package contains the following modules:

## algebra
It contains functions dealing with linear algebra:

- A modified Cholesky factorization introduced by Schnabel and Eskow (1999)
- The calculation of a descent direction based on this factorization.

## bfgs
The functions in this module calculate 

- the BFGS update of the hessian approximation (see Eq. (13.12) in Bierlaire (2015)),
- the inverse BFGS update of the hessian approximation (see Eq. (13.13) in Bierlaire (2015)).

## bounds
This module mainly defines the class `Bounds`that manages the bound constraints.

## diagnostics
This module defines the diagnostic of some optimization subproblems
(dogleg, and comjugate gradient).

## exceptions
It defines the `OptimizationError` exception.

## format
It defines the class `FormattedColumns` that formats the information
reported at each iteration of an algorithm.

## function
It defines the abstract class `FunctionToMinimize` that encapsulate
the calculation of the objective function and its derivatives.

## hybrid_function
It defines the class `HybridFunction` that calculates the objective
function and its derivatives, where the second derivative can be
either the analytical hessian, or a BFGS approximation.

## linesearch
This module implements the line search algorithms (see Chapter 11 in Bierlaire, 2015).

## simple_bounds
This module implements the minimization algorithm under bound
constraints proposed by Conn et al. (1988).

## trust_region
This module implements the trust region algorithms (see Chapter 12 in
Bierlaire, 2015).

## References

- Bierlaire, M. (2015). [Optimization: Principles and
  Algortihms](https://transp-or.epfl.ch/books/optimization/html/OptimizationPrinciplesAlgorithms2018.pdf). EPFL
  Press.
- Conn, A. R., Gould, N. I. M and Toint, Ph. L. (1988) [Testing a
  Class of Methods for Solving Minimization Problems with Simple
  Bounds on the
  Variables](https://www.ams.org/journals/mcom/1988-50-182/S0025-5718-1988-0929544-3/S0025-5718-1988-0929544-3.pdf). Mathematics
  of Computation, 50(182), 399-430.
- Schnabel, R. B. and Eskow, E. (1999) [A Revised Modified Cholesky Factorization Algorithm](https://doi.org/10.1137/s105262349833266x). SIAM Journal on Optimization, 9(4), 1135-1148.
