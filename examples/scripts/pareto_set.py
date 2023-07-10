"""File pareto_set.py

:author: Michel Bierlaire
:date: Wed Jul  5 09:42:07 2023

Example of the use of the Pareto class
"""

import matplotlib.pyplot as plt
from biogeme_optimization.pareto import (
    SetElement,
    Pareto
)

""" We consider an example with two objectives that are supposed to be minimized.
"""
pareto_set = Pareto()

#The first element is introduced as invalid. It will not be stored in the Pareto set
pareto_set.add_invalid(SetElement('1', [1.0, 2.0]))
# The next element is valid
pareto_set.add(SetElement('3', [5.0, 6.0]))
#The next element is valid, and dominates the previous. It will
#therefore replace it in the Pareto set
pareto_set.add(SetElement('2', [3.0, 4.0]))

# The set "pareto" contains only element 2, which is not dominated by any other.
# All the valid elements are stored in the set "considered".
# All the invalid elements are stored in the set "invalid".
# The elements that used to be in the Pareto set, but have been
# removed (here, element 3) are stored in the set "removed".
print(pareto_set)

pareto_set.filename='example_pareto.txt'
pareto_set.dump(['Example of a file for the pareto set', 'It is written in TOML format'])


ax = pareto_set.plot()
plt.show()
