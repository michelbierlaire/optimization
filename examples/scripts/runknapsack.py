"""File runknapsack.py

:author: Michel Bierlaire
:date: Fri Apr 14 14:33:18 2023

Example of how to run the VNS multi-objective optimization algorithm
on the knapsack problem.  The two objectives are: minimizing weight
and maximizing utility.

"""

from biogeme_optimization.vns import ParetoClass, vns
from knapsack import Knapsack, Sack
from logger import logger

logger.info('Example runknapsack.py')

UTILITY = [80, 31, 48, 17, 27, 84, 34, 39, 46, 58, 23, 67]
WEIGHT = [84, 27, 47, 22, 21, 96, 42, 46, 54, 53, 32, 78]
COST = [80, 8, 80, 8, 80, 8, 80, 8, 80, 8, 80, 8]
CAPACITY = 300
size = len(UTILITY)
FILE_NAME = 'knapsack.pareto'
Sack.utility_data = UTILITY
Sack.weight_data = WEIGHT
Sack.cost_data = COST

# We create an empty sack as starting point.
empty_sack = Sack([0] * size)

the_pareto = ParetoClass(max_neighborhood=5, pareto_file=FILE_NAME)

the_knapsack = Knapsack(utility=UTILITY, weight=WEIGHT, capacity=CAPACITY)


the_pareto = vns(
    problem=the_knapsack,
    first_solutions=[empty_sack.get_element()],
    pareto=the_pareto,
    number_of_neighbors=5,
)

print(f'Number of pareto solutions: {len(the_pareto.pareto)}')
print(f'Number of considered solutions: {len(the_pareto.considered)}')
print(f'Pareto solutions: {the_pareto.pareto}')

for p in the_pareto.pareto:
    the_sack = Sack.from_string_representation(p.element_id)
    print(the_sack.describe())

the_pareto.plot()
