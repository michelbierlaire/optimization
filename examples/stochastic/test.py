import numpy as np
from ols import Ols

the_function = Ols()

x = np.array(
    [
        1.49482047,
        -0.64109603,
        -1.2935423,
        -0.51923531,
        0.54525744,
        1.25769609,
        1.27906769,
        -2.07700305,
        0.05927564,
        1.0392733,
    ]
)

the_function.set_variables(x)

f = the_function.f()

print(x)

print(f)
