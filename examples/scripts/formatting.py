"""File formatting.py

:author: Michel Bierlaire
:date: Mon Jul  3 14:36:35 2023

Example of formatted output
"""
import random
import string
from biogeme_optimization.format import Column, FormattedColumns


def random_string(max_length=5):
    length = random.randint(1, max_length)
    letters = string.ascii_letters
    return ''.join(random.choice(letters) for _ in range(length))


columns = (
    Column(title='Iter.', width=5),
    Column(title='Gradient', width=10),
    Column(title='Status', width=6),
)

the_formatter = FormattedColumns(columns)

print(the_formatter.formatted_title())

gradient = 1.0
for k in range(10):
    the_string = random_string()
    the_row = [k, gradient, the_string]
    print(the_formatter.formatted_row(the_row))
    gradient = gradient / 5.0
