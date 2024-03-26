import unittest
import numpy as np
import matplotlib.pyplot as plt
from biogeme_optimization.teaching.plot_function import plot_function


class TestPlotFunction(unittest.TestCase):

    def test_plot_function_runs_without_errors(self):
        # This is a dummy function for the test
        def test_func(x):
            return x**2

        # Attempt to run the function, mainly to see if any errors are thrown
        ax = plot_function(
            my_function=test_func,
            label='y = x^2',
            x_label='x',
            y_label='y',
            x_min=0,
            x_max=10,
            y_min=0,
            y_max=100,
        )

        # If the function has run and returned an Axes object, check few properties
        self.assertIsInstance(
            ax, plt.Axes, "The function should return an Axes object."
        )
        self.assertEqual(
            ax.get_title(), 'y = x^2', "The title of the plot is incorrect."
        )
        self.assertEqual(ax.get_xlabel(), 'x', "The x-axis label is incorrect.")
        self.assertEqual(ax.get_ylabel(), 'y', "The y-axis label is incorrect.")
        self.assertEqual(ax.get_ylim(), (0, 100), "The y-axis limits are incorrect.")

        # Optionally, check the lines drawn in the plot (assuming it's a line plot)
        lines = ax.get_lines()
        self.assertEqual(len(lines), 1, "There should be exactly one line in the plot.")


if __name__ == '__main__':
    unittest.main()
