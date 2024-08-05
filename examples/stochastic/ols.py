"""File ols.py

:author: Nicola Ortelli, Michel Bierlaire
:date: Fri Aug 11 10:36:15 2023

Least square example
"""

import logging
from abc import ABC, abstractmethod
from typing import final, NamedTuple
import pandas as pd
import numpy as np
from biogeme_optimization.stochastic_function import StochasticFunction


class FunctionData(NamedTuple):
    function: float
    gradient: np.ndarray
    hessian: np.ndarray
    relative_batch_size: float


logger = logging.getLogger(__name__)

R = 4
MAX_WEIGHT = 10


class Ols(StochasticFunction):
    """This is an abstract class. The actual function to minimize
    must be implemented in a concrete class deriving from this one.

    """

    def __init__(self):
        super().__init__()
        self.batch_size = 0.01
        self.df_orig = pd.read_csv('toy_data.csv')
        self.df = self.df_orig.copy(deep=True)
        self.a = np.random.normal(0, 1, (self.dimension, R))
        self.w = 1.0

    def dimension(self):
        """Provides the number of unsorted_set_of_variables of the problem"""
        return 10

    def change_relative_batch_size(self, desired_factor):
        """Increase the size of the batch

        :param factor: multiplicative factor for the batch size. If
            larger than 1, the size is increased, if smaller than 1,
            it is decreased. Must be positive.
        :type factor: float

        """
        self.w /= desired_factor

    def first_sample(self):
        """Instruct to use the first sample for calculation"""

    def use_full_sample(self):
        """Instruct to use the full sample for calculation"""
        self.batch_size = 1.0

    def _f(self):
        """Calculate the canonical_value of the function

        :return: canonical_value of the function
        :rtype: float
        """
        expla = np.asarray(self.df.drop(columns=['target', 'weight']))
        target = np.asarray(self.df.target)
        weight = np.asarray(self.df.weight)
        f = (
            weight[:, np.newaxis]
            * (expla.dot(self.x[:, np.newaxis]) - target[:, np.newaxis]) ** 2
        ).sum()
        result = FunctionData(
            function=f, gradient=None, hessian=None, relative_batch_size=1
        )
        return result

    def _f_g(self):
        """Calculate the canonical_value of the function and the gradient

        :return: canonical_value of the function and the gradient
        :rtype: FunctionData
        """
        expla = np.asarray(self.df.drop(columns=['target', 'weight']))
        target = np.asarray(self.df.target)
        weight = np.asarray(self.df.weight)
        result = FunctionData(
            function=(
                weight[:, np.newaxis]
                * (expla.dot(self.x[:, np.newaxis]) - target[:, np.newaxis]) ** 2
            ).sum(),
            gradient=(
                weight[:, np.newaxis]
                * 2
                * expla
                * (expla.dot(self.x[:, np.newaxis]) - target[:, np.newaxis])
            ).sum(axis=0),
            hessian=None,
            relative_batch_size=1.0,
        )
        return result

    def _f_g_h(self):
        """Calculate the canonical_value of the function, the gradient and the Hessian

        :return: canonical_value of the function, the gradient and the Hessian
        :rtype: FunctionData
        """
        return None

    def lsh(self, w, a):
        # normalizing only the explanatory unsorted_set_of_variables
        df_expla = self.df.drop(columns=['target', 'weight'])

        # Should be done once for all
        df_expla_norm = (df_expla - df_expla.min()) / (df_expla.max() - df_expla.min())

        # hashing into buckets according to LSH
        b = np.random.rand(a.shape[1]) * w
        buckets = np.floor((df_expla_norm.dot(a) + b) / w).astype(int)

        # adding back the buckets to the original df
        df = self.df.merge(
            buckets, how='right', left_index=True, right_index=True
        ).reset_index()

        # shuffle df
        df = df.sample(frac=1)

        # adding a column that guarantees max_weight is never exceeded
        if MAX_WEIGHT:
            group_counts = df.groupby([*buckets.columns], sort=False)[
                'index'
            ].transform('cumcount')
            df['sieve'] = (group_counts / MAX_WEIGHT).astype(int)

        # preparing aggregation dictionary for final grouping
        agg_dict = dict()
        agg_dict['index'] = 'first'
        agg_dict['weight'] = 'sum'

        # final grouping, based on target, buckets and sieve
        return (
            df.groupby([*buckets.columns, 'sieve'], sort=False)
            .agg(agg_dict)
            .set_index('index')
            .rename_axis(index=None)
        )
