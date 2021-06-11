#!/usr/bin/env python
# -*- coding: utf-8 -*-
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
import pandas as pd
from pandas import DataFrame
import numpy as np


def IQR_Outliers(X, features):
    print('# of features: ', len(features))
    print('Features: ', features)

    indices = [x for x in X.index]
    # print(indices)
    print('Number of samples: ', len(indices))

    out_indexlist = []

    for col in features:
        # Using nanpercentile instead of percentile because of nan values
        Q1 = np.nanpercentile(X[col], 25.)
        Q3 = np.nanpercentile(X[col], 75.)

        cut_off = (Q3 - Q1) * 10
        upper, lower = Q3 + cut_off, Q1 - cut_off
        print('\nFeature: ', col)
        print('Upper and Lower limits: ', upper, lower)

        outliers_index = X[col][(X[col] < lower) | (X[col] > upper)].index.tolist()
        outliers = X[col][(X[col] < lower) | (X[col] > upper)].values
        print('Number of outliers: ', len(outliers))
        print('Outliers Index: ', outliers_index)
        print('Outliers: ', outliers)

        out_indexlist.extend(outliers_index)

    # using set to remove duplicates
    out_indexlist = list(set(out_indexlist))
    out_indexlist.sort()
    print('\nNumber of rows with outliers: ', len(out_indexlist))
    print('List of rows with outliers: ', out_indexlist)
    return out_indexlist


class Custom_Encoder(BaseEstimator, TransformerMixin):
    def __init__(self, binary_columns, ordinal_columns, nominal_columns, loan_grade_map, cb_person_default_on_file_map):
        self.binary_columns = binary_columns
        self.ordinal_columns = ordinal_columns
        self.nominal_columns = nominal_columns
        self.loan_grade_map = loan_grade_map
        self.cb_person_default_on_file_map = cb_person_default_on_file_map

    def fit(self, X=None, y=None):
        return self

    def transform(self, X, y=None):
        if not isinstance(X, DataFrame):
            raise TypeError(f"Input type to SumByPrefixes transformer should be pandas DataFrame, but is {type(X)}")

        oneHotdf = pd.get_dummies(X[self.nominal_columns])
        X = pd.concat([X, oneHotdf], axis=1)
        X = X.drop(self.nominal_columns, axis=1)

        X['loan_grade'] = X.loan_grade.map(self.loan_grade_map)
        X['cb_person_default_on_file'] = X.cb_person_default_on_file.map(self.cb_person_default_on_file_map)

        return X


class SumByPrefixes(BaseEstimator, TransformerMixin):
    """
    Allows to sum values of selected features columns that share the same prefix
    and create new column <prefix>_agg. Can keep or drop the columns use to sum up the values.
    """

    def __init__(self, prefixes, drop=False):
        """
        Args:
            prefixes: list of prefixes to group
            drop: remove the columns that share the
        """
        self.prefixes = prefixes
        self.drop = drop
        self.column_names_ = None

    def fit(self, X=None, y=None):
        return self

    def transform(self, X, y=None):
        if not isinstance(X, DataFrame):
            raise TypeError(f"Input type to SumByPrefixes transformer should be pandas DataFrame, but is {type(X)}")

        for group_prefix in self.prefixes:
            group_features = [
                feature
                for feature in X.columns
                if feature.startswith(group_prefix) and feature != f"{group_prefix}_agg"
            ]

            # if matches at least one column
            if len(group_features) > 0:
                # sum up the values
                X.loc[:, f"{group_prefix}_agg"] = X[group_features].sum(axis=1)

                # drop source columns
                if self.drop:
                    X = X.drop(columns=group_features, axis=1)
            else:
                # if no columns
                print(f"The prefix `{group_prefix}` does not much any column")

        # store final column names
        self.column_names_ = X.columns.to_list()
        return X


class SimplePandasInputer(SimpleImputer):

    def __init__(
        self, *, missing_values=np.nan, strategy="mean", fill_value=None, verbose=0, copy=True, add_indicator=False
    ) -> None:
        super().__init__(
            missing_values=missing_values,
            strategy=strategy,
            fill_value=fill_value,
            verbose=verbose,
            copy=copy,
            add_indicator=add_indicator,
        )
        self.column_names_ = None

    def fit(self, X, y=None):
        if isinstance(X, DataFrame):
            self.column_names_ = X.columns.to_list()
        return super().fit(X=X, y=y)

    def transform(self, X):
        if X.shape[1] != len(self.column_names_):
            raise TypeError(f"Input shape does not match column names size")

        transformed = super(SimplePandasInputer, self).transform(X)

        return DataFrame(transformed, columns=self.column_names_)
