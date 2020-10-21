#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: dwalkerpage
"""

import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

class GenerateData:

    def generate_multivariable_linear_data(self,
                                           data_size,
                                           Xmaxs,
                                           slopes,
                                           intercept):
        '''
        Generate pseudo-random linear data for testing

        Parameters
        ----------
        data_size : int specifying number of data points
        Xmaxs : list containing maximum numerical values for each independent
                variable
        slopes : list of numbers specifying slopes to generate linear data
        intercept : int or float specifying intercept to generate linear data

        Returns
        -------
        xvalues : numpy array containing generated xvalues
        yvalues : numpy array containing generated yvalues

        '''
        Xmaxs = np.array(Xmaxs)
        slopes = np.array(slopes)
        # Choose data_size number of random values from interval [0, 1)
        # multiplied by xmax value for each independent variable
        xvalues = np.array([xmax * np.random.rand(data_size)
                            for xmax in Xmaxs]).T
        # Generate random yvalues as a linear function of the xvalues
        yvalues = ((slopes * xvalues).sum(axis=1)
                  + np.random.randn(data_size)
                  + intercept
                  )
        return xvalues, yvalues

    def feat_norm(self, xvalues):
        '''
        Normalize independent variables to range within (0, 1)

        Parameters
        ----------
        xvalues : array containing each independent variable

        Returns
        -------
        normalized_xvalues : array containing each independent variable
                             normalized

        '''
        xvalues = np.array(xvalues)
        normalized_xvalues = ((xvalues - xvalues.min(axis=0))
                              / (xvalues.max(axis=0) - xvalues.min(axis=0)
                                 )
                              )
        return normalized_xvalues

    def feat_stand(self, xvalues):
        '''
        Standardize independent variables to have mean=0 and std=1

        Parameters
        ----------
        xvalues : array containing each independent variable

        Returns
        -------
        normalized_xvalues : array containing each independent variable
                             standardized

        '''
        xvalues = np.array(xvalues)
        standardized_xvalues = ((xvalues - xvalues.mean(axis=0))
                               / xvalues.std(axis=0))
        return standardized_xvalues

if __name__ == '__main__':

    g = GenerateData()

    # Test with auto-generated data, two independent variables
    Xs, y = g.generate_multivariable_linear_data(data_size=250,
                                                 Xmaxs=[50, 150],
                                                 slopes=[1, 2],
                                                 intercept=1)
    norm_Xs = g.feat_norm(Xs)
    X_train, X_test, y_train, y_test = train_test_split(norm_Xs,
                                                        y,
                                                        test_size=0.25,
                                                        random_state=7)
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r_squared = r2_score(y_test, y_pred)
    print(f'MAE: {mae}, MSE: {mse}, R^2: {r_squared}')

    # Test with auto-generated data, three independent variables
    Xs, y = g.generate_multivariable_linear_data(data_size=250,
                                                 Xmaxs=[50, 150, 250],
                                                 slopes=[1, 2, 3],
                                                 intercept=1)
    norm_Xs = g.feat_norm(Xs)
    X_train, X_test, y_train, y_test = train_test_split(norm_Xs,
                                                        y,
                                                        test_size=0.25,
                                                        random_state=7)
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r_squared = r2_score(y_test, y_pred)
    print(f'MAE: {mae}, MSE: {mse}, R^2: {r_squared}')

    # Test with auto-generated data, five independent variables
    Xs, y = g.generate_multivariable_linear_data(data_size=250,
                                                 Xmaxs=[50, 150, 250, 350, 450],
                                                 slopes=[1, 2, 3, 4, 5],
                                                 intercept=1)
    norm_Xs = g.feat_norm(Xs)
    X_train, X_test, y_train, y_test = train_test_split(norm_Xs,
                                                        y,
                                                        test_size=0.25,
                                                        random_state=7)
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r_squared = r2_score(y_test, y_pred)
    print(f'MAE: {mae}, MSE: {mse}, R^2: {r_squared}')

    # Test with "real" data
    path = '/Users/dwalkerpage/Documents/Data_Science/Projects/Linear_Regres'\
           'sion/Multivariate_Linear_Regression_From_Scratch/Data/ex1data2.txt'
    data = pd.read_csv(path, names=['Size', 'Bedrooms', 'Price'])
    Xs = np.array(data[['Size', 'Bedrooms']])
    y = np.array(data['Price'])
    norm_Xs = g.feat_norm(Xs)
    X_train, X_test, y_train, y_test = train_test_split(norm_Xs,
                                                        y,
                                                        test_size=0.25,
                                                        random_state=7)
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r_squared = r2_score(y_test, y_pred)
    print(f'MAE: {mae}, MSE: {mse}, R^2: {r_squared}')