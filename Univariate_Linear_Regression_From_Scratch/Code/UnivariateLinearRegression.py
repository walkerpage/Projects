#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  1 10:38:29 2020

@author: dwalkerpage
"""

import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

%matplotlib inline


class UnivariateLinearRegression:

    def generate_linear_data(self, data_size, xmax, slope, intercept):
        '''
        Generate pseudo-random linear data for testing

        Parameters
        ----------
        data_size : int specifying number of data points
        xmax : int or float specifying max value for xvalues
        slope : int or float specifying slope to generate linear data
        intercept : int or float specifying intercept to generate linear data

        Returns
        -------
        xvalues : numpy array containing generated xvalues
        yvalues : numpy array containing generated yvalues

        '''
        # Choose data_size number of random values from interval [0, 1)
        # multiplied by xmax value
        xvalues = xmax * np.random.rand(data_size)
        # Generate random yvalues as a linear function of the xvalues
        yvalues = slope * xvalues + np.random.randn(data_size) + intercept
        return xvalues, yvalues

    def mae_cost(self, xvalues, yvalues, m, b):
        '''
        Compute mean absolute error

        Parameters
        ----------
        xvalues : iterable containing ints or floats
        yvalues : iterable containing ints or floats
        m : int or float representation of slope parameter
        b : int or float representation of intercept parameter

        Returns
        -------
        mae : float representation of mean absolute error

        '''
        predicted_values = (m * np.array(xvalues)) + b
        abs_errors = np.absolute(yvalues - predicted_values)
        errors_sum = np.sum(abs_errors)
        mae = errors_sum / len(xvalues)
        return mae

    def mse_cost(self, xvalues, yvalues, m, b):
        '''
        Compute mean squared error

        Parameters
        ----------
        xvalues : iterable containing ints or floats
        yvalues : iterable containing ints or floats
        m : int or float representation of slope parameter
        b : int or float representation of intercept parameter

        Returns
        -------
        mse : float representation of mean squared error

        '''
        predicted_values = (m * np.array(xvalues)) + b
        sq_errors = np.power((yvalues - predicted_values), 2)
        rss = np.sum(sq_errors)
        mse = rss / len(xvalues)
        return mse

    def rmse_cost(self, xvalues, yvalues, m, b):
        '''
        Compute root mean squared error

        Parameters
        ----------
        xvalues : iterable containing ints or floats
        yvalues : iterable containing ints or floats
        m : int or float representation of slope parameter
        b : int or float representation of intercept parameter

        Returns
        -------
        rmse : float representation of root mean squared error

        '''
        mse = self.mse_cost(xvalues, yvalues, m, b)
        rmse = math.sqrt(mse)
        return rmse

    def gradient_descent(self, xvalues, yvalues, m=0, b=0, alpha=0.01,
                         iterations=1000):
        '''
        Execute basic gradient descent

        Parameters
        ----------
        xvalues : iterable containing ints or floats
        yvalues : iterable containing ints or floats
        m : int or float representation of initial slope parameter.
            Default is 0.
        b : int or float representation of initial intercept parameter.
            Default is 0.
        alpha : int or float representation of initial learning rate for GD.
                The default is 0.01.
        iterations : int representation of number of iterations to perform
                     during GD. The default is 1000.

        Returns
        -------
        m : int or float representation of final slope parameter from GD
        b : int or float representation of final intercept parameter from GD
        params : list of tuples where each tuple contains a pair containing
                 the slope and intercept parameters at each step of GD
        cost : list containing float representation of error at each step of GD

        '''
        params = []
        cost = []
        for iteration in range(iterations):
            predicted_values = (m * np.array(xvalues)) + b
            errors = np.array(yvalues) - predicted_values
            # initial parameter - (learning rate * value of partial derivative
            # of cost function)
            m -= alpha * (-2 * xvalues.squeeze().dot(errors) / len(xvalues))
            b -= alpha * (-2 * errors.sum() / len(xvalues))
            new_mse = self.mse_cost(xvalues, yvalues, m, b)

            params.append((m, b))
            cost.append(new_mse)

        return m, b, params, cost

    def plot_regression(self, xvalues, yvalues, m, b, params, show_reg=True):
        '''
        Plot linear regression

        Parameters
        ----------
        xvalues : iterable containing ints or floats
        yvalues : iterable containing ints or floats
        m : int or float representation of final slope parameter from GD
        b : int or float representation of final intercept parameter from GD
        params : list of tuples where each tuple contains a pair containing
                 the slope and intercept parameters at each step of GD
        show_reg : bool determining whether to show regression lines in plot.
                   The default is True.

        Returns
        -------
        plot of linear regression

        '''
        y_pred = (m * xvalues) + b
        plt.scatter(xvalues, yvalues, marker='+', color='black', alpha=0.7)

        if show_reg:
            # Plot regression lines
            xlist = []
            ylist = []
            for param in params:
                m = param[0]
                b = param[1]
                y_preds = (m * xvalues) + b
                xlist.extend(xvalues.squeeze())
                xlist.append(None)
                ylist.extend(y_preds.squeeze())
                ylist.append(None)
            plt.plot(xlist, ylist, color='blue', alpha=0.15)

        plt.plot(xvalues, y_pred, color='blue')
        return plt.show()


if __name__ == '__main__':
    # Test with auto_generated data
    l1 = UnivariateLinearRegression()
    
    x, y = l1.generate_linear_data(250, 15, 1, 1)
    m, b, params, cost = l1.gradient_descent(x,
                                            y,
                                            m=0.5,
                                            b=0,
                                            alpha=0.01,
                                            iterations=1000
                                            )
    l1.plot_regression(x, y, m, b, params, show_reg=True)
#    sns.regplot(x.squeeze(), y.squeeze())


    # Test with "real" data
    path = '/Users/dwalkerpage/Documents/Data_Science/Projects/Univariate_'\
           'Linear_Regression_From_Scratch/Data/ex1data1.txt'
    data = pd.read_csv(path, names=['Population', 'Profit'])
    l2 = UnivariateLinearRegression()
    x = data['Population']
    y = data['Profit']
    m, b, params, cost = l2.gradient_descent(x,
                                             y,
                                             m=0.5,
                                             b=0,
                                             alpha=0.01,
                                             iterations=1000
                                             )
    l2.plot_regression(x, y, m, b, params, show_reg=True)


# =============================================================================
#  Helpful Sources:
#  https://towardsdatascience.com/gradient-descent-from-scratch-e8b75fa986cc
#  https://www.johnwittenauer.net/machine-learning-exercises-in-python-part-1/
#  https://nbviewer.jupyter.org/github/arseniyturin/sgd-from-scratch/blob/master/Gradient%20Descent.ipynb
#  https://towardsdatascience.com/gradient-descent-in-python-a0d07285742f
# =============================================================================
