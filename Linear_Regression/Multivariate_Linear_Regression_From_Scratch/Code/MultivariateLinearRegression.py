#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: dwalkerpage
"""

import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

%matplotlib inline

class MultivariateLinearRegression:

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

    def mse_cost(self, xvalues, yvalues, Ms, b):
        '''
        Compute mean squared error

        Parameters
        ----------
        xvalues : array containing each independent variable
        yvalues : array containing values for the dependent variable
        Ms : iterable containing int or float representation of slope
             parameters for weighting the xvalues
        b : int or float representation of intercept parameter

        Returns
        -------
        mse : float representation of mean squared error

        '''
        xvalues = np.array(xvalues)
        Ms = np.array(Ms)
        if len(Ms) == xvalues.shape[1]:
            predicted_values = (Ms * xvalues).sum(axis=1) + b
            sq_errors = np.power((yvalues - predicted_values), 2)
            rss = np.sum(sq_errors)
            mse = rss / len(yvalues)
            return mse
        else:
            raise ValueError('The number of slope parameters must equal the '
                             'number of independent variables.')

    def gradient_descent(self, xvalues, yvalues, Ms, b=0, alpha=0.01,
                         iterations=1000):
        '''
        Execute basic gradient descent

        Parameters
        ----------
        xvalues : array containing each independent variable
        yvalues : array containing values for the dependent variable
        Ms : iterable containing int or float representation of slope
             parameters for weighting the xvalues
        b : int or float representation of initial intercept parameter.
            Default is 0.
        alpha : int or float representation of initial learning rate for GD.
                The default is 0.01.
        iterations : int representation of number of iterations to perform
                     during GD. The default is 1000.

        Returns
        -------
        Ms : iterable containing final slope parameters from GD
        b : int or float representation of final intercept parameter from GD
        params : list of tuples where each tuple contains a pair containing
                 the slope and intercept parameters at each step of GD
        cost : list containing float representation of error at each step of GD

        '''
        xvalues = np.array(xvalues)
        yvalues = np.array(yvalues)
        Ms = np.array(Ms)
        params = []
        cost = []
        for iteration in range(iterations):
            predicted_values = (Ms * xvalues).sum(axis=1) + b
            errors = yvalues - predicted_values
            # initial parameter - (learning rate * value of partial derivative
            # of cost function)
            Ms = Ms - (alpha * (-2 * errors.dot((xvalues)) / len(yvalues)))
            b = b - (alpha * (-2 * errors.sum() / len(yvalues)))
            new_mse = self.mse_cost(xvalues, yvalues, Ms, b)

            params.append((Ms, b))
            cost.append(new_mse)

        return Ms, b, params, cost

    def plot_descent(self, costs, iterations):
        '''
        Plot descent of cost with each iteration during gradient descent

        Parameters
        ----------
        costs : list containing float representation of error at each step of GD
        iterations : int representation of number of iterations performed
                     during GD.

        Returns
        -------
        None.

        '''
        plt.plot(range(iterations), costs)
        plt.xlabel('Iterations')
        plt.ylabel('Cost')
        plt.title('Cost vs. Training Iterations')

if __name__ == '__main__':

    # Test with auto-generated data, two independent variables
    m1 = MultivariateLinearRegression()    
    Xs, y = m1.generate_multivariable_linear_data(data_size=250,
                                                  Xmaxs=[50, 150],
                                                  slopes=[1, 2],
                                                  intercept=1)
    norm_Xs = m1.feat_norm(Xs)
    Ms, b, params, cost = m1.gradient_descent(norm_Xs,
                                              y,
                                              Ms=[0, 0],
                                              b=0,
                                              alpha=0.01,
                                              iterations=5000)
    m1.plot_descent(cost, 5000)
    

    # Test with auto-generated data, three independent variables
    m2 = MultivariateLinearRegression()    
    Xs, y = m2.generate_multivariable_linear_data(data_size=250,
                                                  Xmaxs=[50, 150, 250],
                                                  slopes=[1, 2, 3],
                                                  intercept=1)
    norm_Xs = m2.feat_norm(Xs)
    Ms, b, params, cost = m2.gradient_descent(norm_Xs,
                                              y,
                                              Ms=[0, 0, 0],
                                              b=0,
                                              alpha=0.01,
                                              iterations=5000)
    m2.plot_descent(cost, 5000)


    # Test with auto-generated data, five independent variables
    m3 = MultivariateLinearRegression()    
    Xs, y = m3.generate_multivariable_linear_data(data_size=250,
                                                  Xmaxs=[50, 150, 250, 350, 450],
                                                  slopes=[1, 2, 3, 4, 5],
                                                  intercept=1)
    norm_Xs = m3.feat_norm(Xs)
    Ms, b, params, cost = m3.gradient_descent(norm_Xs,
                                              y,
                                              Ms=[0, 0, 0, 0, 0],
                                              b=0,
                                              alpha=0.01,
                                              iterations=5000)
    m3.plot_descent(cost, 5000)


    # Test with "real" data
    m4 = MultivariateLinearRegression()
    path = '/Users/dwalkerpage/Documents/Data_Science/Projects/Linear_Regres'\
           'sion/Multivariate_Linear_Regression_From_Scratch/Data/ex1data2.txt'
    data = pd.read_csv(path, names=['Size', 'Bedrooms', 'Price'])
    Xs = np.array(data[['Size', 'Bedrooms']])
    y = np.array(data['Price'])
    norm_Xs = m4.feat_norm(Xs)
    Ms, b, params, cost = m4.gradient_descent(norm_Xs,
                                              y,
                                              Ms=[0, 0],
                                              b=0,
                                              alpha=0.01,
                                              iterations=5000)
    m4.plot_descent(cost, 5000)


# =============================================================================
# Helpful Sources:
# https://towardsdatascience.com/gradient-descent-from-scratch-e8b75fa986cc
# https://www.johnwittenauer.net/machine-learning-exercises-in-python-part-2/
# =============================================================================