#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: dwalkerpage
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.metrics import roc_auc_score

%matplotlib inline

class BinaryLogisticRegression:

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
        standardized_xvalues : array containing each independent variable
                               standardized

        '''
        xvalues = np.array(xvalues)
        standardized_xvalues = ((xvalues - xvalues.mean(axis=0))
                               / xvalues.std(axis=0))
        return standardized_xvalues    

    def logistic_sigmoid(self, z):
        '''
        Transform linear values to values between 0 and 1

        Parameters
        ----------
        z : array of linear values

        Returns
        -------
        predicted_values : array containing values between 0 and 1

        '''
        return 1 / (1 + np.exp(-z))

    def predict(self, xvalues, Ms, b=0):
        '''
        Generate and transform linear predictions from independent variables

        Parameters
        ----------
        xvalues : array containing each independent variable
        Ms : array containing slope parameters/coefficients for each
             independent variable.
        b : int or float representation of intercept parameter. Default is 0.

        Raises
        ------
        ValueError
            Independent variables must be scaled.
            The number of slope parameters must equal the number of independent
            variables.

        Returns
        -------
        predicted_values : array containing predictions for the dependent
                           variable

        '''
        xvalues = np.array(xvalues)
        Ms = np.array(Ms)
        if len(Ms) == xvalues.shape[1]:
            linear_values = (Ms * xvalues).sum(axis=1) + b
            predicted_values = self.logistic_sigmoid(linear_values)
            if 1 in predicted_values:
                raise ValueError('The independent variables must be scaled.')
            else:
                return predicted_values
        else:
            raise ValueError('The number of slope parameters must equal the '
                             'number of independent variables.')

    def cost(self, xvalues, yvalues, predicted_values):
        '''
        Compute logistic regression cost function.

        Parameters
        ----------
        xvalues : array containing each independent variable
        yvalues : array containing values for the dependent variable
        predicted_values : array containing predictions for the dependent
                           variable

        Returns
        -------
        average_cost : float representation of the average cost for the
                       predictions

        '''
        xvalues = np.array(xvalues)
        costs = ((-yvalues * np.log(predicted_values))
                - ((1 - yvalues) * np.log(1 - predicted_values)))
        average_cost = np.sum(costs) / len(yvalues)
        return average_cost

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
            predicted_values = self.predict(xvalues, Ms, b=b)
            errors = predicted_values - yvalues
            # initial parameter - (learning rate * value of partial derivative
            # of cost function)
            Ms = Ms - (alpha * (errors.dot(xvalues).sum() / len(yvalues)))
            b = b - (alpha * (errors.dot(xvalues).sum() / len(yvalues)))
            new_predicted_values = self.predict(xvalues, Ms, b=b)
            new_cost = self.cost(xvalues, yvalues, new_predicted_values)

            params.append((Ms, b))
            cost.append(new_cost)

        return Ms, b, params, cost

    def classify(self, predictions, threshold=0.5):
        '''
        Assign predicted labels to predictions based on an assigned threshold.

        Parameters
        ----------
        predictions : Probability predictions for dependent variable.
        threshold : Threshold for assigning a label to a given predictions.
                    The default is 0.5.

        Returns
        -------
        classifications : Array of labels assigned for each prediction

        '''
        classifications = np.array([1 if prediction >= threshold else 0
                                    for prediction in predictions
                                    ]
                                   )
        return classifications

    def plot_descent(self, costs, iterations):
        '''
        Plot descent of cost with each iteration during gradient descent

        Parameters
        ----------
        costs : list containing float representation of error at each step of
                GD
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
        plt.show()

    def accuracy(self, actual_labels, predicted_labels):
        '''
        Compute Accuracy score (total correct predictions / total number of
        predictions)

        Parameters
        ----------
        actual_labels : Array of actual labels for each observation
        predicted_labels : Array of labels assigned for each prediction

        Returns
        -------
        percentage_correct : float representation of accuracy score

        '''
        total_correct = sum(actual_labels == predicted_labels)
        percentage_correct = 100 * (total_correct / len(actual_labels))
        return percentage_correct

    def error(self, actual_labels, predicted_labels):
        '''
        Compute Error score (total incorrect predictions / total number of
        predictions)

        Parameters
        ----------
        actual_labels : Array of actual labels for each observation
        predicted_labels : Array of labels assigned for each prediction

        Returns
        -------
        float representation of error score

        '''
        return 100 - self.accuracy(actual_labels, predicted_labels)

    def confusion_matrix(self, actual_labels, predicted_labels):
        '''
        Compute confusion matrix containing number of false positives, true
        positives, false negatives, and true negatives

        Parameters
        ----------
        actual_labels : Array of actual labels for each observation
        predicted_labels : Array of labels assigned for each prediction

        Returns
        -------
        confusion_df : 2x2 DataFrame containing number of false positives,
                      true positives, false negatives, and true negatives

        '''
        actual_labels = pd.Series(actual_labels, name='Actual_Class')
        predicted_labels = pd.Series(predicted_labels, name='Classification')
        confusion_df = pd.crosstab(actual_labels,
                                   predicted_labels,
                                   margins=True
                                   )
        print(confusion_df)
        return confusion_df

    def precision(self, actual_labels, predicted_labels):
        '''
        Compute precision scores (number of instances correctly assigned a
        label / the total number of instances assigned that label) for
        positive and negative classes

        Parameters
        ----------
        actual_labels : Array of actual labels for each observation
        predicted_labels : Array of labels assigned for each prediction

        Returns
        -------
        positive_precision : precision score for positive class
        negative_precision : precision score for negative class

        '''
        # Find number of true positives
        tp_n = len([predicted_labels[i]
                    for i in range(len(predicted_labels))
                    if (predicted_labels[i] == 1
                        and predicted_labels[i] == actual_labels[i]
                        )
                    ]
                   )
        # Find number of true negatives
        tn_n = len([predicted_labels[i]
                    for i in range(len(predicted_labels))
                    if (predicted_labels[i] == 0
                        and predicted_labels[i] == actual_labels[i]
                        )
                    ]
                   )
        # Find number of positive predictions
        pos_preds_n = len([i for i in predicted_labels if i == 1])
        # Find number of negative predictions
        neg_preds_n = len([i for i in predicted_labels if i == 0])
        try:
            positive_precision = tp_n / pos_preds_n
        except ZeroDivisionError:
            positive_precision = 0
        try:
            negative_precision = tn_n / neg_preds_n
        except ZeroDivisionError:
            negative_precision = 0
        return (positive_precision, negative_precision)

    def recall(self, actual_labels, predicted_labels):
        '''
        Compute recall scores (number of instances correctly assigned a
        label / the total number of instances actually in that label's class)
        for positive and negative classes

        Parameters
        ----------
        actual_labels : Array of actual labels for each observation
        predicted_labels : Array of labels assigned for each prediction

        Returns
        -------
        positive_recall : recall score for positive class
        negative_recall : recall score for negative class

        '''
        # Find number of true positives
        tp_n = len([predicted_labels[i]
                    for i in range(len(predicted_labels))
                    if (predicted_labels[i] == 1
                        and predicted_labels[i] == actual_labels[i]
                        )
                    ]
                   )
        # Find number of true negatives
        tn_n = len([predicted_labels[i]
                    for i in range(len(predicted_labels))
                    if (predicted_labels[i] == 0
                        and predicted_labels[i] == actual_labels[i]
                        )
                    ]
                   )
        # Find number of actual positives
        pos_actual_n = len([i for i in actual_labels if i == 1])
        # Find number of actual negatives
        neg_actual_n = len([i for i in actual_labels if i == 0])
        try:
            positive_recall = tp_n / pos_actual_n
        except ZeroDivisionError:
            positve_recall = 0
        try:
            negative_recall = tn_n / neg_actual_n
        except ZeroDivisionError:
            negative_recall = 0
        return (positive_recall, negative_recall)

    def F_score(self, actual_labels, predicted_labels):
        '''
        Compute F-score (harmonic mean of precision and recall) for positive
        and negative classes

        Parameters
        ----------
        actual_labels : Array of actual labels for each observation
        predicted_labels : Array of labels assigned for each prediction

        Returns
        -------
        pos_F_score : F-score for positive class
        neg_F_score : F-score for negative class

        '''
        pos_precision, neg_precision = self.precision(actual_labels,
                                                      predicted_labels
                                                      )
        pos_recall, neg_recall = self.recall(actual_labels, predicted_labels)
        try:
            pos_F_score = (2 * pos_precision * pos_recall) \
                          / (pos_precision + pos_recall)
        except ZeroDivisionError:
            pos_F_score = 0
        try:
            neg_F_score = (2 * neg_precision * neg_recall) \
                          / (neg_precision + neg_recall)
        except ZeroDivisionError:
            neg_F_score = 0
        return (pos_F_score, neg_F_score)

    def tp_and_fp_rates(self, actual_labels, predictions, thresholds):
        '''
        Compute true positive rate (number of true positives / number of actual
        positives) and false positive rate (number of false positives / number
        of actual negatives) for multiple classification threshold values

        Parameters
        ----------
        actual_labels : Array of actual labels for each observation
        predictions : Array of probability values for each observation
        thresholds : Array of classification threshold values

        Returns
        -------
        tp_rates : Array containing true positive rate for each classification
                  threshold value
        fp_rates : Array containing false positive rate for each classification
                  threshold value

        '''
        tp_rates = []
        fp_rates = []
        for threshold in thresholds:
            predicted_labels = self.classify(predictions, threshold=threshold)
            tp_n = len([predicted_labels[i]
                        for i in range(len(predicted_labels))
                        if (predicted_labels[i] == 1
                            and predicted_labels[i] == actual_labels[i]
                            )
                        ]
                       )
            fp_n = len([predicted_labels[i]
                        for i in range(len(predicted_labels))
                        if (predicted_labels[i] == 1
                            and predicted_labels[i] != actual_labels[i]
                            )
                        ]
                       )
            pos_actual_n = len([i for i in actual_labels if i == 1])
            neg_actual_n = len([i for i in actual_labels if i == 0])
            try:
                tpr = tp_n / pos_actual_n
            except ZeroDivisionError:
                tpr = 0
            try:
                fpr = fp_n / neg_actual_n
            except ZeroDivisionError:
                fpr = 0
            tp_rates.append(tpr)
            fp_rates.append(fpr)
        return (tp_rates, fp_rates)

    def roc_curve(self, tp_rates, fp_rates):
        '''
        Plot ROC Curve with false positive rates on x-axis and true positive
        rates on y-axis

        Parameters
        ----------
        tp_rates : Array containing true positive rate for each classification
                  threshold value
        fp_rates : Array containing false positive rate for each classification
                  threshold value

        Returns
        -------
        None.

        '''
        plt.plot(fp_rates, tp_rates, marker='.')
        plt.xlabel('False Positive Rate (false positives / actual negatives)')
        plt.ylabel('True Positive Rate (true positives / actual positives)')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.show()

    def roc_auc_score(self, actual_labels, predictions):
        '''
        Return ROC AUC (area under curve).

        Parameters
        ----------
        actual_labels : Array of actual labels for each observation
        predictions : Array of probability values for each observation

        Returns
        -------
        Float representation of ROC AUC value

        '''
        return roc_auc_score(actual_labels, predictions)

    def precision_recall_curve(self, actual_labels, predictions, thresholds):
        '''
        Plot precision-recall curve for positive and negative classes with
        recall scores on x-axis and precision scores on y-axis

        Parameters
        ----------
        actual_labels : Array of actual labels for each observation
        predictions : Array of probability values for each observation
        thresholds : Array of classification threshold values

        Returns
        -------
        None.

        '''
        pos_precision_scores = []
        neg_precision_scores = []
        pos_recall_scores = []
        neg_recall_scores = []
        for threshold in thresholds:
            predicted_labels = self.classify(predictions, threshold=threshold)
            pos_precision, neg_precision = self.precision(actual_labels,
                                                          predicted_labels
                                                          )
            pos_recall, neg_recall = self.recall(actual_labels,
                                                 predicted_labels
                                                 )
            pos_precision_scores.append(pos_precision)
            neg_precision_scores.append(neg_precision)
            pos_recall_scores.append(pos_recall)
            neg_recall_scores.append(neg_recall)
        plt.plot(pos_recall_scores,
                 pos_precision_scores,
                 marker='.',
                 label='Positive Class'
                 )
        plt.plot(neg_recall_scores,
                 neg_precision_scores,
                 marker='.',
                 label='Negative Class'
                 )
        plt.xlabel('Recall Scores')
        plt.ylabel('Precision Scores')
        plt.title('Precision-Recall Curves for Positive and Negative Classes')
        plt.legend()
        plt.show()


if __name__ == '__main__':

    # Test with "Real" Data
    path = '/Users/dwalkerpage/Documents/Data_Science/Projects/Logistic_Regre'\
           'ssion/Binary_Logistic_Regression_From_Scratch/Data/ex2data1.txt'
    data = pd.read_csv(path, names=['Test 1', 'Test 2', 'Admission'])

    admitted = data[data['Admission'] == 1]
    not_admitted = data[data['Admission'] == 0]
    
    fig, ax = plt.subplots(figsize=(8,5))
    ax.scatter(admitted['Test 1'],
               admitted['Test 2'],
               c='b',
               marker='.',
               label='Admitted'
               )
    ax.scatter(not_admitted['Test 1'],
               not_admitted['Test 2'],
               c='r',
               marker='x',
               label='Not Admitted'
               )
    ax.legend()
    ax.set_xlabel('Test 1 Score')
    ax.set_ylabel('Test 2 Score')
    plt.scatter


    # Testing/Troubleshooting
    blr = BinaryLogisticRegression()
    xvalues = np.array(data[['Test 1', 'Test 2']])
    norm_xvalues = blr.feat_norm(xvalues)
    yvalues = data['Admission']
    Ms = np.array([1, 1])
    b = 0
    f_Ms, f_b, params, cost = blr.gradient_descent(norm_xvalues, yvalues, Ms, b=b)
    predictions = blr.predict(norm_xvalues, f_Ms, b=f_b)
    classifications = blr.classify(predictions)
    blr.plot_descent(cost, 1000)
    print(blr.accuracy(yvalues, classifications))
    blr.confusion_matrix(yvalues, classifications)
    print(blr.precision(yvalues, classifications))
    print(blr.recall(yvalues, classifications))
    print(blr.F_score(yvalues, classifications))
    tp_rates, fp_rates = blr.tp_and_fp_rates(yvalues, predictions, [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    blr.roc_curve(tp_rates, fp_rates)
    print(blr.roc_auc_score(yvalues, predictions))
    blr.precision_recall_curve(yvalues, predictions, [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])




# =============================================================================
# Helpful Sources:
# https://www.johnwittenauer.net/machine-learning-exercises-in-python-part-3/
# https://ml-cheatsheet.readthedocs.io/en/latest/logistic_regression.html
# https://stackoverflow.com/questions/36229340/divide-by-zero-encountered-in-log-when-not-dividing-by-zero/43419703
# https://medium.com/analytics-vidhya/derivative-of-log-loss-function-for-logistic-regression-9b832f025c2d
# https://en.wikipedia.org/wiki/Confusion_matrix
# https://towardsdatascience.com/beyond-accuracy-precision-and-recall-3da06bea9f6c
# https://towardsdatascience.com/understanding-confusion-matrix-a9ad42dcfd62
# https://machinelearningmastery.com/roc-curves-and-precision-recall-curves-for-classification-in-python/
# https://machinelearningmastery.com/precision-recall-and-f-measure-for-imbalanced-classification/
# =============================================================================
