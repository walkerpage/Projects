#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: dwalkerpage
"""

import numpy as np
import pandas as pd

class GaussianNaiveBayes:

    def train_test_split(self, df, test_size=0.3):
        '''
        Split data into training and test sets.

        Parameters
        ----------
        df : DataFrame containing data
        test_size : float, optional
                    Desired size of test set. The default is 0.3.

        Returns
        -------
        train_df : DataFrame containing training data
        test_df : DataFrame containing test data

        '''
        test_df = df.sample(frac=test_size) # random sample from dataset
        train_df = df.drop(test_df.index)
        return train_df, test_df

    def cross_validation_split(self, df, n_folds=5):
        '''
        Split data into specified number of subsets for cross validation.

        Parameters
        ----------
        df : DataFrame containing data
        n_folds : Integer, optional
                  Desired number of folds. The default is 5.

        Returns
        -------
        folds : Array of DataFrames

        '''
        df = df.sample(frac=1) # randomly shuffle dataset
        folds = np.array_split(df, n_folds)
        return folds

    def separate_classes(self, df):
        '''
        Separate data into separate DataFrames for each class

        Parameters
        ----------
        df : DataFrame containing data

        Returns
        -------
        separated_classes : Array of DataFrames. Each DataFrame contains all
                            data for a single class.

        '''
        separated_classes = [df[df['class'] == _class]
                             for _class in df['class'].unique()
                             ]
        return separated_classes

    def get_priors(self, df):
        '''
        Get prior probability of each class in the data.
        Prior probability is defined as the proportion of the class in the
        data set.

        Parameters
        ----------
        df : DataFrame containing data

        Returns
        -------
        priors : Series containing prior probabilities for each class.

        '''
        priors = df['class'].value_counts(normalize=True)
        return priors
    
    def get_gaussian_likelihood(self, train_df, test_df):
        '''
        Get gaussian likelihoods for each feature value for each class.

        Parameters
        ----------
        train_df : DataFrame containing training data
        test_df : DataFrame containing test data

        Returns
        -------
        class_feature_likelihoods : Array of DataFrames where each DataFrame
                        contains gaussian likelihoods of the test data for each
                        feature value for a single class.

        '''
        separated_classes = self.separate_classes(train_df)
        test_features = test_df.iloc[:,:-1]
        class_feature_likelihoods = []
        for _class in separated_classes:
            class_feature_means = _class.mean()
            class_feature_stds = _class.std()
            exponent = np.exp(((-1/2) * np.power(((test_features - class_feature_means) / class_feature_stds), 2)))
            gaussian_likelihood = (1 / (np.sqrt(2 * np.pi) * class_feature_stds)) * exponent
            class_feature_likelihoods.append(gaussian_likelihood)
        return class_feature_likelihoods

    def get_pseudo_posteriors(self, train_df, test_df, likelihood_func=None):
        '''
        Get Naive Bayes posterior probability of each data point for each
        class in the data.
        Naive Bayes posterior probability is defined as feature likelihood of
        a data point times class prior probability.
        Uses log transformations to avoid floating point precision errors.

        Parameters
        ----------
        train_df : DataFrame containing training data
        test_df : DataFrame containing test data
        likelihood_func : Function to be used to compute feature likeihoods.
                          The default is None. If no function is specified,
                          then gaussian likelihood function is used.

        Returns
        -------
        class_posteriors : DataFrame where each column contains the posterior
                           probabilities of the test data for a single class.

        '''
        priors = self.get_priors(train_df)
        if likelihood_func is None:
            likelihood_func = self.get_gaussian_likelihood
        class_feature_likelihoods = likelihood_func(train_df, test_df)
        class_posteriors = []
        for i in range(len(priors)):
            likelihoods = class_feature_likelihoods[i]
            # Log transform to avoid floating point precision errors
            log_likelihoods = likelihoods.apply(np.log)
            class_name = priors.index[i]
            prior = priors.iloc[i]
            log_prior = np.log(prior) # Log transform
            # Sum instead of product because of log transform
            feature_likelihood_sum = log_likelihoods.sum(axis=1)
            # Sum instead of product because of log transform
            posteriors = feature_likelihood_sum + log_prior
            posteriors.name = class_name
            class_posteriors.append(posteriors)
        class_posteriors = pd.concat(class_posteriors, axis=1)
        return class_posteriors

    def classify(self, posteriors):
        '''
        Classify each data point based on which class has the highest posterior
        probability.

        Parameters
        ----------
        posteriors : DataFrame where each column contains the posterior
                     probabilities of the test data for a single class.

        Returns
        -------
        classifications : Series containing a class assignment for each data
                          point.

        '''
        classifications = posteriors.idxmax(axis=1)
        return classifications

    def naive_bayes(self, train_df, test_df, likelihood_func=None):
        '''
        Execute Naive Bayes algorithm

        Parameters
        ----------
        train_df : DataFrame containing training data
        test_df : DataFrame containing test data
        likelihood_func : Function to be used to compute feature likeihoods.
                          The default is None. If no function is specified,
                          then gaussian likelihood function is used.

        Returns
        -------
        classifications : Series containing a class assignment for each data
                          point.

        '''
        class_posteriors = self.get_pseudo_posteriors(train_df,
                                                      test_df,
                                                      likelihood_func=likelihood_func
                                                      )
        classifications = self.classify(class_posteriors)
        return classifications

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

    def evaluate_algorithm(self, df,
                           algorithm=None,
                           n_folds=5,
                           performance_metric=None,
                           likelihood_func=None):
        '''
        Execute cross validation


        Parameters
        ----------
        df : DataFrame containing data
        algorithm : Algorithm to be used in cross validation, optional
                    The default is None. If no algorithm is specified, then
                    Naive Bayes is used.
        n_folds : Integer, optional
                  Desired number of folds. The default is 5.
        performance_metric : Performance metric to be used, optional
                             The default is None. If no metric is specified,
                             Accuracy is used.
        likelihood_func : Function to be used to compute feature likeihoods.
                          The default is None. If no function is specified,
                          then gaussian likelihood function is used.

        Returns
        -------
        performance_scores : Array of floats representing the performance score
                             for each iteration of cross validation.

        '''
        folds = self.cross_validation_split(df, n_folds=n_folds)
        performance_scores = []
        for i in range(len(folds)):
            test = folds[i]
            # train is all data not assigned to test
            train = pd.concat(folds[:i] + folds[i+1:])
            if algorithm is None:
                algorithm = self.naive_bayes
            classifications = algorithm(train,
                                        test,
                                        likelihood_func=likelihood_func
                                        )
            if performance_metric is None:
                performance_metric = self.accuracy
            performance_score = performance_metric(test['class'],
                                                   classifications
                                                   )
            performance_scores.append(performance_score)
        return performance_scores


if __name__ == '__main__':

    data = pd.read_csv('iris.csv',
                       names=['sepal_length',
                              'sepal_width',
                              'petal_length',
                              'petal_width',
                              'class'
                              ]
                       )

    gnb_clf = GaussianNaiveBayes()

    train, test = gnb_clf.train_test_split(data)
    folds = gnb_clf.cross_validation_split(data, n_folds=5)
    separated_classes = gnb_clf.separate_classes(data)
    priors = gnb_clf.get_priors(data)
    class_feature_likelihoods = gnb_clf.get_gaussian_likelihood(train, test)
    class_posteriors = gnb_clf.get_pseudo_posteriors(train, test)
    classifications = gnb_clf.classify(class_posteriors)
    print(gnb_clf.accuracy(test['class'], classifications))
    print(gnb_clf.error(test['class'], classifications))
    print(gnb_clf.confusion_matrix(test['class'], classifications))
    scores = gnb_clf.evaluate_algorithm(data)




# =============================================================================
# Helpful Sources:
# https://chrisalbon.com/machine_learning/naive_bayes/naive_bayes_classifier_from_scratch/
# https://machinelearningmastery.com/naive-bayes-classifier-scratch-python/
# https://machinelearningmastery.com/classification-as-conditional-probability-and-the-naive-bayes-algorithm/
# https://machinelearningmastery.com/better-naive-bayes/
# http://www.cs.rhodes.edu/~kirlinp/courses/ai/f18/projects/proj3/naive-bayes-log-probs.pdf
# https://stats.stackexchange.com/questions/163088/how-to-use-log-probabilities-for-gaussian-naive-bayes
# =============================================================================
'''
Bayes Theorem:
P(class | data) = (P(data | class) * P(class)) / P(data)
- posterior = P(class | data)
- likelihood = P(data | class)
- prior = P(class)
- marginal probability = P(data)

Naive Bayes:
P(class | data) = (P(data | class) * P(class))
- Assumes, naively, that each feature variable is independent of the others.
- Eliminates marginal probability from Bayes Theorem since for classification
we care only about the highest relative posterior probability among the
classes, which is not affected by marginal probability.
'''