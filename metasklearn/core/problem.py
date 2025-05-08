#!/usr/bin/env python
# Created by "Thieu" at 07:31, 08/05/2025 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import numpy as np
from sklearn.model_selection import KFold, cross_val_score
from sklearn.base import BaseEstimator, clone
from mealpy import Problem


class HyperparameterProblem(Problem):
    """
    This class defines the Hyper-parameter tuning problem that will be used for Mealpy library.

    Parameters
    ----------
    bounds : from Mealpy library.

    minmax : from Mealpy library.

    X : array-like of shape (n_samples, n_features)
        Test samples. For some estimators this may be a precomputed kernel matrix or a list of generic objects instead with shape
        ``(n_samples, n_samples_fitted)``, where ``n_samples_fitted`` is the number of samples used in the fitting for the estimator.

    y : array-like of shape (n_samples,) or (n_samples, n_outputs)
        True values for `X`.

    model_class : RvflRegressor or RvflClassifier
        The class definition of RVFL network for regression or classification problem.

    metric_class : RegressionMetric or ClassificationMetric
        The class definition of Performance Metrics for regression or classification problem.

    obj_name : str
        The name of the loss function used in network

    cv : int, default=None
        The k fold cross-validation method

    shuffle: bool, default=True
        Shuffle or not the dataset when performs k-fold cross validation.

    seed: int, default=None
        Determines random number generation for weights and bias initialization.
        Pass an int for reproducible results across multiple function calls.
    """

    def __init__(self, bounds=None, minmax="max", X=None, y=None, estimator=None, metric_class=None,
                 obj_name=None, sklearn_score=None, cv=None, n_jobs=None, shuffle=True, seed=None, **kwargs):
        self.estimator = estimator
        self.X = X
        self.y = y
        self.metric_class = metric_class
        self.obj_name = obj_name

        if sklearn_score:
            self.get_obj_score_ = self._get_sklearn_score
        else:
            self.get_obj_score_ = self._get_custom_score

        self.cv = cv
        if cv is None or cv < 2:
            self.cv = 2
        self.n_jobs = n_jobs
        self.shuffle = shuffle
        self.kf = KFold(n_splits=self.cv, shuffle=shuffle, random_state=seed)
        super().__init__(bounds, minmax, **{**kwargs, "seed":seed})

    def _get_sklearn_score(self):
        scores = cross_val_score(self.estimator, self.X, self.y,
                                 cv=self.cv, scoring=self.obj_name, n_jobs=self.n_jobs)
        return np.mean(scores)

    def _get_custom_score(self):
        scores = []
        # Perform custom cross-validation
        for train_idx, test_idx in self.kf.split(self.X):
            # Split the data into training and test sets
            X_train, X_test = self.X[train_idx], self.X[test_idx]
            y_train, y_test = self.y[train_idx], self.y[test_idx]
            # Train the model on the training set
            self.estimator.fit(X_train, y_train)
            # Make predictions on the test set
            y_pred = self.estimator.predict(X_test)
            # Calculate accuracy for the current fold
            mt = self.metric_class(y_test, y_pred)
            score = mt.get_metric_by_name(self.obj_name)[self.obj_name]
            # Accumulate accuracy across folds
            scores.append(score)
        return np.mean(scores)

    def obj_func(self, x):
        x_decoded = self.decode_solution(x)
        self.model = self.estimator.set_params(**x_decoded)
        score = self.get_obj_score_()
        return score
