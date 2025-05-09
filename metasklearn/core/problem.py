#!/usr/bin/env python
# Created by "Thieu" at 07:31, 08/05/2025 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import numpy as np
from sklearn.model_selection import KFold, cross_val_score
from mealpy import Problem


class HyperparameterProblem(Problem):
    """
    A class to define a hyperparameter optimization problem for machine learning models.

    Inherits from the `Problem` class in the `mealpy` library and provides functionality
    to evaluate hyperparameter configurations using cross-validation.

    Attributes:
        estimator: The machine learning model to optimize.
        X: The feature matrix.
        y: The target vector.
        metric_class: A custom metric class for evaluation.
        obj_name: The name of the objective metric.
        cv: The number of cross-validation folds.
        n_jobs: The number of parallel jobs for cross-validation.
        shuffle: Whether to shuffle the data before splitting into folds.
        kf: The KFold cross-validator instance.
        get_obj_score_: The scoring function to use (either sklearn or custom).
    """

    def __init__(self, bounds=None, minmax="max", X=None, y=None, estimator=None, metric_class=None,
                 obj_name=None, sklearn_score=None, cv=None, n_jobs=None, shuffle=True, seed=None, **kwargs):
        """
        Initializes the HyperparameterProblem instance.

        Args:
            bounds: The bounds for the hyperparameters.
            minmax: The optimization direction ("max" for maximization, "min" for minimization).
            X: The feature matrix.
            y: The target vector.
            estimator: The machine learning model to optimize.
            metric_class: A custom metric class for evaluation.
            obj_name: The name of the objective metric.
            sklearn_score: Whether to use sklearn's scoring function.
            cv: The number of cross-validation folds.
            n_jobs: The number of parallel jobs for cross-validation.
            shuffle: Whether to shuffle the data before splitting into folds.
            seed: The random seed for reproducibility.
            **kwargs: Additional arguments for the parent class.
        """
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
        """
        Computes the cross-validation score using sklearn's scoring function.

        Returns:
            float: The mean cross-validation score.
        """
        scores = cross_val_score(self.estimator, self.X, self.y,
                                 cv=self.cv, scoring=self.obj_name, n_jobs=self.n_jobs)
        return np.mean(scores)

    def _get_custom_score(self):
        """
        Computes the cross-validation score using a custom scoring function from PerMetrics library

        Returns:
            float: The mean cross-validation score.
        """
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
        """
        Objective function to evaluate a hyperparameter configuration.

        Args:
            x: The encoded hyperparameter configuration.

        Returns:
            float: The evaluation score for the given configuration.
        """
        x_decoded = self.decode_solution(x)
        self.model = self.estimator.set_params(**x_decoded)
        score = self.get_obj_score_()
        return score
