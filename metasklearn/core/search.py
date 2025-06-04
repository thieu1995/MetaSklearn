#!/usr/bin/env python
# Created by "Thieu" at 07:11, 08/05/2025 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import inspect
import pickle
import pprint
from pathlib import Path
import numpy as np
import pandas as pd
from permetrics import RegressionMetric, ClassificationMetric
from sklearn.base import BaseEstimator
from sklearn.metrics import get_scorer_names
from mealpy import get_optimizer_by_class, Optimizer
from metasklearn.core.problem import HyperparameterProblem
from metasklearn.utils import validation
from metasklearn.utils.evaluation import get_all_classification_metrics, get_all_regression_metrics

class MetaSearchCV(BaseEstimator):
    """
    A metaheuristic-powered hyperparameter optimization framework for scikit-learn models.

    This class uses metaheuristic optimization algorithms from the Mealpy library to perform
    hyperparameter tuning for scikit-learn-compatible models via cross-validation.

    Parameters:
        estimator (BaseEstimator):
            The machine learning model to optimize. Must implement scikit-learn's fit/predict interface.

        param_bounds (list, tuple, or dict):
            A dictionary specifying the boundary of the hyperparameters to be optimized.

        task_type (str, default='classification'):
            The type of task: 'classification' or 'regression'. Determines the evaluation metric used.

        optim (str or Optimizer, default='BaseGA'):
            The name of the metaheuristic algorithm to use (from Mealpy), or an Optimizer instance.

        optim_params (dict, optional):
            Dictionary of additional parameters passed to the optimizer (e.g., pop_size, epoch, etc.).

        cv (int, default=5):
            Number of cross-validation folds.

        scoring (str, optional):
            Name of the scoring metric. Can be a scikit-learn metric or a custom metric supported by permetrics.

        seed (int, optional):
            Random seed for reproducibility.

        n_jobs (int, default=1):
            Number of jobs to run in parallel during cross-validation.

        verbose (bool, default=True):
            Whether to display logs and progress during optimization.

        mode (str, default='single'):
            Execution mode for the optimizer: 'single', 'swarm', 'thread', or 'process'.

        n_workers (int, optional):
            Number of parallel workers used by the optimizer in threaded or multiprocessing mode.

        termination (dict, optional):
            Dictionary defining custom termination conditions for the optimizer.

        **kwargs:
            Additional keyword arguments passed to the internal problem definition.

    Attributes:
        best_params (dict):
            The best hyperparameter configuration found during optimization.

        best_estimator (BaseEstimator):
            A clone of the input estimator trained with the best-found parameters.

        best_score (float):
            The best evaluation score achieved during optimization.

        loss_train (list):
            A list of best scores over iterations.

        problem (HyperparameterProblem):
            Internal representation of the hyperparameter optimization problem.
    """

    SUPPORTED_CLS_METRICS = get_all_classification_metrics()
    SUPPORTED_REG_METRICS = get_all_regression_metrics()

    def __init__(self, estimator, param_bounds, task_type="classification",
                 optim="BaseGA", optim_params=None,
                 cv=5, scoring=None, seed=None, n_jobs=1, verbose=True,
                 mode='single', n_workers=None, termination=None, **kwargs):
        self.estimator = estimator
        self.param_bounds = param_bounds

        if task_type == "regression":
            self.task_type = task_type
            self.metric_class = RegressionMetric
        else:
            self.task_type = "classification"
            self.metric_class = ClassificationMetric

        self.scoring_name = scoring
        if scoring in get_scorer_names():
            self.sklearn_score = True
            self.minmax = "max"
        else:
            self.sklearn_score = False
            if task_type == "regression":
                self.scoring_name = validation.check_str("scoring", scoring, self.SUPPORTED_REG_METRICS)
                self.minmax = self.SUPPORTED_REG_METRICS[self.scoring_name]
            else:
                self.scoring_name = validation.check_str("scoring", scoring, self.SUPPORTED_CLS_METRICS)
                self.minmax = self.SUPPORTED_CLS_METRICS[self.scoring_name]

        self.optim = optim
        self.optim_params = optim_params
        self.cv = cv
        self.seed = seed
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.mode = mode
        self.n_workers = n_workers
        self.termination = termination

        self.best_params = None
        self.best_estimator = None
        self.best_score = None
        self.loss_train = None
        self.problem = None
        self.kwargs = kwargs

    def __repr__(self, **kwargs):
        """
        Returns a string representation of the MetaSearchCV instance.

        Returns:
            str: A formatted string of the instance's parameters.
        """
        param_order = list(inspect.signature(self.__init__).parameters.keys())
        param_dict = {k: getattr(self, k) for k in param_order}

        param_str = ", ".join(f"{k}={repr(v)}" for k, v in param_dict.items())
        if len(param_str) <= 80:
            return f"{self.__class__.__name__}({param_str})"
        else:
            formatted_params = ",\n  ".join(f"{k}={pprint.pformat(v)}" for k, v in param_dict.items())
            return f"{self.__class__.__name__}(\n  {formatted_params}\n)"

    def _set_optimizer(self, optim=None, optim_params=None):
        """
        Sets the optimizer for the hyperparameter search.

        Args:
            optim: The name of the optimizer or an Optimizer instance.
            optim_params: Parameters for the optimizer.

        Returns:
            Optimizer: An instance of the optimizer.

        Raises:
            TypeError: If the `optim` parameter is not a string or Optimizer instance.
        """
        if type(optim) is str:
            opt_class = get_optimizer_by_class(optim)
            if type(optim_params) is dict:
                return opt_class(**optim_params)
            else:
                return opt_class(epoch=250, pop_size=20)
        elif isinstance(optim, Optimizer):
            if type(optim_params) is dict:
                if "name" in optim_params:  # Check if key exists and remove it
                    optim.name = optim_params.pop("name")
                optim.set_parameters(optim_params)
            return optim
        else:
            raise TypeError(f"`optim` parameter needs to set as a string and supported by Mealpy library.")

    def fit(self, X, y):
        """
        Fits the model using the provided data and performs hyperparameter optimization.

        Args:
            X: The feature matrix.
            y: The target vector.

        Returns:
            MetaSearchCV: The fitted instance.
        """
        log_to = "console" if self.verbose else "None"
        self.optim_params = self.optim_params or {}
        self.optim = self._set_optimizer(self.optim, self.optim_params)
        self.problem = HyperparameterProblem(self.param_bounds, self.minmax, X, y,
                                             self.estimator, self.metric_class,
                                             obj_name=self.scoring_name, sklearn_score=self.sklearn_score,
                                             cv=self.cv, n_jobs=None, shuffle=True, seed=self.seed,
                                             log_to=log_to, **self.kwargs)
        self.optim.solve(self.problem, mode=self.mode, n_workers=self.n_workers, termination=self.termination, seed=self.seed)
        self.best_params = self.optim.problem.decode_solution(self.optim.g_best.solution)
        self.best_estimator = self.estimator.set_params(**self.best_params)
        self.best_estimator.fit(X, y)
        self.best_score = self.optim.g_best.target.fitness
        self.loss_train = self.optim.history.list_global_best_fit
        return self

    def predict(self, X):
        """
        Predicts the target values for the given feature matrix.

        Args:
            X: The feature matrix.

        Returns:
            np.ndarray: The predicted target values.

        Raises:
            ValueError: If the model is not trained.
        """
        if self.best_params is None or self.best_estimator is None:
            raise ValueError(f"Model is not trained, please call the fit() function.")
        return self.best_estimator.predict(X)

    def score(self, X, y):
        """
        Computes the score of the model on the given data.

        Args:
            X: The feature matrix.
            y: The target vector.

        Returns:
            float: The score of the model.

        Raises:
            ValueError: If the model is not trained.
        """
        if self.best_params is None or self.best_estimator is None:
            raise ValueError(f"Model is not trained, please call the fit() function.")
        return self.best_estimator.score(X, y)

    def evaluate(self, y_true, y_pred, list_metrics=("AS", "RS")):
        """
        Evaluates the model's predictions using the specified metrics.

        Args:
            y_true: The ground truth target values.
            y_pred: The predicted target values.
            list_metrics: A list of metric names to evaluate.

        Returns:
            dict: A dictionary of metric names and their corresponding values.
        """
        if self.task_type == "regression":
            rm = RegressionMetric(y_true=y_true, y_pred=y_pred)
            return rm.get_metrics_by_list_names(list_metrics)
        else:
            cm = ClassificationMetric(y_true, y_pred)
            return cm.get_metrics_by_list_names(list_metrics)

    def scores(self, X, y, list_metrics=("AS", "RS")):
        """
        Computes evaluation metrics for the model's predictions.

        Args:
            X: The feature matrix.
            y: The target vector.
            list_metrics: A list of metric names to evaluate.

        Returns:
            dict: A dictionary of metric names and their corresponding values.
        """
        y_pred = self.predict(X)
        res = self.evaluate(y, y_pred, list_metrics=list_metrics)
        return res

    def save_convergence(self, save_path="history", filename="convergence.csv"):
        """
        Save the convergence (fitness value) during the training process to csv file.

        Parameters
        ----------
        save_path : saved path (relative path, consider from current executed script path)
        filename : name of the file, needs to have ".csv" extension
        """
        Path(save_path).mkdir(parents=True, exist_ok=True)
        if self.loss_train is None:
            print(f"{self.__class__.__name__} network doesn't have training loss!")
        else:
            data = {"epoch": list(range(1, len(self.loss_train) + 1)), "loss": self.loss_train}
            pd.DataFrame(data).to_csv(f"{save_path}/{filename}", index=False)

    def save_performance_metrics(self, y_true, y_pred, list_metrics=("RMSE", "MAE"), save_path="history", filename="metrics.csv"):
        """
        Save evaluation metrics to csv file

        Parameters
        ----------
        y_true : ground truth data
        y_pred : predicted output
        list_metrics : list of evaluation metrics
        save_path : saved path (relative path, consider from current executed script path)
        filename : name of the file, needs to have ".csv" extension
        """
        Path(save_path).mkdir(parents=True, exist_ok=True)
        results = self.best_estimator.evaluate(y_true, y_pred, list_metrics)
        df = pd.DataFrame.from_dict(results, orient='index').T
        df.to_csv(f"{save_path}/{filename}", index=False)

    def save_y_predicted(self, X, y_true, save_path="history", filename="y_predicted.csv"):
        """
        Save the predicted results to csv file

        Parameters
        ----------
        X : The features data, nd.ndarray
        y_true : The ground truth data
        save_path : saved path (relative path, consider from current executed script path)
        filename : name of the file, needs to have ".csv" extension
        """
        Path(save_path).mkdir(parents=True, exist_ok=True)
        y_pred = self.predict(X)
        data = {"y_true": np.squeeze(np.asarray(y_true)), "y_pred": np.squeeze(np.asarray(y_pred))}
        pd.DataFrame(data).to_csv(f"{save_path}/{filename}", index=False)

    def save_model(self, save_path="history", filename="network.pkl"):
        """
        Save network to pickle file

        Parameters
        ----------
        save_path : saved path (relative path, consider from current executed script path)
        filename : name of the file, needs to have ".pkl" extension
        """
        Path(save_path).mkdir(parents=True, exist_ok=True)
        if filename[-4:] != ".pkl":
            filename += ".pkl"
        pickle.dump(self, open(f"{save_path}/{filename}", 'wb'))

    @staticmethod
    def load_model(load_path="history", filename="network.pkl"):
        """
        Load a saved model from a pickle file.

        Parameters
        ----------
        load_path : str, default="history"
            Directory containing the saved file.
        filename : str, default="network.pkl"
            Name of the file (must end with `.pkl`).

        Returns
        -------
        model : BaseRVFL
            Loaded model instance.
        """
        if filename[-4:] != ".pkl":
            filename += ".pkl"
        return pickle.load(open(f"{load_path}/{filename}", 'rb'))
