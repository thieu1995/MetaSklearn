#!/usr/bin/env python
# Created by "Thieu" at 23:52, 09/05/2025 ----------%
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import pytest
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from permetrics import ClassificationMetric
from metasklearn import IntegerVar
from metasklearn.core.problem import HyperparameterProblem


@pytest.fixture
def sample_data():
    X, y = make_classification(n_samples=100, n_features=10, random_state=42)
    return train_test_split(X, y, test_size=0.2, random_state=42)


@pytest.fixture
def problem_instance(sample_data):
    X_train, X_test, y_train, y_test = sample_data
    estimator = RandomForestClassifier(random_state=42)
    metric_class = ClassificationMetric
    bounds = [
        IntegerVar(lb=10, ub=100, name="n_estimators"),
        IntegerVar(lb=1, ub=5, name="max_depth"),
        IntegerVar(lb=1, ub=5, name="min_samples_leaf"),
    ]
    return HyperparameterProblem(bounds=bounds, X=X_train, y=y_train, estimator=estimator,
                                 metric_class=metric_class, obj_name="AS", sklearn_score=False, cv=3)


def test_initialization(problem_instance):
    assert problem_instance.estimator is not None
    assert problem_instance.X is not None
    assert problem_instance.y is not None
    assert problem_instance.cv == 3
    assert problem_instance.shuffle is True


def test_get_custom_score(problem_instance):
    score = problem_instance._get_custom_score()
    assert isinstance(score, float)
    assert 0.0 <= score <= 1.0


def test_obj_func(problem_instance):
    x = [[3], [2], [2]]
    encoded_x = problem_instance.encode_solution(x)
    score = problem_instance.obj_func(encoded_x)
    assert isinstance(score, float)
    assert 0.0 <= score <= 1.0
