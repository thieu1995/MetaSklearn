#!/usr/bin/env python
# Created by "Thieu" at 00:03, 10/05/2025 ----------%
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import pytest
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from metasklearn.core.search import MetaSearchCV
from metasklearn import IntegerVar


@pytest.fixture
def sample_data():
    X, y = make_classification(n_samples=100, n_features=10, random_state=42)
    return X, y


@pytest.fixture
def meta_search_instance(sample_data):
    X, y = sample_data
    param_bounds = [
        IntegerVar(lb=10, ub=20, name="n_estimators"),
        IntegerVar(lb=1, ub=5, name="max_depth"),
    ]
    estimator = RandomForestClassifier(random_state=42)
    searcher = MetaSearchCV(
        estimator=estimator,
        param_bounds=param_bounds,
        task_type="classification",
        optim="BaseGA",
        optim_params={"epoch": 10, "pop_size": 20, "name": "GA"},
        cv=3,
        scoring="accuracy",
        seed=42,
        n_jobs=2,
        verbose=True
    )
    return searcher


def test_initialization(meta_search_instance):
    assert meta_search_instance.estimator is not None
    assert meta_search_instance.param_bounds is not None
    assert meta_search_instance.task_type == "classification"
    assert meta_search_instance.scoring_name == "accuracy"


def test_fit(meta_search_instance, sample_data):
    X, y = sample_data
    meta_search_instance.fit(X, y)
    assert meta_search_instance.best_params is not None
    assert meta_search_instance.best_estimator is not None


def test_predict(meta_search_instance, sample_data):
    X, y = sample_data
    meta_search_instance.fit(X, y)
    predictions = meta_search_instance.predict(X)
    assert predictions is not None
    assert len(predictions) == len(y)


def test_score(meta_search_instance, sample_data):
    X, y = sample_data
    meta_search_instance.fit(X, y)
    score = meta_search_instance.score(X, y)
    assert isinstance(score, float)
    assert 0.0 <= score <= 1.0


def test_save_and_load_model(meta_search_instance, sample_data, tmp_path):
    X, y = sample_data
    meta_search_instance.fit(X, y)
    save_path = tmp_path / "model"
    save_path.mkdir()
    meta_search_instance.save_model(save_path=str(save_path), filename="test_model.pkl")

    loaded_model = MetaSearchCV.load_model(load_path=str(save_path), filename="test_model.pkl")
    assert loaded_model is not None
    assert loaded_model.best_params == meta_search_instance.best_params
