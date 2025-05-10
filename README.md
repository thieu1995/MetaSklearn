# MetaSklearn: A Metaheuristic-Powered Hyperparameter Optimization Framework for Scikit-Learn Models.

[![GitHub release](https://img.shields.io/badge/release-0.1.0-yellow.svg)](https://github.com/thieu1995/MetaSklearn/releases)
[![PyPI version](https://badge.fury.io/py/metasklearn.svg)](https://badge.fury.io/py/metasklearn)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/metasklearn.svg)
![PyPI - Downloads](https://img.shields.io/pypi/dm/metasklearn.svg)
[![Downloads](https://pepy.tech/badge/metasklearn)](https://pepy.tech/project/metasklearn)
[![Tests & Publishes to PyPI](https://github.com/thieu1995/MetaSklearn/actions/workflows/publish-package.yaml/badge.svg)](https://github.com/thieu1995/MetaSklearn/actions/workflows/publish-package.yaml)
[![Documentation Status](https://readthedocs.org/projects/metasklearn/badge/?version=latest)](https://metasklearn.readthedocs.io/en/latest/?badge=latest)
[![Chat](https://img.shields.io/badge/Chat-on%20Telegram-blue)](https://t.me/+fRVCJGuGJg1mNDg1)
[![DOI](https://img.shields.io/badge/DOI-10.6084%2Fm9.figshare.28978805-blue)](https://doi.org/10.6084/m9.figshare.28978805)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)


---

## 🌟 Overview

**MetaSklearn** is a flexible and extensible Python library that brings metaheuristic optimization to 
hyperparameter tuning of scikit-learn models. It provides a seamless interface to optimize hyperparameters 
using nature-inspired algorithms from the [Mealpy](https://github.com/thieu1995/mealpy) library.
It is designed to be user-friendly and efficient, making it easy to integrate into your machine learning workflow.


## 🚀 Features

- ✅ Hyperparameter optimization with **metaheuristic algorithms**
- ✅ Compatible with any **scikit-learn** model (SVM, RandomForest, XGBoost, etc.)
- ✅ Supports **classification** and **regression** tasks
- ✅ Custom and scikit-learn scoring support
- ✅ Integration with [`PerMetrics`](https://github.com/thieu1995/permetrics) for rich evaluation metrics
- ✅ Scikit-learn compatible API: `.fit()`, `.predict()`, `.score()`


## 📦 Installation

You can install the library using `pip` (once published to PyPI):

```bash
pip install metasklearn
```

After installation, you can import `MetaSklearn` as any other Python module:

```sh
$ python
>>> import metasklearn
>>> metasklearn.__version__
```

## 🧠 How It Works

`MetaSklearn` defines a custom `MetaSearchCV` class that wraps your model and performs hyperparameter tuning using 
any optimizer supported by Mealpy. The framework evaluates model performance using either 
scikit-learn’s metrics or additional ones from `PerMetrics` library.


## 🚀 Quick Start

#### 📘 Example with SVM model for regression task


```python
from sklearn.svm import SVR
from sklearn.datasets import load_diabetes
from metasklearn import MetaSearchCV, FloatVar, StringVar, Data

## Load data object
X, y = load_diabetes(return_X_y=True)
data = Data(X, y)

## Split train and test
data.split_train_test(test_size=0.2, random_state=42, inplace=True)
print(data.X_train.shape, data.X_test.shape)

## Scaling dataset
data.X_train, scaler_X = data.scale(data.X_train, scaling_methods=("standard", "minmax"))
data.X_test = scaler_X.transform(data.X_test)

data.y_train, scaler_y = data.scale(data.y_train, scaling_methods=("standard", "minmax"))
data.y_train = data.y_train.ravel()
data.y_test = scaler_y.transform(data.y_test.reshape(-1, 1)).ravel()

# Define param bounds for SVC

# param_bounds = {          ==> This is for GridSearchCV, show you how to convert to our MetaSearchCV
#     "C": [0.1, 100],
#     "gamma": [1e-4, 1],
#     "kernel": ["linear", "rbf", "poly"]
# }

param_bounds = [
    FloatVar(lb=0., ub=100., name="C"),
    FloatVar(lb=1e-4, ub=1., name="gamma"),
    StringVar(valid_sets=("linear", "rbf", "poly"), name="kernel")
]

# Initialize and fit MetaSearchCV
searcher = MetaSearchCV(
    estimator=SVR(),
    param_bounds=param_bounds,
    task_type="regression",
    optim="BaseGA",
    optim_params={"epoch": 20, "pop_size": 30, "name": "GA"},
    cv=3,
    scoring="MSE",  # or any custom scoring like "F1_macro"
    seed=42,
    n_jobs=2,
    verbose=True
)

searcher.fit(data.X_train, data.y_train)
print("Best parameters (Classification):", searcher.best_params)
print("Best model: ", searcher.best_estimator)
print("Best score during searching: ", searcher.best_score)

# Make prediction after re-fit
y_pred = searcher.predict(data.X_test)
print("Test Accuracy:", searcher.score(data.X_test, data.y_test))
print("Test Score: ", searcher.scores(data.X_test, data.y_test, list_metrics=("RMSE", "R", "KGE", "NNSE")))
```

#### 📘 Example with SVM model for classification task

```python
from sklearn.svm import SVC
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from metasklearn import MetaSearchCV, FloatVar, StringVar

# Load dataset
X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define param bounds for SVC

# param_bounds = {          ==> This is for GridSearchCV, show you how to convert to our MetaSearchCV
#     "C": [0.1, 100],
#     "gamma": [1e-4, 1],
#     "kernel": ["linear", "rbf", "poly"]
# }

param_bounds = [
    FloatVar(lb=0., ub=100., name="C"),
    FloatVar(lb=1e-4, ub=1., name="gamma"),
    StringVar(valid_sets=("linear", "rbf", "poly"), name="kernel")
]

# Initialize and fit MetaSearchCV
searcher = MetaSearchCV(
    estimator=SVC(),
    param_bounds=param_bounds,
    task_type="classification",
    optim="BaseGA",
    optim_params={"epoch": 20, "pop_size": 30, "name": "GA"},
    cv=3,
    scoring="AS",  # or any custom scoring like "F1_macro"
    seed=42,
    n_jobs=2,
    verbose=True
)

searcher.fit(X_train, y_train)
print("Best parameters (Classification):", searcher.best_params)
print("Best model: ", searcher.best_estimator)
print("Best score during searching: ", searcher.best_score)

# Make prediction after re-fit
y_pred = searcher.predict(X_test)
print("Test Accuracy:", searcher.score(X_test, y_test))
print("Test Score: ", searcher.scores(X_test, y_test, list_metrics=("AS", "RS", "PS", "F1S")))
```

As can be seen, you do it like any other model from Scikit-Learn library such as Random Forest, Decision Tree, XGBoost,...

## 📋 Parameters - Variable Types in MetaSearchCV. How to choose them?

This section explains how to use different types of variables from the `MetaSearchCV` library when defining hyperparameter 
search spaces. Each variable type is suitable for different kinds of optimization parameters.

#### 1. `IntegerVar` – Integer Variable
```python
from metasklearn import IntegerVar

var = IntegerVar(lb=1, ub=100, name="n_estimators")
```
Used for discrete numerical parameters like number of neighbors in KNN, number of estimators in ensembles, etc.

#### 2. `FloatVar` – Float/Continuous Variable
```python
from metasklearn import FloatVar

var = FloatVar(lb=0.001, ub=1.0, name="learning_rate")
```
Used for continuous numerical parameters such as `learning_rate`, `C`, `gamma`, etc.

#### 3. `StringVar` – Categorical/String Variable
```python
from metasklearn import StringVar

var = StringVar(valid_sets=("linear", "poly", "rbf"), name="kernel")
```
Used for categorical parameters with a limited number of choices, e.g., `kernel` in SVM.

#### 4. `BinaryVar` – Binary Variable (0 or 1)
```python
from metasklearn import BinaryVar

var = BinaryVar(n_vars=1, name="feature_selected")
```
Used in binary feature selection problems or any 0/1-based decision.

#### 5. `BoolVar` – Boolean Variable (True or False)
```python
from metasklearn import BoolVar

var = BoolVar(n_vars=1, name="use_bias")
```
Used for Boolean-type arguments such as `fit_intercept`, `use_bias`, etc.

#### 6. `PermutationVar` – Permutation Variable
```python
from metasklearn import PermutationVar

var = PermutationVar(valid_set=(1, 2, 5, 10), name="job_order")
```
Used for optimization problems involving permutations, like scheduling or routing.

#### 7. `MixedSetVar` – Mixed-Type Set Variable
```python
from metasklearn import MixedSetVar

var = MixedSetVar(valid_sets=((10, 15), (20, 10), (30, 5)), name="mixed_choice")
```
Used when the parameter can take values of different types (int, string, bool, float, set,...).

#### 8. `TransferBinaryVar` – Transfer Binary Variable
```python
from metasklearn import TransferBinaryVar

var = TransferBinaryVar(n_vars=1, tf_func="vstf_01", lb=-8., ub=8., all_zeros=True, name="transfer_binary")
```
Used in binary search spaces that support transformation-based metaheuristics.

#### 9. `TransferBoolVar` – Transfer Boolean Variable
```python
from metasklearn import TransferBoolVar

var = TransferBoolVar(n_vars=1, tf_func="vstf_01", lb=-8., ub=8., name="transfer_bool")
```
Used in Boolean search spaces with transferable logic between states.

#### 🔧 Example: Define a Mixed Search Space

```python
from metasklearn import (IntegerVar, FloatVar, StringVar, BinaryVar, BoolVar, 
        PermutationVar, MixedSetVar, TransferBinaryVar, TransferBoolVar)

param_bounds = [
    IntegerVar(lb=1, ub=20, name="n_neighbors"),
    FloatVar(lb=0.001, ub=1.0, name="alpha"),
    StringVar(valid_sets=["uniform", "distance"], name="weights"),
    BinaryVar(name="use_feature"),
    BoolVar(name="fit_bias"),
    PermutationVar(valid_set=(1, 2, 5, 10), name="job_order"),
    MixedSetVar(valid_sets=[0.1, "relu", False], name="activation_choice"),
    MixedSetVar(valid_sets=((10, 15), (20, 10), (30, 5)), name="mixed_choice"),
    TransferBinaryVar(name="bin_transfer"),
    TransferBoolVar(name="bool_transfer")
]
```
Use this format when designing hyperparameter spaces for advanced models in `MetaSearchCV`.


## ⚙ Supported Optimizers

`MetaSklearn` integrates all metaheuristic algorithms from Mealpy, including:

+ AOA (Arithmetic Optimization Algorithm)
+ GWO (Grey Wolf Optimizer)
+ PSO (Particle Swarm Optimization)
+ DE (Differential Evolution)
+ WOA, SSA, MVO, and many more...

You can pass any optimizer name or an instantiated optimizer object to MetaSearchCV. For more details, please refer to the [link](https://mealpy.readthedocs.io/en/latest/pages/support.html#classification-table)


## 📊 Custom Metrics
You can use custom scoring functions from:

+ sklearn.metrics.get_scorer_names()

+ permetrics.RegressionMetric and ClassificationMetric

For details on `PerMetrics` library, please refer to the [link](https://permetrics.readthedocs.io/en/latest/pages/support.html#all-performance-metrics)


## 📚 Documentation

Documentation is available at: 👉 https://metasklearn.readthedocs.io

You can build the documentation locally:

```shell
cd docs
make html
```

## 🧪 Testing
You can run unit tests using:

```shell
pytest tests/
```

## 🤝 Contributing
We welcome contributions to `MetaSklearn`! If you have suggestions, improvements, or bug fixes, feel free to fork 
the repository, create a pull request, or open an issue.


## 📄 License
This project is licensed under the GPLv3 License. See the LICENSE file for more details.


## Citation Request
Please include these citations if you plan to use this library:

```bibtex
@software{thieu20250510MetaSklearn,
  author       = {Nguyen Van Thieu},
  title        = {MetaSklearn: A Metaheuristic-Powered Hyperparameter Optimization Framework for Scikit-Learn Models},
  month        = may,
  year         = 2025,
  doi         = {10.6084/m9.figshare.28978805},
  url          = {https://github.com/thieu1995/MetaSklearn}
}
```

## Official Links 

* Official source code repo: https://github.com/thieu1995/MetaSklearn
* Official document: https://metasklearn.readthedocs.io/
* Download releases: https://pypi.org/project/metasklearn/
* Issue tracker: https://github.com/thieu1995/MetaSklearn/issues
* Notable changes log: https://github.com/thieu1995/MetaSklearn/blob/master/ChangeLog.md
* Official chat group: https://t.me/+fRVCJGuGJg1mNDg1

---

Developed by: [Thieu](mailto:nguyenthieu2102@gmail.com?Subject=MetaSklearn_QUESTIONS) @ 2025
