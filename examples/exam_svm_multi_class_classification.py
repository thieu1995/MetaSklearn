#!/usr/bin/env python
# Created by "Thieu" at 09:46, 08/05/2025 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

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
