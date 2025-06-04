#!/usr/bin/env python
# Created by "Thieu" at 09:09, 10/05/2025 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from metasklearn import MetaSearchCV, IntegerVar, StringVar

# Load dataset
X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define param bounds

# param_grid = {            ==> This is for GridSearchCV, show you how to convert to our MetaSearchCV
#     'n_neighbors': [2, 3, 5, 7, 9, 11],
#     'weights': ['uniform', 'distance'],
#     'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
#     'leaf_size': [10, 20, 30, 40, 50],
#     'p': [1, 2],
#     'metric': ['minkowski'],
# }

param_bounds = [
    IntegerVar(lb=2, ub=12, name="n_neighbors"),
    StringVar(valid_sets=("uniform", "distance"), name="weights"),
    StringVar(valid_sets=("auto", "ball_tree", "kd_tree", "brute"), name="algorithm"),
    IntegerVar(lb=10, ub=50, name="leaf_size"),
    IntegerVar(lb=1, ub=2, name="p"),  # 1 (Manhattan), 2 (Euclidean)
    StringVar(valid_sets=("minkowski", "manhattan"), name="metric"),
]

# Initialize and fit MetaSearchCV
searcher = MetaSearchCV(
    estimator=KNeighborsClassifier(),
    param_bounds=param_bounds,
    task_type="classification",
    optim="BaseGA",
    optim_params={"epoch": 20, "pop_size": 30, "name": "GA"},
    cv=3,
    scoring="AS",  # or any custom scoring like "F1_macro"
    seed=42,
    n_jobs=2,
    verbose=True,
    mode='single', n_workers=None, termination=None
)

searcher.fit(X_train, y_train)
print("Best parameters:", searcher.best_params)
print("Best model: ", searcher.best_estimator)
print("Best score during searching: ", searcher.best_score)

# Make prediction after re-fit
y_pred = searcher.predict(X_test)
print("Test Accuracy:", searcher.score(X_test, y_test))
print("Test Score: ", searcher.scores(X_test, y_test, list_metrics=("AS", "RS", "PS", "F1S")))
