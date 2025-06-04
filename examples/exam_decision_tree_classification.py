#!/usr/bin/env python
# Created by "Thieu" at 09:17, 10/05/2025 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from metasklearn import MetaSearchCV, IntegerVar, StringVar, FloatVar

# Load dataset
X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define param bounds

# param_grid = {            ==> This is for GridSearchCV, base on this, you can convert to our MetaSearchCV bounds
#     'criterion': ['gini', 'entropy', 'log_loss'],
#     'splitter': ['best', 'random'],
#     'max_depth': [None, 5, 10, 20, 30],
#     'min_samples_split': [2, 5, 10],
#     'min_samples_leaf': [1, 2, 4],
#     'max_features': [None, 'sqrt', 'log2'],
#     'max_leaf_nodes': [None, 10, 20, 50],
#     'ccp_alpha': [0.0, 0.01, 0.05, 0.1],
# }

param_bounds = [
    StringVar(valid_sets=("gini", "entropy", "log_loss"), name="criterion"),
    StringVar(valid_sets=("best", "random"), name="splitter"),
    IntegerVar(lb=2, ub=15, name="max_depth"),
    IntegerVar(lb=2, ub=10, name="min_samples_split"),
    IntegerVar(lb=1, ub=5, name="min_samples_leaf"),
    StringVar(valid_sets=("sqrt", "log2"), name="max_features"),
    IntegerVar(lb=2, ub=30, name="max_leaf_nodes"),
    FloatVar(lb=0.0, ub=0.1, name="ccp_alpha"),
]

# Initialize and fit MetaSearchCV
searcher = MetaSearchCV(
    estimator=DecisionTreeClassifier(random_state=42),
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
