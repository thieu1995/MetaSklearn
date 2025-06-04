#!/usr/bin/env python
# Created by "Thieu" at 10:00, 10/05/2025 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from metasklearn import MetaSearchCV, IntegerVar, FloatVar

# Load dataset
X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define param bounds

# param_grid = {            ==> This is for GridSearchCV, based on this, you can convert to our MetaSearchCV bounds
#     'n_estimators': [50, 100, 200],                  # Number of weak learners to use
#     'learning_rate': [0.01, 0.1, 0.5, 1.0],          # Weight applied to each classifier at each boosting iteration
#     'base_estimator__max_depth': [1, 3, 5],          # Depth of the base decision tree
#     'base_estimator__min_samples_split': [2, 5],     # Minimum samples to split a node in the base estimator
#     'base_estimator__min_samples_leaf': [1, 2],      # Minimum samples at a leaf node in the base estimator
# }

param_bounds = [
    IntegerVar(lb=20, ub=100, name="n_estimators"),
    FloatVar(lb=0.01, ub=1.0, name="learning_rate"),
    IntegerVar(lb=2, ub=5, name="estimator__max_depth"),
    IntegerVar(lb=2, ub=6, name="estimator__min_samples_split"),
    IntegerVar(lb=1, ub=4, name="estimator__min_samples_leaf"),
]

# Initialize and fit MetaSearchCV
searcher = MetaSearchCV(
    estimator=AdaBoostClassifier(estimator=DecisionTreeClassifier(random_state=42), random_state=42),
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
