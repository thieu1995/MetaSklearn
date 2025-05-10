#!/usr/bin/env python
# Created by "Thieu" at 09:34, 10/05/2025 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from metasklearn import MetaSearchCV, IntegerVar, StringVar, MixedSetVar, FloatVar

# Load dataset
X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define param bounds

# param_grid = {            # ==> This is for GridSearchCV, show you how to convert to our MetaSearchCV
#     'n_estimators': [100, 200, 500],               # Number of trees in the forest
#     'criterion': ['gini', 'entropy', 'log_loss'],  # Function to measure the quality of a split (for classification)
#     'max_depth': [None, 10, 20, 30],               # Maximum depth of each tree
#     'min_samples_split': [2, 5, 10],               # Minimum number of samples required to split a node
#     'min_samples_leaf': [1, 2, 4],                 # Minimum number of samples required at a leaf node
#     'max_features': [3, 5, 7, 0.2, 0.5, 'sqrt', 'log2'],      # Number of features to consider when looking for the best split
#     'max_samples': [None, 0.5, 0.8],               # If bootstrap=True, this controls the number of samples to draw
#     'ccp_alpha': [0.0, 0.01, 0.05],                # Complexity parameter used for Minimal Cost-Complexity Pruning
# }

param_bounds = [
    IntegerVar(lb=30, ub=100, name="n_estimators"),
    StringVar(valid_sets=("gini", "entropy", "log_loss"), name="criterion"),
    IntegerVar(lb=3, ub=10, name="max_depth"),
    IntegerVar(lb=2, ub=10, name="min_samples_split"),
    IntegerVar(lb=1, ub=5, name="min_samples_leaf"),
    MixedSetVar(valid_sets=("sqrt", "log2", 3, 5, 7, 0.2, 0.5), name="max_features"),
    MixedSetVar(valid_sets=(None, 0.5, 0.8), name="max_samples"),  # If bootstrap=True, this controls the number of samples to draw
    FloatVar(lb=0.0, ub=0.1, name="ccp_alpha"),  # Complexity parameter used for Minimal Cost-Complexity Pruning
]

# Initialize and fit MetaSearchCV
searcher = MetaSearchCV(
    estimator=RandomForestClassifier(random_state=42),
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
print("Best parameters:", searcher.best_params)
print("Best model: ", searcher.best_estimator)
print("Best score during searching: ", searcher.best_score)

# Make prediction after re-fit
y_pred = searcher.predict(X_test)
print("Test Accuracy:", searcher.score(X_test, y_test))
print("Test Score: ", searcher.scores(X_test, y_test, list_metrics=("AS", "RS", "PS", "F1S")))
