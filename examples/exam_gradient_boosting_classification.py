#!/usr/bin/env python
# Created by "Thieu" at 10:10, 10/05/2025 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from metasklearn import MetaSearchCV, IntegerVar, StringVar, FloatVar

# Load dataset
X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define param bounds

# param_grid = {            ==> This is for GridSearchCV, show you how to convert to our MetaSearchCV
param_grid = {
    'n_estimators': [50, 75, 100],                 # Number of boosting stages (trees)
    'learning_rate': [0.01, 0.05, 0.1, 0.2],         # Shrinks the contribution of each tree
    'max_depth': [3, 5, 7],                          # Maximum depth of individual trees
    'min_samples_split': [2, 5, 10],                 # Minimum number of samples to split a node
    'min_samples_leaf': [1, 2, 4],                   # Minimum samples required at a leaf node
    'max_features': ['auto', 'sqrt', 'log2'],        # Number of features to consider when looking for the best split
    'subsample': [1.0, 0.8, 0.6],                    # Fraction of samples to be used for fitting the individual base learners
    'loss': ['log_loss'],             # Classification loss functions
    'ccp_alpha': [0.0, 0.01, 0.1],                   # Complexity parameter for Minimal Cost-Complexity Pruning
}

param_bounds = [
    IntegerVar(lb=50, ub=200, name="n_estimators"),
    FloatVar(lb=0.01, ub=0.2, name="learning_rate"),
    IntegerVar(lb=2, ub=5, name="max_depth"),
    IntegerVar(lb=2, ub=10, name="min_samples_split"),
    IntegerVar(lb=1, ub=5, name="min_samples_leaf"),
    StringVar(valid_sets=("sqrt", "log2"), name="max_features"),
    FloatVar(lb=0.5, ub=1.0, name="subsample"),
    StringVar(valid_sets=(("log_loss",)), name="loss"),
    FloatVar(lb=0.0, ub=0.1, name="ccp_alpha"),
]

# Initialize and fit MetaSearchCV
searcher = MetaSearchCV(
    estimator=GradientBoostingClassifier(random_state=42),
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
