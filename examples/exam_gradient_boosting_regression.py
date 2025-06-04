#!/usr/bin/env python
# Created by "Thieu" at 10:05, 10/05/2025 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.datasets import load_diabetes
from metasklearn import MetaSearchCV, IntegerVar, StringVar, FloatVar, Data

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
data.y_test = scaler_y.transform(data.y_test)

# Define param bounds

# param_grid = {            ==> This is for GridSearchCV, show you how to convert to our MetaSearchCV
#     'n_estimators': [50, 75, 100],                 # Number of boosting stages (trees)
#     'learning_rate': [0.01, 0.05, 0.1, 0.2],         # Shrinks the contribution of each tree
#     'max_depth': [3, 5, 7],                          # Maximum depth of individual trees
#     'min_samples_split': [2, 5, 10],                 # Minimum number of samples to split a node
#     'min_samples_leaf': [1, 2, 4],                   # Minimum samples required at a leaf node
#     'max_features': ['auto', 'sqrt', 'log2'],        # Number of features to consider when looking for the best split
#     'subsample': [1.0, 0.8, 0.6],                    # Fraction of samples to be used for fitting the individual base learners
#     'loss': ['squared_error', 'absolute_error', 'huber'],  # Regression loss functions
#     'ccp_alpha': [0.0, 0.01, 0.1],                   # Complexity parameter for Minimal Cost-Complexity Pruning
# }

param_bounds = [
    IntegerVar(lb=50, ub=200, name="n_estimators"),
    FloatVar(lb=0.01, ub=0.2, name="learning_rate"),
    IntegerVar(lb=2, ub=5, name="max_depth"),
    IntegerVar(lb=2, ub=10, name="min_samples_split"),
    IntegerVar(lb=1, ub=5, name="min_samples_leaf"),
    StringVar(valid_sets=("sqrt", "log2"), name="max_features"),
    FloatVar(lb=0.5, ub=1.0, name="subsample"),
    StringVar(valid_sets=("squared_error", "absolute_error", 'huber'), name="loss"),
    FloatVar(lb=0.0, ub=0.1, name="ccp_alpha"),
]

# Initialize and fit MetaSearchCV
searcher = MetaSearchCV(
    estimator=GradientBoostingRegressor(random_state=42),
    param_bounds=param_bounds,
    task_type="regression",
    optim="BaseGA",
    optim_params={"epoch": 20, "pop_size": 30, "name": "GA"},
    cv=3,
    scoring="MSE",  # or any custom scoring like "F1_macro"
    seed=42,
    n_jobs=2,
    verbose=True,
    mode='single', n_workers=None, termination=None
)

searcher.fit(data.X_train, data.y_train)
print("Best parameters:", searcher.best_params)
print("Best model: ", searcher.best_estimator)
print("Best score during searching: ", searcher.best_score)

# Make prediction after re-fit
y_pred = searcher.predict(data.X_test)
print("Test R2:", searcher.score(data.X_test, data.y_test))
print("Test Score: ", searcher.scores(data.X_test, data.y_test, list_metrics=("RMSE", "R", "KGE", "NNSE")))
