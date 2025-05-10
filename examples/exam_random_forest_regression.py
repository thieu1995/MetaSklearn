#!/usr/bin/env python
# Created by "Thieu" at 09:21, 10/05/2025 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import load_diabetes
from metasklearn import MetaSearchCV, IntegerVar, StringVar, MixedSetVar, FloatVar, Data

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

# Define param bounds

# param_grid = {            ==> This is for GridSearchCV, show you how to convert to our MetaSearchCV
#     'n_estimators': [100, 200, 500],               # Number of trees in the forest
#     'criterion': ['squared_error', 'absolute_error', 'friedman_mse', 'poisson'],  # For regression
#     'max_depth': [None, 10, 20, 30],               # Maximum depth of each tree
#     'min_samples_split': [2, 5, 10],               # Minimum number of samples required to split a node
#     'min_samples_leaf': [1, 2, 4],                 # Minimum number of samples required at a leaf node
#     'max_features': [3, 5, 7, 0.2, 0.5, 'sqrt', 'log2'],      # Number of features to consider when looking for the best split
#     'max_samples': [None, 0.5, 0.8],               # If bootstrap=True, this controls the number of samples to draw
#     'ccp_alpha': [0.0, 0.01, 0.05],                # Complexity parameter used for Minimal Cost-Complexity Pruning
# }

param_bounds = [
    IntegerVar(lb=30, ub=100, name="n_estimators"),
    StringVar(valid_sets=("squared_error", "absolute_error", "friedman_mse", "poisson"), name="criterion"),
    IntegerVar(lb=3, ub=10, name="max_depth"),
    IntegerVar(lb=2, ub=10, name="min_samples_split"),
    IntegerVar(lb=1, ub=5, name="min_samples_leaf"),
    MixedSetVar(valid_sets=("sqrt", "log2", 3, 5, 7, 0.2, 0.5), name="max_features"),
    MixedSetVar(valid_sets=(None, 0.5, 0.8), name="max_samples"),  # If bootstrap=True, this controls the number of samples to draw
    FloatVar(lb=0.0, ub=0.1, name="ccp_alpha"),  # Complexity parameter used for Minimal Cost-Complexity Pruning
]

# Initialize and fit MetaSearchCV
searcher = MetaSearchCV(
    estimator=RandomForestRegressor(random_state=42),
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
print("Best parameters:", searcher.best_params)
print("Best model: ", searcher.best_estimator)
print("Best score during searching: ", searcher.best_score)

# Make prediction after re-fit
y_pred = searcher.predict(data.X_test)
print("Test R2:", searcher.score(data.X_test, data.y_test))
print("Test Score: ", searcher.scores(data.X_test, data.y_test, list_metrics=("RMSE", "R", "KGE", "NNSE")))
