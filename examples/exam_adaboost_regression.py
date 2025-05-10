#!/usr/bin/env python
# Created by "Thieu" at 09:46, 10/05/2025 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.datasets import load_diabetes
from metasklearn import MetaSearchCV, IntegerVar, FloatVar, Data

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
#     'n_estimators': [50, 100, 200],                  # Number of weak learners to use
#     'learning_rate': [0.01, 0.1, 0.5, 1.0],          # Weight applied to each classifier at each boosting iteration
#     'base_estimator__max_depth': [1, 3, 5],          # Depth of the base decision tree
#     'base_estimator__min_samples_split': [2, 5],     # Minimum samples to split a node in the base estimator
#     'base_estimator__min_samples_leaf': [1, 2],      # Minimum samples at a leaf node in the base estimator
# }

param_bounds = [
    IntegerVar(lb=20, ub=100, name="n_estimators"),  # Số lượng weak learners
    FloatVar(lb=0.01, ub=1.0, name="learning_rate"),  # Tốc độ học
    IntegerVar(lb=2, ub=5, name="estimator__max_depth"),  # Độ sâu của cây quyết định cơ sở
    IntegerVar(lb=2, ub=6, name="estimator__min_samples_split"),  # Số lượng mẫu tối thiểu để một node được chia
    IntegerVar(lb=1, ub=4, name="estimator__min_samples_leaf"),  # Số lượng mẫu tối thiểu tại một node lá
]

# Initialize and fit MetaSearchCV
searcher = MetaSearchCV(
    estimator=AdaBoostRegressor(estimator=DecisionTreeRegressor(random_state=42), random_state=42),
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
