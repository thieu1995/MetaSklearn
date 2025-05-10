#!/usr/bin/env python
# Created by "Thieu" at 09:12, 10/05/2025 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

from sklearn.tree import DecisionTreeRegressor
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
data.y_train = data.y_train.ravel()
data.y_test = scaler_y.transform(data.y_test.reshape(-1, 1)).ravel()

# Define param bounds

# param_grid = {            ==> This is for GridSearchCV, show you how to convert to our MetaSearchCV
#     'criterion': ['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],  # (regression)
#     'splitter': ['best', 'random'],                     # Cách chọn cách chia tại mỗi node
#     'max_depth': [None, 5, 10, 20, 30],                 # Độ sâu tối đa của cây
#     'min_samples_split': [2, 5, 10],                    # Số lượng mẫu tối thiểu để một node được chia
#     'min_samples_leaf': [1, 2, 4],                      # Số lượng mẫu tối thiểu tại một node lá
#     'max_features': [None, 'sqrt', 'log2'],             # Số đặc trưng tối đa khi tìm split tốt nhất
#     'max_leaf_nodes': [None, 10, 20, 50],               # Giới hạn số lượng node lá
#     'ccp_alpha': [0.0, 0.01, 0.05, 0.1],                 # Complexity parameter để cắt tỉa cây (post-pruning)
# }

param_bounds = [
    StringVar(valid_sets=("squared_error", "friedman_mse", "absolute_error", "poisson"), name="criterion"),  # (regression)
    StringVar(valid_sets=("best", "random"), name="splitter"),  # Cách chọn cách chia tại mỗi node
    IntegerVar(lb=2, ub=15, name="max_depth"),  # Độ sâu tối đa của cây
    IntegerVar(lb=2, ub=10, name="min_samples_split"),  # Số lượng mẫu tối thiểu để một node được chia
    IntegerVar(lb=1, ub=5, name="min_samples_leaf"),  # Số lượng mẫu tối thiểu tại một node lá
    StringVar(valid_sets=("sqrt", "log2"), name="max_features"),  # Số đặc trưng tối đa khi tìm split tốt nhất
    IntegerVar(lb=2, ub=30, name="max_leaf_nodes"),  # Giới hạn số lượng node lá
    FloatVar(lb=0.0, ub=0.1, name="ccp_alpha"),  # Complexity parameter để cắt tỉa cây (post-pruning)
]

# Initialize and fit MetaSearchCV
searcher = MetaSearchCV(
    estimator=DecisionTreeRegressor(random_state=42),
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
