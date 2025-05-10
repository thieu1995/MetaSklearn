#!/usr/bin/env python
# Created by "Thieu" at 09:02, 10/05/2025 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

from sklearn.neighbors import KNeighborsRegressor
from sklearn.datasets import load_diabetes
from metasklearn import MetaSearchCV, IntegerVar, StringVar, Data

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
#     'n_neighbors': [2, 3, 5, 7, 9, 11],               # Số lượng hàng xóm gần nhất
#     'weights': ['uniform', 'distance'],              # Trọng số: đều nhau hoặc theo khoảng cách
#     'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],  # Thuật toán tìm kiếm
#     'leaf_size': [10, 20, 30, 40, 50],               # Kích thước node lá, ảnh hưởng đến performance
#     'p': [1, 2],                                     # Tham số cho khoảng cách Minkowski: 1 (Manhattan), 2 (Euclidean)
#     'metric': ['minkowski'],                         # Khoảng cách sử dụng; thường dùng 'minkowski' kết hợp với p
#     # Có thể thêm các metric khác nếu cần như 'euclidean', 'manhattan', 'chebyshev', 'mahalanobis'
# }

param_bounds = [
    IntegerVar(lb=2, ub=12, name="n_neighbors"),
    StringVar(valid_sets=("uniform", "distance"), name="weights"),
    StringVar(valid_sets=("auto", "ball_tree", "kd_tree", "brute"), name="algorithm"),
    IntegerVar(lb=10, ub=50, name="leaf_size"),
    IntegerVar(lb=1, ub=2, name="p"),  # 1 (Manhattan), 2 (Euclidean)
    StringVar(valid_sets=("minkowski", "manhattan"), name="metric"),  # Khoảng cách sử dụng; thường dùng 'minkowski' kết hợp với p
]

# Initialize and fit MetaSearchCV
searcher = MetaSearchCV(
    estimator=KNeighborsRegressor(),
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
