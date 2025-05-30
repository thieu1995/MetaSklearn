#!/usr/bin/env python
# Created by "Thieu" at 10:30, 10/05/2025 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

from sklearn.neural_network import MLPClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from metasklearn import MetaSearchCV, IntegerVar, FloatVar, CategoricalVar, StringVar, SequenceVar

# Load dataset
X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define param bounds

# param_grid = {            ==> This is for GridSearchCV, show you how to convert to our MetaSearchCV
#     'hidden_layer_sizes': [(50,), (100,), (100, 50), (50, 50, 50)],  # Architecture: number of neurons per layer
#     'activation': ['relu', 'tanh', 'logistic'],                     # Activation functions: ReLU, tanh, sigmoid
#     'solver': ['adam', 'sgd', 'lbfgs'],                             # Optimizers: Adam, SGD, or LBFGS
#     'alpha': [0.0001, 0.001, 0.01],                                  # L2 regularization (penalty term)
#     'learning_rate': ['constant', 'adaptive', 'invscaling'],        # Learning rate schedule
#     'learning_rate_init': [0.001, 0.01, 0.1],                        # Initial learning rate
#     'max_iter': [300, 500, 1000],                                    # Maximum number of iterations (epochs)
#     'batch_size': ['auto', 32, 64],                                  # Batch size for training
# }

param_bounds = [
    SequenceVar(valid_sets=((30, ), (20, 5), (15, 10), (15, 30, 10)), name="hidden_layer_sizes"),
    StringVar(valid_sets=("relu", "tanh", "logistic"), name="activation"),
    StringVar(valid_sets=("adam", "sgd", "lbfgs"), name="solver"),
    FloatVar(lb=0.0001, ub=0.01, name="alpha"),
    StringVar(valid_sets=("constant", "adaptive", "invscaling"), name="learning_rate"),
    FloatVar(lb=0.001, ub=0.1, name="learning_rate_init"),
    IntegerVar(lb=500, ub=600, name="max_iter"),
    CategoricalVar(valid_sets=('auto', 32, 64), name="batch_size"),
]


# Initialize and fit MetaSearchCV
searcher = MetaSearchCV(
    estimator=MLPClassifier(random_state=42),
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
