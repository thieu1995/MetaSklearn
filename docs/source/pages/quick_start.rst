============
Installation
============

* Install the `current PyPI release <https://pypi.python.org/pypi/metasklearn />`_

.. code-block:: bash

    $ pip install metasklearn==0.2.0


* Install directly from source code.

.. code-block:: bash

    $ git clone https://github.com/thieu1995/MetaSklearn.git
    $ cd MetaSklearn
    $ python setup.py install

* In case, you want to install the development version from Github

.. code-block:: bash

   $ pip install git+https://github.com/thieu1995/MetaSklearn


After installation, you can check the version of installed MetaSklearn::

   $ python
   >>> import metasklearn
   >>> metasklearn.__version__

=========
Tutorials
=========

In this section, we will explore the usage of the MetaSklearn model with the assistance of a dataset. While all the
preprocessing steps mentioned below can be replicated using Scikit-Learn, we have implemented some utility functions
to provide users with convenience and faster usage.


Provided classes
----------------

Classes that hold Searcher and Dataset

.. code-block:: python

	from metasklearn import DataTransformer, Data
	from metasklearn import MetaSearchCV


`DataTransformer` class
-----------------------

We provide many scaler classes that you can select and make a combination of transforming your data via
`DataTransformer` class. For example: scale data by `Loge` and then `Sqrt` and then `MinMax`.

.. code-block:: python

	from metasklearn import DataTransformer
	import pandas as pd
	from sklearn.model_selection import train_test_split

	dataset = pd.read_csv('Position_Salaries.csv')
	X = dataset.iloc[:, 1:5].values
	y = dataset.iloc[:, 5].values
	X_train, y_train, X_test, y_test = train_test_split(X, y, test_size=0.2)

	dt = DataTransformer(scaling_methods=("loge", "sqrt", "minmax"))
	X_train_scaled = dt.fit_transform(X_train)
	X_test_scaled = dt.transform(X_test)


`Data` class
------------

+ You can load your dataset into Data class
+ You can split dataset to train and test set
+ You can scale dataset without using DataTransformer class
+ You can scale labels using LabelEncoder

.. code-block:: python

	from metasklearn import Data
	import pandas as pd

	dataset = pd.read_csv('Position_Salaries.csv')
	X = dataset.iloc[:, 1:5].values
	y = dataset.iloc[:, 5].values

	data = Data(X, y, name="position_salaries")

	#### Split dataset into train and test set
	data.split_train_test(test_size=0.2, shuffle=True, random_state=100, inplace=True)

	#### Feature Scaling
	data.X_train, scaler_X = data.scale(data.X_train, scaling_methods=("standard", "sqrt", "minmax"))
	data.X_test = scaler_X.transform(data.X_test)

	data.y_train, scaler_y = data.encode_label(data.y_train)  # This is for classification problem only
	data.y_test = scaler_y.transform(data.y_test)


`Searcher` class
----------------

In this example, we will use `MetaSearchCV` to search for the best hyper-parameters of the SVM model.

.. code-block:: python

    from sklearn.svm import SVC
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split
    from metasklearn import MetaSearchCV, FloatVar, StringVar

    # Load dataset
    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Define param bounds for SVC

    # param_bounds = {          ==> This is for GridSearchCV, show you how to convert to our MetaSearchCV
    #     "C": [0.1, 100],
    #     "gamma": [1e-4, 1],
    #     "kernel": ["linear", "rbf", "poly"]
    # }

    param_bounds = [
        FloatVar(lb=0., ub=100., name="C"),
        FloatVar(lb=1e-4, ub=1., name="gamma"),
        StringVar(valid_sets=("linear", "rbf", "poly"), name="kernel")
    ]

    # Initialize and fit MetaSearchCV
    searcher = MetaSearchCV(
        estimator=SVC(),
        param_bounds=param_bounds,
        task_type="classification",
        optim="BaseGA",     # Using Genetic Algorithm for hyper-parameter optimization
        optim_params={"epoch": 20, "pop_size": 30, "name": "GA"},
        cv=3,
        scoring="AS",  # or any custom scoring like "F1_macro"
        seed=42,
        n_jobs=2,
        verbose=True,
        mode='single', n_workers=None, termination=None
    )

    searcher.fit(X_train, y_train)
    print("Best parameters (Classification):", searcher.best_params)
    print("Best model: ", searcher.best_estimator)
    print("Best score during searching: ", searcher.best_score)

    # Make prediction after re-fit
    y_pred = searcher.predict(X_test)
    print("Test Accuracy:", searcher.score(X_test, y_test))
    print("Test Score: ", searcher.scores(X_test, y_test, list_metrics=("AS", "RS", "PS", "F1S")))


Please check out the examples for more details on how to use the `MetaSearchCV` class.

.. toctree::
   :maxdepth: 4

.. toctree::
   :maxdepth: 4

.. toctree::
   :maxdepth: 4
