.. MetaSklearn documentation master file, created by
   sphinx-quickstart on Sat May 20 16:59:33 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to MetaSklearn's documentation!
=======================================

.. image:: https://img.shields.io/badge/release-0.1.0-yellow.svg
   :target: https://github.com/thieu1995/MetaSklearn/releases

.. image:: https://badge.fury.io/py/metasklearn.svg
   :target: https://badge.fury.io/py/metasklearn

.. image:: https://img.shields.io/pypi/pyversions/metasklearn.svg
   :target: https://www.python.org/

.. image:: https://img.shields.io/pypi/dm/metasklearn.svg
   :target: https://img.shields.io/pypi/dm/metasklearn.svg

.. image:: https://github.com/thieu1995/MetaSklearn/actions/workflows/publish-package.yaml/badge.svg
   :target: https://github.com/thieu1995/MetaSklearn/actions/workflows/publish-package.yaml

.. image:: https://pepy.tech/badge/metasklearn
   :target: https://pepy.tech/project/metasklearn

.. image:: https://readthedocs.org/projects/metasklearn/badge/?version=latest
   :target: https://metasklearn.readthedocs.io/en/latest/?badge=latest

.. image:: https://img.shields.io/badge/Chat-on%20Telegram-blue
   :target: https://t.me/+fRVCJGuGJg1mNDg1

.. image:: https://img.shields.io/badge/DOI-10.6084%2Fm9.figshare.28978805-blue
   :target: https://doi.org/10.6084/m9.figshare.28978805

.. image:: https://img.shields.io/badge/License-GPLv3-blue.svg
   :target: https://www.gnu.org/licenses/gpl-3.0


**MetaSklearn** is a flexible and extensible Python library that brings metaheuristic optimization to
hyperparameter tuning of scikit-learn models. It provides a seamless interface to optimize hyperparameters
using nature-inspired algorithms from the [Mealpy](https://github.com/thieu1995/mealpy) library.
It is designed to be user-friendly and efficient, making it easy to integrate into your machine learning workflow.

* **Free software:** GNU General Public License (GPL) V3 license
* **Provided Searcher**: `MetaSearchCV`
* **Total Metaheuristic-based Scikit-Learn Regressor**: > 200 Models
* **Total Metaheuristic-based Scikit-Learn Classifier**: > 200 Models
* **Supported performance metrics**: >= 67 (47 regressions and 20 classifications)
* **Supported objective functions (as fitness functions or loss functions)**: >= 67 (47 regressions and 20 classifications)
* **Documentation:** https://metasklearn.readthedocs.io
* **Python versions:** >= 3.8.x
* **Dependencies:** numpy, scipy, scikit-learn, pandas, mealpy, permetrics, matplotlib


.. toctree::
   :maxdepth: 4
   :caption: Quick Start:

   pages/quick_start.rst

.. toctree::
   :maxdepth: 4
   :caption: Models API:

   pages/metasklearn.rst

.. toctree::
   :maxdepth: 4
   :caption: Support:

   pages/support.rst


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
