#!/usr/bin/env python
# Created by "Thieu" at 13:19, 09/05/2025 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

from setuptools import setup, find_packages


def readme():
    with open('README.md', encoding='utf-8') as f:
        README = f.read()
    return README


setup(
    name="metasklearn",
    version="0.2.0",
    author="Thieu",
    author_email="nguyenthieu2102@gmail.com",
    description="MetaSklearn: A Metaheuristic-Powered Hyperparameter Optimization Framework for Scikit-Learn Models",
    long_description=readme(),
    long_description_content_type="text/markdown",
    keywords=["hyperparameter tuning", "machine learning", "artificial intelligence",
              "hyperparameter optimization", "neural networks", "metaheuristics",
              "metaheuristic optimization", "scikit-learn", "sklearn", "grid search",
              "random search", "Bayesian optimization", "hyperparameter search", "nature-inspired algorithms",
              "classification", "regression", "supervised learning", "generalization",
              "optimization algorithms", "model selection", "Cross-validation"
              "Genetic algorithm (GA)", "Particle swarm optimization (PSO)", "Ant colony optimization (ACO)",
              "Differential evolution (DE)", "Simulated annealing", "Grey wolf optimizer (GWO)",
              "Whale Optimization Algorithm (WOA)", "automl", "parameter search", "mealpy", "search algorithm",
              "optimization framework", "global optimization", "local optimization",
              "Computational intelligence", "Robust optimization", "metaheuristic algorithms",
              "nature-inspired computing", "swarm-based computation", "gradient-free optimization"],
    url="https://github.com/thieu1995/MetaSklearn",
    project_urls={
        'Documentation': 'https://metasklearn.readthedocs.io/',
        'Source Code': 'https://github.com/thieu1995/MetaSklearn',
        'Bug Tracker': 'https://github.com/thieu1995/MetaSklearn/issues',
        'Change Log': 'https://github.com/thieu1995/MetaSklearn/blob/main/ChangeLog.md',
        'Forum': 'https://t.me/+fRVCJGuGJg1mNDg1',
    },
    packages=find_packages(exclude=['tests*', 'examples*']),
    include_package_data=True,
    license="GPLv3",
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Information Technology",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Natural Language :: English",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: System :: Benchmark",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Topic :: Scientific/Engineering :: Visualization",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Topic :: Software Development :: Build Tools",
        "Topic :: Software Development :: Libraries",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Utilities",
    ],
    install_requires=["numpy>=1.17.1", "scipy>=1.7.1", "scikit-learn>=1.0.2",
                      "pandas>=1.3.5", "mealpy>=3.0.2", "permetrics>=2.0.0"],
    extras_require={
        "dev": ["pytest>=7.0", "pytest-cov==4.0.0", "flake8>=4.0.1"],
    },
    python_requires='>=3.8',
)
