#!/usr/bin/env python
# Created by "Thieu" at 23:29, 24/09/2023 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import numpy as np
from metasklearn import Data
from metasklearn.utils.scaler import OneHotEncoder


X = np.array([[ 1., -2.,  2.],
                 [ -2.,  1.,  3.],
                 [ 4.,  1., -2.]])
y = np.array([[1, 2, 0]])

data = Data(X, y)
y, le = data.encode_label(y)
print(y)


# Input data
data = np.array(['cat', 'dog', 'bird', 'cat', 'dog', 'None', 1.4, 10, np.nan])

# Create and fit-transform
encoder = OneHotEncoder()
one_hot_encoded = encoder.fit_transform(data)

# Results
print("Categories:", encoder.categories_)
print("One-Hot Encoded Matrix:\n", one_hot_encoded)

# Inverse transform
original_data = encoder.inverse_transform(one_hot_encoded)

# Results
print("Categories:", encoder.categories_)
print("One-Hot Encoded Matrix:\n", one_hot_encoded)
print("Inverse Transformed Data:", original_data)
