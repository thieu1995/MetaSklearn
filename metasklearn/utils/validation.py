#!/usr/bin/env python
# Created by "Thieu" at 21:39, 10/08/2023 ----------%
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import operator
import numpy as np
from numbers import Number

# Type aliases for convenience
SEQUENCE = (list, tuple, np.ndarray)
DIGIT = (int, np.integer)
REAL = (float, np.floating)


def is_in_bound(value, bound):
    """
    Check whether a numeric value is within the given bound.

    The function accepts bounds as a list (inclusive) or tuple (exclusive).
    Supports open-ended bounds using float('-inf') or float('inf').

    Args:
        value (float or int): The value to check.
        bound (list or tuple): A 2-element sequence defining the lower and upper bounds.

    Returns:
        bool: True if the value is within the bounds, False otherwise.
    """
    ops = None
    if type(bound) is tuple:
        ops = operator.lt
    elif type(bound) is list:
        ops = operator.le
    if bound[0] == float("-inf") and bound[1] == float("inf"):
        return True
    elif bound[0] == float("-inf") and ops(value, bound[1]):
        return True
    elif ops(bound[0], value) and bound[1] == float("inf"):
        return True
    elif ops(bound[0], value) and ops(value, bound[1]):
        return True
    return False


def is_str_in_list(value: str, my_list: list):
    """
    Check whether a string is present in a given list.

    Args:
        value (str): The string to check.
        my_list (list): The list of valid strings.

    Returns:
        bool: True if the string is in the list, False otherwise.
    """
    if type(value) == str and my_list is not None:
        return True if value in my_list else False
    return False


def check_int(name: str, value: None, bound=None):
    """
    Validate and cast a value to an integer, with optional bounds.

    Args:
        name (str): Name of the parameter (used in error message).
        value (int or float): The value to check.
        bound (tuple or list, optional): Inclusive or exclusive bounds for validation.

    Returns:
        int: The validated integer value.

    Raises:
        ValueError: If the value is not an integer or not within bounds.
    """
    if isinstance(value, Number):
        if bound is None:
            return int(value)
        elif is_in_bound(value, bound):
            return int(value)
    bound = "" if bound is None else f"and value should be in range: {bound}"
    raise ValueError(f"'{name}' is an integer {bound}.")


def check_float(name: str, value: None, bound=None):
    """
    Validate and cast a value to a float, with optional bounds.

    Args:
        name (str): Name of the parameter (used in error message).
        value (int or float): The value to check.
        bound (tuple or list, optional): Inclusive or exclusive bounds.

    Returns:
        float: The validated float value.

    Raises:
        ValueError: If the value is not a float or not within bounds.
    """
    if isinstance(value, Number):
        if bound is None:
            return float(value)
        elif is_in_bound(value, bound):
            return float(value)
    bound = "" if bound is None else f"and value should be in range: {bound}"
    raise ValueError(f"'{name}' is a float {bound}.")


def check_str(name: str, value: str, bound=None):
    """
    Validate a string against a list of allowed values.

    Args:
        name (str): Name of the parameter.
        value (str): The string value to check.
        bound (list, optional): List of allowed values.

    Returns:
        str: The validated string.

    Raises:
        ValueError: If the string is not allowed or not a string.
    """
    if type(value) is str:
        if bound is None or is_str_in_list(value, bound):
            return value
    bound = "" if bound is None else f"and value should be one of this: {bound}"
    raise ValueError(f"'{name}' is a string {bound}.")


def check_bool(name: str, value: bool, bound=(True, False)):
    """
    Validate a boolean value against allowed values.

    Args:
        name (str): Name of the parameter.
        value (bool): Boolean value to validate.
        bound (tuple, optional): Tuple of allowed boolean values.

    Returns:
        bool: The validated boolean.

    Raises:
        ValueError: If the value is not a valid boolean or not in allowed set.
    """
    if type(value) is bool:
        if value in bound:
            return value
    bound = "" if bound is None else f"and value should be one of this: {bound}"
    raise ValueError(f"'{name}' is a boolean {bound}.")


def check_tuple_int(name: str, values: None, bounds=None):
    """
    Validate a sequence of integers with optional individual bounds.

    Args:
        name (str): Name of the parameter.
        values (list, tuple, or ndarray): Sequence of integer values.
        bounds (list of tuple/list, optional): Bounds for each element.

    Returns:
        list or tuple or ndarray: The validated sequence.

    Raises:
        ValueError: If values are not integers or not within bounds.
    """
    if isinstance(values, SEQUENCE) and len(values) > 1:
        value_flag = [isinstance(item, DIGIT) for item in values]
        if np.all(value_flag):
            if bounds is not None and len(bounds) == len(values):
                value_flag = [is_in_bound(item, bound) for item, bound in zip(values, bounds)]
                if np.all(value_flag):
                    return values
            else:
                return values
    bounds = "" if bounds is None else f"and values should be in range: {bounds}"
    raise ValueError(f"'{name}' are integer {bounds}.")


def check_tuple_float(name: str, values: tuple, bounds=None):
    """
    Validate a sequence of floats with optional individual bounds.

    Args:
        name (str): Name of the parameter.
        values (list, tuple, or ndarray): Sequence of numeric values.
        bounds (list of tuple/list, optional): Bounds for each element.

    Returns:
        list or tuple or ndarray: The validated sequence.

    Raises:
        ValueError: If values are not floats or not within bounds.
    """
    if isinstance(values, SEQUENCE) and len(values) > 1:
        value_flag = [isinstance(item, Number) for item in values]
        if np.all(value_flag):
            if bounds is not None and len(bounds) == len(values):
                value_flag = [is_in_bound(item, bound) for item, bound in zip(values, bounds)]
                if np.all(value_flag):
                    return values
            else:
                return values
    bounds = "" if bounds is None else f"and values should be in range: {bounds}"
    raise ValueError(f"'{name}' are float {bounds}.")
