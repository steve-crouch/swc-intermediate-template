"""Tests for statistics functions within the Model layer."""

import numpy as np
import numpy.testing as npt
import pytest


@pytest.mark.parametrize(
    "test, expected", 
    [
     ([[0, 0], [0, 0], [0, 0]], [0, 0]),
     ([[1, 2], [3, 4], [5, 6]], [3, 4])
    ])
def test_daily_mean(test, expected):
    """Test mean function works for array of zeroes and positive integers."""
    from inflammation.models import daily_mean
    npt.assert_array_equal(np.array(expected), daily_mean(np.array(test)))


def test_daily_max():
    """Test that max function works for an array of positive integers."""
    from inflammation.models import daily_max

    test_array = np.array([[4, 2, 5],
                           [1, 6, 2],
                           [4, 1, 9]])  # yapf: disable

    # Need to use Numpy testing functions to compare arrays
    npt.assert_array_equal(np.array([4, 6, 9]), daily_max(test_array))


def test_daily_min():
    """Test that min function works for an array of positive and negative integers."""
    from inflammation.models import daily_min

    test_array = np.array([[ 4, -2, 5],
                           [ 1, -6, 2],
                           [-4, -1, 9]])  # yapf: disable

    # Need to use Numpy testing functions to compare arrays
    npt.assert_array_equal(np.array([-4, -6, 2]), daily_min(test_array))


def test_daily_min_string():
    """Test for TypeError when passing strings"""
    from inflammation.models import daily_min

    with pytest.raises(TypeError):
        error_expected = daily_min([['Hello', 'there'], ['General', 'Kenobi']])


# TODO(lesson-robust) Implement tests for the other statistical functions
