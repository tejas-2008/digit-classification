import itertools
import pytest
def inc(x):
    return x+1

def test_hparamcount():
    gamma_ranges = [0.001, 0.01, 0.1, 1, 10, 100]
    C_ranges = [0.1, 1, 2, 5, 10]
    list_of_all_param_combinations = [{'gamma': gamma, 'C': C} for gamma, C in itertools.product(gamma_ranges, C_ranges)]
    assert len(C_ranges)*len(gamma_ranges) == len(list_of_all_param_combinations)



