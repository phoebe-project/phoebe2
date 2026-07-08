"""
Pytest fixtures and helpers for PHOEBE test suite.
"""

import pytest
import phoebe


def get_minimal_bundle_with_solution(fit_parameters=None, solution_name='test_solution'):
    """
    Create a minimal bundle with a solution for testing init_from scenarios.
    
    This uses minimal compute phases and maxiter=1 to make the solver run
    as fast as possible (~0.5-1s) while still producing a valid solution.
    
    Parameters
    ----------
    fit_parameters : list, optional
        Parameters to fit. Defaults to ['teff@primary'].
    solution_name : str, optional
        Name for the solution. Defaults to 'test_solution'.
    
    Returns
    -------
    b : phoebe.Bundle
        Bundle with a solution attached.
    """
    if fit_parameters is None:
        fit_parameters = ['teff@primary']
    
    b = phoebe.default_binary()
    b.add_dataset('lc', compute_phases=[0, 0.25, 0.5])  # minimal phases for speed
    b.add_solver('optimizer.nelder_mead', solver='temp_solver', 
                 fit_parameters=fit_parameters, maxiter=1)
    b.run_solver(solver='temp_solver', solution=solution_name)
    b.remove_solver('temp_solver')
    
    return b


@pytest.fixture
def bundle_with_solution():
    """
    Pytest fixture that provides a bundle with a minimal solution.
    
    The solution fits 'teff@primary' with maxiter=1 for speed.
    Solution is named 'test_solution'.
    
    Usage in tests:
        def test_something(bundle_with_solution):
            b = bundle_with_solution
            # b now has 'test_solution' available
    """
    return get_minimal_bundle_with_solution()
