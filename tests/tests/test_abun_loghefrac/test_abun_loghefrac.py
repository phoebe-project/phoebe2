import phoebe
import sys
import os

# Add parent directory to path to import conftest helpers
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from conftest import get_minimal_bundle_with_solution


def test_abun_loghefrac_parameters():
    b = phoebe.default_binary()
    b.add_dataset('lc', compute_phases=phoebe.linspace(0, 1, 11))

    # by default, both stars are ck2004 and therefore have abun but not loghefrac
    assert b.get_value(qualifier='atm', component='primary') == 'ck2004'
    assert b.get_value(qualifier='atm', component='secondary') == 'ck2004'
    assert len(b.filter(qualifier='abun')) == 2
    assert len(b.filter(qualifier='loghefrac')) == 0
    assert len(b.run_checks_compute()) == 0

    # changing one to tmap_sdO should hide abun and show loghefrac
    b.set_value(qualifier='atm', component='primary', value='tmap_sdO')
    assert len(b.filter(qualifier='abun', component='primary')) == 0
    assert len(b.filter(qualifier='loghefrac', component='primary')) == 1
    # but the default value for loghefrac (0.0) is out of bounds for tmap_sdO
    assert len(b.run_checks_compute()) == 1
    assert not b.run_checks_compute().passed
    # changing value within bounds should still pass
    b.set_value(qualifier='loghefrac', component='primary', value=-1.2)
    assert len(b.run_checks_compute()) == 0
    # revert value for next test
    b.set_value(qualifier='loghefrac', component='primary', value=0.0)

    # same for tmap_DO which uses a fixed value (9.4) for loghefrac
    b.set_value(qualifier='atm', component='primary', value='tmap_DO')
    assert len(b.filter(qualifier='abun', component='primary')) == 0
    assert len(b.filter(qualifier='loghefrac', component='primary')) == 1
    assert len(b.run_checks_compute()) == 1
    assert not b.run_checks_compute().passed
    b.set_value(qualifier='loghefrac', component='primary', value=9.4)
    assert len(b.run_checks_compute()) == 0
    # revert value for next test
    b.set_value(qualifier='loghefrac', component='primary', value=0.0)

    # blackbody uses a fixed value (0.0) for abun
    b.set_value(qualifier='atm', component='primary', value='blackbody')
    b.set_value(qualifier='ld_mode', component='primary', value='manual')
    assert len(b.filter(qualifier='abun', component='primary')) == 1
    assert len(b.filter(qualifier='loghefrac', component='primary')) == 0
    assert len(b.run_checks_compute()) == 0
    b.set_value(qualifier='abun', component='primary', value=0.2)
    assert len(b.run_checks_compute()) == 1
    assert not b.run_checks_compute().passed
    # revert value for next test
    b.set_value(qualifier='abun', component='primary', value=0.0)

    # setting abun to non-zero and then switching to an atmosphere that does not
    # use abun should result in a warning
    b.set_value(qualifier='abun', component='primary', value=0.3)
    b.set_value(qualifier='atm', component='primary', value='tmap_sdO')
    b.set_value(qualifier='loghefrac', component='primary', value=-1.0)
    assert len(b.filter(qualifier='abun', component='primary')) == 0
    assert len(b.run_checks_compute()) == 1
    assert b.run_checks_compute().passed


def test_abun_loghefrac_solver_checks():
    """Test run_checks_solver for fitting abun/loghefrac with incompatible atmospheres."""
    b = phoebe.default_binary()
    b.add_dataset('lc', compute_phases=phoebe.linspace(0, 1, 11))
    b.add_solver('optimizer.nelder_mead', solver='nm_solver')

    # test fitting abun when atmosphere supports it (ck2004)
    assert b.get_value(qualifier='atm', component='primary') == 'ck2004'
    b.set_value(qualifier='fit_parameters', solver='nm_solver', value=['abun@primary'])
    assert b.run_checks_solver(solver='nm_solver').passed

    # test fitting abun when atmosphere does not support it (tmap_sdO)
    b.set_value(qualifier='atm', component='primary', value='tmap_sdO')
    b.set_value(qualifier='loghefrac', component='primary', value=-1.0)
    # abun is now hidden, but we're still trying to fit it
    assert not b.run_checks_solver(solver='nm_solver').passed
    assert any('does not support interpolating over abun' in item.message for item in b.run_checks_solver(solver='nm_solver').items)

    # test fitting loghefrac when atmosphere supports interpolation (tmap_sdO)
    b.set_value(qualifier='fit_parameters', solver='nm_solver', value=['loghefrac@primary'])
    assert b.run_checks_solver(solver='nm_solver').passed

    # test fitting loghefrac when atmosphere uses a fixed value (tmap_DO)
    b.set_value(qualifier='atm', component='primary', value='tmap_DO')
    b.set_value(qualifier='loghefrac', component='primary', value=9.4)
    assert not b.run_checks_solver(solver='nm_solver').passed
    assert any('does not support interpolating over loghefrac' in item.message for item in b.run_checks_solver(solver='nm_solver').items)

    # test fitting loghefrac when atmosphere does not have loghefrac axis (ck2004)
    b.set_value(qualifier='atm', component='primary', value='ck2004')
    assert not b.run_checks_solver(solver='nm_solver').passed
    assert any('does not support interpolating over loghefrac' in item.message for item in b.run_checks_solver(solver='nm_solver').items)

    # test fitting abun when atmosphere uses a fixed value (blackbody)
    b.set_value(qualifier='atm', component='primary', value='blackbody')
    b.set_value(qualifier='ld_mode', component='primary', value='manual')
    b.set_value(qualifier='fit_parameters', solver='nm_solver', value=['abun@primary'])
    assert not b.run_checks_solver(solver='nm_solver').passed
    assert any('does not support interpolating over abun' in item.message for item in b.run_checks_solver(solver='nm_solver').items)


def test_abun_loghefrac_solver_init_from_solution():
    """Test run_checks_solver for init_from with solutions containing abun/loghefrac."""
    # Create a bundle with a solution that fitted abun
    b = get_minimal_bundle_with_solution(fit_parameters=['abun@primary'], solution_name='abun_solution')
    
    # Add a new solver that uses init_from with the solution
    # Note: fit_parameters must still be set - init_from just provides initial values
    b.add_solver('optimizer.nelder_mead', solver='nm_solver2', 
                 fit_parameters=['abun@primary'], init_from='abun_solution')
    
    # With ck2004 (default), abun is supported - should pass
    assert b.get_value(qualifier='atm', component='primary') == 'ck2004'
    assert b.run_checks_solver(solver='nm_solver2').passed
    
    # Switch to tmap_sdO which doesn't support abun - should fail
    b.set_value(qualifier='atm', component='primary', value='tmap_sdO')
    b.set_value(qualifier='loghefrac', component='primary', value=-1.0)
    assert not b.run_checks_solver(solver='nm_solver2').passed
    assert any('does not support interpolating over abun' in item.message for item in b.run_checks_solver(solver='nm_solver2').items)


if __name__ == '__main__':
    test_abun_loghefrac_parameters()
    test_abun_loghefrac_solver_checks()
    test_abun_loghefrac_solver_init_from_solution()