"""
Test for issue #1126: the hot-star bolometric warning.

Background: the warning was moved out of run_checks_system into
run_checks_compute (it is now atm-dependent, and atm is a compute-context
parameter). It should fire for a hot star (teff >= 10000 K) with
ld_mode_bol='lookup', EXCEPT when the atm grid is Tremblay/TMAP, in which case
the bolometric passband is considered reliable and the warning is suppressed.
"""

import phoebe

# stable fragment of the warning text (robust to wording/whitespace tweaks)
_WARNING_FRAGMENT = "not reliable for hot stars"

# atm grids for which the warning is suppressed
_TMAP_ATMS = ['tremblay', 'tmap_sdO', 'tmap_DA', 'tmap_DO', 'tmap_DAO']


def _has_hot_star_bol_warning(report):
    """True iff the hot-star bolometric warning is present in the report."""
    return any(_WARNING_FRAGMENT in item.message for item in report.items)


def test_hot_star_bol_warning():
    b = phoebe.Bundle.default_binary()

    # defaults: teff=6000, atm='ck2004', ld_mode_bol='lookup' -> no warning
    assert not _has_hot_star_bol_warning(b.run_checks())

    # --- cold star: warning never fires, regardless of atm ----------------------
    b.set_value('teff', component='primary', value=6000)
    for atm in ['ck2004'] + _TMAP_ATMS:
        b.set_value('atm', component='primary', value=atm)
        assert not _has_hot_star_bol_warning(b.run_checks()), \
            "cold star should not warn (atm={})".format(atm)

    # --- hot star + ld_mode_bol='lookup' ---------------------------------------
    b.set_value('teff', component='primary', value=12000)
    b.set_value('ld_mode_bol', component='primary', value='lookup')

    # normal grid -> warning SHOULD fire (and it's a warning, so checks still pass)
    b.set_value('atm', component='primary', value='ck2004')
    report = b.run_checks()
    assert _has_hot_star_bol_warning(report), \
        "hot star with atm='ck2004' + ld_mode_bol='lookup' should warn"
    assert report.passed

    # the check now lives in run_checks_compute -- verify it there directly too
    assert _has_hot_star_bol_warning(b.run_checks_compute())

    # Tremblay/TMAP grids -> warning suppressed
    for atm in _TMAP_ATMS:
        b.set_value('atm', component='primary', value=atm)
        assert not _has_hot_star_bol_warning(b.run_checks()), \
            "hot star with atm={} should NOT warn".format(atm)
        assert not _has_hot_star_bol_warning(b.run_checks_compute()), \
            "hot star with atm={} should NOT warn (run_checks_compute)".format(atm)

    # --- hot star but ld_mode_bol='manual': warning never fires -----------------
    b.set_value('atm', component='primary', value='ck2004')
    b.set_value('ld_mode_bol', component='primary', value='manual')
    b.set_value('ld_coeffs_bol', component='primary', value=[0.5, 0.5])
    assert not _has_hot_star_bol_warning(b.run_checks()), \
        "ld_mode_bol='manual' should not warn"

if __name__ == '__main__':
    logger = phoebe.logger(clevel='INFO')
    test_hot_star_bol_warning()