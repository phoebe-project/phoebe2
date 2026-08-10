import os
import warnings

import numpy as np
import pytest

import phoebe
import phoebe.solverbackends as _solverbackends
from phoebe.solverbackends.solverbackends import _load_ebai_knn


def test_ebai_knn_load():
    """
    The bundled k-NN models used to ship as pickles of fitted estimators, which
    raised InconsistentVersionWarning under any scikit-learn other than the one
    that wrote them. They are now rebuilt from .npz at load time instead.
    """
    pytest.importorskip('sklearn')
    from sklearn.exceptions import InconsistentVersionWarning

    knn_dir = os.path.join(os.path.dirname(_solverbackends.__file__), 'knn')

    for fname in ['detached.pf.knn.npz', 'contact.2g.knn.npz']:
        with warnings.catch_warnings():
            warnings.simplefilter('error', InconsistentVersionWarning)
            ebai = _load_ebai_knn(os.path.join(knn_dir, fname))

        assert ebai.model is not None
        assert ebai.scaler is not None


def test_ebai_knn_solver():
    pytest.importorskip('sklearn')

    b = phoebe.default_binary()
    b.set_value('incl@binary', 85)
    b.add_dataset('lc', compute_times=phoebe.linspace(0, 1, 101))
    b.run_compute()
    b.set_value('times@dataset', b.get_value('times@model'))
    b.set_value('fluxes@dataset', b.get_value('fluxes@model'))
    b.set_value('sigmas@dataset', np.full_like(b.get_value('fluxes@model'), 0.01))

    b.add_solver('estimator.ebai', ebai_method='knn', solver='knn_solver')
    solution = b.run_solver('knn_solver')

    assert np.all(np.isfinite(np.array(solution.get_value('fitted_values'), dtype=float)))


if __name__ == '__main__':
    logger = phoebe.logger(clevel='INFO')
    test_ebai_knn_load()
    test_ebai_knn_solver()
