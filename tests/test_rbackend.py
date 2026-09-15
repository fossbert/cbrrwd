import numpy as np
import pytest


def test_rbackend_imports_without_r():
    # importing cbrrwd and the rbackend package must not require rpy2
    import cbrrwd.rbackend as rb

    assert hasattr(rb, "require_rpy2")
    assert hasattr(rb, "r_package")


def test_require_rpy2_raises_helpful_error_when_missing(monkeypatch):
    import builtins
    import cbrrwd.rbackend as rb

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "rpy2" or name.startswith("rpy2."):
            raise ImportError("no rpy2 here")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(ImportError, match="cbrrwd\\[r\\]"):
        rb.require_rpy2()


rpy2 = pytest.importorskip("rpy2")


def test_fisher_exact_rc_matches_scipy_on_2x2():
    from scipy.stats import fisher_exact

    from cbrrwd.rbackend.contingency import fisher_exact_rc

    table = np.array([[8, 2], [1, 9]])
    scipy_or, scipy_p = fisher_exact(table)
    r_or, r_p = fisher_exact_rc(table)

    assert r_p == pytest.approx(scipy_p, abs=1e-6)
    # R's conditional MLE odds ratio and SciPy's unconditional (sample) odds
    # ratio are different estimators and won't match closely -- just check
    # they agree on the direction of the association.
    assert (r_or > 1) == (scipy_or > 1)


def test_fisher_exact_rc_reports_nan_odds_ratio_for_larger_tables():
    from cbrrwd.rbackend.contingency import fisher_exact_rc

    table = np.array([[8, 2, 1], [1, 9, 3], [4, 4, 4]])
    orr, pval = fisher_exact_rc(table)

    assert np.isnan(orr)
    assert 0 <= pval <= 1
