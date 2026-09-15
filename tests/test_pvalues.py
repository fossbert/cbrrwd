import numpy as np
import pandas as pd
import pytest

from cbrrwd.pvalues import cut_p, fdr, fisher_test


@pytest.mark.parametrize("p,expected", [
    (0.0001, "***"),
    (0.005, "**"),
    (0.02, "*"),
    (0.07, "0.07"),
    (0.5, "ns"),
])
def test_cut_p(p, expected):
    assert cut_p(p) == expected


def test_fdr_monotonic_and_bounded():
    pvals = [0.001, 0.01, 0.2, 0.8]
    adj = fdr(pvals)
    assert len(adj) == len(pvals)
    assert all(0 <= p <= 1 for p in adj)
    assert adj[0] <= adj[1] <= adj[2] <= adj[3]


def test_fisher_test_2x2_uses_scipy_and_ranks_by_pval():
    rng = np.random.default_rng(0)
    n = 40
    ref = np.array(["A"] * (n // 2) + ["B"] * (n // 2))

    # 'hit' strongly associated with ref, 'null' independent
    hit = np.where(ref == "A", rng.random(n) < 0.9, rng.random(n) < 0.1)
    null = rng.random(n) < 0.5

    df = pd.DataFrame({"hit": hit, "null": null})
    res = fisher_test(df, ref)

    assert list(res.index)[0] == "hit"
    assert {"pval", "fdr", "odds_ratio"} == set(res.columns)
    assert res.loc["hit", "pval"] < 0.05


def test_fisher_test_drops_degenerate_columns():
    ref = np.array(["A", "A", "A", "B", "B", "B"])
    df = pd.DataFrame({
        "constant": [1, 1, 1, 1, 1, 1],  # single category -> dropped
        "varies": [0, 0, 1, 1, 1, 0],
    })
    res = fisher_test(df, ref)
    assert "constant" not in res.index
    assert "varies" in res.index
