from stock_analysis.weighting import DEFAULT_WEIGHTS, REGIME_PROFILES, compute_weights


def test_compute_weights_normalised() -> None:
    weights = compute_weights("expansion")
    assert set(weights) == set(DEFAULT_WEIGHTS)
    assert abs(sum(weights.values()) - 1.0) < 1e-6
    assert weights["trend"] > weights["risk"]


def test_compute_weights_stress_emphasises_risk() -> None:
    weights = compute_weights("stress")
    assert weights["risk"] >= max(weights.values())


def test_default_weights_when_unknown_regime() -> None:
    weights = compute_weights("unknown")
    expected = compute_weights("expansion")
    assert weights == expected


def test_profiles_are_normalised() -> None:
    for profile in REGIME_PROFILES.values():
        assert abs(sum(profile.values()) - 1.0) < 1e-6
