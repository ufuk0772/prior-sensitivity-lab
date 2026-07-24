import pytest
from src.prior_sensitivity_lab.domain.power_scaling import scale_beta_prior
from src.prior_sensitivity_lab.domain.power_scaling_is import importance_weighted_moments

@pytest.mark.parametrize("alpha", [0.5, 1.5, 2.0])
def test_is_matches_analytic_mean(alpha):
    a_ref, b_ref = 39, 13  # Flat senaryosunun posterior'u
    a_true, b_true = scale_beta_prior(a_ref, b_ref, alpha)
    true_mean = a_true / (a_true + b_true)

    mean_est, _, _ = importance_weighted_moments(a_ref, b_ref, alpha, n_samples=50_000)
    assert mean_est == pytest.approx(true_mean, abs=0.01)