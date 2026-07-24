import pytest
from src.prior_sensitivity_lab.domain.power_scaling import scale_beta_prior, analytical_power_scaled_posterior

def test_scale_beta_prior_alpha_one():
    """alpha=1 durumunda önsel değişmemelidir."""
    a, b = 15.0, 5.0
    a_scaled, b_scaled = scale_beta_prior(a, b, 1.0)
    assert a_scaled == a
    assert b_scaled == b

def test_scale_beta_prior_alpha_zero():
    """alpha=0 durumunda önsel Beta(1,1) (Flat) olmalıdır."""
    a_scaled, b_scaled = scale_beta_prior(15.0, 5.0, 0.0)
    assert a_scaled == 1.0
    assert b_scaled == 1.0

def test_scale_beta_prior_alpha_two():
    """alpha=2 durumunda parametre denklemi doğru çalışmalıdır."""
    # a_scaled = 2 * (15 - 1) + 1 = 29
    # b_scaled = 2 * (5 - 1) + 1 = 9
    a_scaled, b_scaled = scale_beta_prior(15.0, 5.0, 2.0)
    assert a_scaled == 29.0
    assert b_scaled == 9.0

def test_analytical_power_scaled_posterior():
    """Ölçeklenmiş önsel ile posterior güncellenmesi doğru yapılmalıdır."""
    n = 50
    k = 38
    # alpha=2 için önsel Beta(29, 9) olur. 
    # Posterior: (29 + 38, 9 + (50-38)) = (67, 21)
    a_post, b_post = analytical_power_scaled_posterior(15.0, 5.0, k, n, 2.0)
    assert a_post == 67.0
    assert b_post == 21.0



