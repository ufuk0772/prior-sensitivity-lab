import pytest
from src.prior_sensitivity_lab.distances.wasserstein import wasserstein_distance_beta

def test_wasserstein_identical_distributions():
    """Aynı dağılımın kendisiyle Wasserstein mesafesi 0 olmalıdır."""
    w_val = wasserstein_distance_beta(2.0, 2.0, 2.0, 2.0)
    assert w_val == pytest.approx(0.0, abs=1e-4)

def test_wasserstein_is_nonnegative_and_symmetric():
    """Wasserstein mesafesi negatif olamaz ve simetriktir."""
    w_1 = wasserstein_distance_beta(1.0, 1.0, 15.0, 5.0)
    w_2 = wasserstein_distance_beta(15.0, 5.0, 1.0, 1.0)
    
    assert w_1 >= 0.0
    assert w_1 == pytest.approx(w_2, abs=1e-4)  # Wasserstein mesafesi simetriktir

def test_wasserstein_invalid_params_raises():
    """Geçersiz (negatif veya sıfır) parametreler ValueError fırlatmalıdır."""
    with pytest.raises(ValueError):
        wasserstein_distance_beta(0, 1, 2, 2)