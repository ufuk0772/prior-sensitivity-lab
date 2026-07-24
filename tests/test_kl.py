import pytest
from src.prior_sensitivity_lab.distances.kl import kl_divergence

def test_kl_identical_distributions():
    """Aynı dağılımın kendisiyle KL ıraksaması 0 olmalıdır."""
    kl_val = kl_divergence(2.0, 2.0, 2.0, 2.0)
    assert kl_val == pytest.approx(0.0, abs=1e-5)

def test_kl_properties():
    """KL ıraksaması negatif olamaz ve asimetriktir."""
    kl_pq = kl_divergence(39.0, 13.0, 98.0, 32.0)
    kl_qp = kl_divergence(98.0, 32.0, 39.0, 13.0)
    
    assert kl_pq >= 0.0
    assert kl_qp >= 0.0
    assert kl_pq != kl_qp  # Genel olarak KL asimetriktir