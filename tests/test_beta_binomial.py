import pytest
from src.prior_sensitivity_lab.domain.beta_binomial import posterior_params, shrinkage

def test_shrinkage_values():
    """
    Grafiklerde doğruladığımız 5 farklı önsel için büzülme (shrinkage S) 
    değerlerinin n=50 için doğru hesaplandığını test eder.
    """
    n = 50
    
    # 1. Flat (Beta 1, 1) -> S = (2) / (2 + 50) = 2/52 ≈ 0.0384
    assert shrinkage(1, 1, n) == pytest.approx(0.038, abs=1e-3)
    
    # 2. Zayıf (Beta 2, 2) -> S = (4) / (4 + 50) = 4/54 ≈ 0.0740
    assert shrinkage(2, 2, n) == pytest.approx(0.074, abs=1e-3)
    
    # 3. Bilgilendirici (Beta 15, 5) -> S = (20) / (20 + 50) = 20/70 ≈ 0.2857
    assert shrinkage(15, 5, n) == pytest.approx(0.286, abs=1e-3)
    
    # 4. Güçlü (Beta 60, 20) -> S = (80) / (80 + 50) = 80/130 ≈ 0.6153
    assert shrinkage(60, 20, n) == pytest.approx(0.615, abs=1e-3)
    
    # 5. Uyuşmazlıklı (Beta 5, 25) -> S = (30) / (30 + 50) = 30/80 = 0.3750
    assert shrinkage(5, 25, n) == pytest.approx(0.375, abs=1e-3)


def test_posterior_params():
    """
    Binom verisi (k=38, n=50) ile analitik sonsal parametrelerinin (a_post, b_post)
    doğru güncellendiğini test eder.
    """
    n = 50
    k = 38
    
    # Flat önsel için test: (1+38, 1+(50-38)) -> (39, 13)
    a_post, b_post = posterior_params(1, 1, k, n)
    assert a_post == 39
    assert b_post == 13
    
    # Güçlü önsel için test: (60+38, 20+(50-38)) -> (98, 32)
    a_post_strong, b_post_strong = posterior_params(60, 20, k, n)
    assert a_post_strong == 98
    assert b_post_strong == 32