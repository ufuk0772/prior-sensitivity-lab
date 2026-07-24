import pytest
from src.prior_sensitivity_lab.distances.js import js_divergence_beta

def test_js_identical_distributions():
    """Aynı dağılımın kendisiyle JS ıraksaması 0 olmalıdır."""
    assert js_divergence_beta(2.0, 2.0, 2.0, 2.0) == pytest.approx(0.0, abs=1e-4)

def test_js_symmetric():
    """JS ıraksaması simetrik olmalıdır."""
    js1 = js_divergence_beta(1.0, 1.0, 15.0, 5.0)
    js2 = js_divergence_beta(15.0, 5.0, 1.0, 1.0)
    assert js1 == pytest.approx(js2, abs=1e-4)

def test_js_invalid_params_raises():
    """Geçersiz (negatif veya sıfır) parametreler ValueError fırlatmalıdır."""
    with pytest.raises(ValueError):
        js_divergence_beta(0, 1, 2, 2)