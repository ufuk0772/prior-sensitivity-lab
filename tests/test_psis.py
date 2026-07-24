import pytest
import numpy as np
from scipy.stats import genpareto
from src.prior_sensitivity_lab.domain.psis import calculate_pareto_k

def test_calculate_pareto_k_good_tail():
    """Güvenilir ağırlıklarda (hafif kuyruk) k parametresi 0.5'in altında olmalıdır."""
    np.random.seed(42)
    # Eksponansiyel dağılım ince kuyrukludur (teorik k=0)
    weights = np.random.exponential(scale=1.0, size=2000)
    k_val = calculate_pareto_k(weights)
    
    assert k_val < 0.5

def test_calculate_pareto_k_bad_tail():
    """Ağır kuyruklu, bozulmuş ağırlıklarda k parametresi 0.7'nin üzerine çıkıp uyarı vermelidir."""
    np.random.seed(42)
    # Şekil parametresi k=0.8 olan gerçek bir Pareto dağılımından örneklem çekiyoruz
    weights = genpareto.rvs(c=0.8, size=2000)
    k_val = calculate_pareto_k(weights)
    
    assert k_val >= 0.7
    
def test_psis_invalid_length_raises():
    """Yetersiz örneklem boyutu ValueError fırlatmalıdır."""
    with pytest.raises(ValueError):
        calculate_pareto_k(np.array([1.5, 2.0, 3.1]))