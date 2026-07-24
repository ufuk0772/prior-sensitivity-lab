"""
Beta-Binom Modeli İçin Güç Ölçekleme (Power-Scaling) Modülü
"""
from src.prior_sensitivity_lab.domain.beta_binomial import posterior_params

def scale_beta_prior(a: float, b: float, alpha: float) -> tuple[float, float]:
    """
    Beta(a, b) önsel dağılımını alpha parametresi ile güç-ölçeklemesine (power-scaling) tabi tutar.
    
    Design Notes:
        - alpha = 1.0: Orijinal önseli döndürür.
        - alpha < 1.0: Önseli zayıflatır (Flat dağılıma, yani Beta(1,1)'e yaklaştırır).
        - alpha > 1.0: Önseli keskinleştirir (Önselin veri üzerindeki hakimiyetini artırır).
    """
    if a <= 0 or b <= 0:
        raise ValueError("Beta parametreleri (a, b) pozitif olmalıdır.")
    if alpha < 0:
        raise ValueError("Güç ölçekleme faktörü (alpha) negatif olamaz.")
        
    a_scaled = alpha * (a - 1) + 1
    b_scaled = alpha * (b - 1) + 1
    
    if a_scaled <= 0 or b_scaled <= 0:
        raise ValueError(f"Ölçeklenmiş parametreler pozitif olmalıdır. Geçersiz alpha: {alpha}")
        
    return float(a_scaled), float(b_scaled)


def analytical_power_scaled_posterior(a_prior: float, b_prior: float, k: int, n: int, alpha: float) -> tuple[float, float]:
    """
    Güç-ölçeklenmiş önsel ve mevcut Binom verisi (k, n) kullanılarak yeni 
    analitik sonsal (posterior) parametrelerini hesaplar.
    """
    a_scaled, b_scaled = scale_beta_prior(a_prior, b_prior, alpha)
    return posterior_params(a_scaled, b_scaled, k, n)


import numpy as np
import scipy.stats as stats

def importance_sampling_power_scale(a_post: float, b_post: float, a_prior: float, b_prior: float, alpha: float, num_samples: int = 100000) -> tuple[float, float, np.ndarray]:
    
    theta_samples = stats.beta.rvs(a_post, b_post, size=num_samples)
    
    log_prior_pdf = stats.beta.logpdf(theta_samples, a_prior, b_prior)
    log_weights = (alpha - 1.0) * log_prior_pdf
    
    # LogSumExp taktiği: Sayısal taşmayı önlemek için maksimuma göre kaydır
    max_log_w = np.max(log_weights)
    raw_weights = np.exp(log_weights - max_log_w) 
    
    # Ortalama/Varyans hesabı için normalize et
    norm_weights = raw_weights / np.sum(raw_weights)
    
    is_mean = float(np.sum(norm_weights * theta_samples))
    is_var = float(np.sum(norm_weights * (theta_samples - is_mean)**2))
    
    # DİKKAT: PSIS diyagnostiği için 'norm_weights' yerine 'raw_weights' döndürüyoruz!
    return is_mean, is_var, raw_weights