"""
Beta-Binom Modeli için Önem Örneklemesi (Importance Sampling) ile 
Güç-Ölçeklenmiş Posterior Moment Tahmini

Design Notes:
    - Referans posterior'dan (alpha=1) cekilen ornekler agirliklandirilarak
      guc-olceklenmis posterior'un momentleri tahmin edilir.
    - w(theta) = pi(theta)^(alpha-1), pi = referans posterior yogunlugu.
    - Bu modul, analytical_power_scaled_posterior ile karsilastirilarak
      IS yonteminin dogrulugunu kanitlamak icin var; gercek DCR modellerinde
      analitik form olmayacagi icin bu dogrulama kritik.
"""
import numpy as np
from scipy.stats import beta as beta_dist


def importance_weighted_moments(
    a_ref: float, b_ref: float, alpha: float,
    n_samples: int = 20_000, seed: int = 42,
) -> tuple[float, float, np.ndarray]:
    if a_ref <= 0 or b_ref <= 0:
        raise ValueError("Referans Beta parametreleri pozitif olmalıdır.")
    if alpha < 0:
        raise ValueError("Güç ölçekleme faktörü (alpha) negatif olamaz.")

    rng = np.random.default_rng(seed)
    theta = beta_dist.rvs(a_ref, b_ref, size=n_samples, random_state=rng)

    log_pi = beta_dist.logpdf(theta, a_ref, b_ref)
    log_w = (alpha - 1) * log_pi
    log_w -= log_w.max()  # sayısal kararlılık (overflow önleme)
    w = np.exp(log_w)
    w /= w.sum()

    mean_est = float(np.sum(w * theta))
    var_est = float(np.sum(w * (theta - mean_est) ** 2))
    return mean_est, var_est, w