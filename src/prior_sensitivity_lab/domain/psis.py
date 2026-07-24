"""
Pareto Smoothed Importance Sampling (PSIS) - k-Diyagnostiği Modülü
"""
import numpy as np
import warnings
from scipy.stats import genpareto

def calculate_pareto_k(weights: np.ndarray, tail_fraction: float = 0.2,min_samples: int = 5) -> float:
    if not (0 < tail_fraction < 1):
        raise ValueError("tail_fraction (0,1) araligindan olmalidir.")
    
    n = len(weights)
    if n < min_samples:
        raise ValueError((
            f"Yetersiz örneklem boyutu: {n} < {min_samples}. "
            "Pareto k tahmini için minimum örneklem sayısı sağlanmalı."
        ))
    # Vehtari ve ark. (2015) onerisi: M = min(0.2*S, 3*sqrt(S))
    n_tail = max(min(int(n * tail_fraction), int(3 * np.sqrt(n))), 5)
    
    sorted_w = np.sort(weights)
    tail = sorted_w[-n_tail:]
    threshold = sorted_w[-n_tail - 1] if len(sorted_w) > n_tail else 0.0
    exceedances = tail - threshold
    
    k_hat, _, _ = genpareto.fit(exceedances, floc=0)
    return float(k_hat)