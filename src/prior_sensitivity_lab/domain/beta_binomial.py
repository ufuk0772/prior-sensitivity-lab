"""
Beta-Binomial Analitik Güncelleme ve Büzülme (Shrinkage) Modülü
"""

def posterior_params(a: float, b: float, k: int, n: int) -> tuple[float, float]:
    """
    Beta önseli ve Binom verisi (k, n) kullanılarak analitik sonsal parametrelerini hesaplar.
    
    Formül:
    a_post = a + k
    b_post = b + (n - k)
    """
    a_post = a + k
    b_post = b + (n - k)
    return a_post, b_post


def shrinkage(a: float, b: float, n: int) -> float:
    """
    Önselin ağırlığını ve verinin baskınlığını ölçen büzülme faktörünü (S) hesaplar.
    
    Formül:
    S = (a + b) / (a + b + n)
    """
    s_value = (a + b) / (a + b + n)
    return s_value