"""
Beta Dağılımları Arası Kullback-Leibler (KL) Iraksama Modülü
"""
import numpy as np
from scipy.special import betaln, digamma

def kl_divergence(a1: float, b1: float, a2: float, b2: float) -> float:
    """
    Beta(a1, b1) dağılımının Beta(a2, b2) dağılımına göre KL ıraksamasını 
    (D_KL(P || Q)) analitik kapalı form ile hesaplar.
    
    D_KL(P || Q) = ln(B(a2,b2) / B(a1,b1)) + (a1 - a2)*psi(a1) + (b1 - b2)*psi(b1) + (a2 - a1 + b2 - b1)*psi(a1 + b1)
    """
    # Log-uzayda Beta fonksiyonu (Sayısal kararlılık için betaln kullanılır)
    log_B_ratio = betaln(a2, b2) - betaln(a1, b1)
    
    term1 = log_B_ratio
    term2 = (a1 - a2) * digamma(a1)
    term3 = (b1 - b2) * digamma(b1)
    term4 = (a2 - a1 + b2 - b1) * digamma(a1 + b1)
    
    kl_val = term1 + term2 + term3 + term4
    return float(kl_val)