"""
Beta Dağılımları Arası Wasserstein Mesafesi Modülü
"""
import numpy as np
import scipy.stats as stats

def wasserstein_distance_beta(a1: float, b1: float, a2: float, b2: float, num_points: int = 1000) -> float:
    """
    Beta(a1, b1) ve Beta(a2, b2) dağılımları arasında, ortak bir grid üzerinde
    örnekleme yaparak nümerik Wasserstein mesafesini hesaplar.
    
    Design Notes:
        - Beta ailesi için analitik Wasserstein kapalı formu olmadığından,
          scipy.stats.wasserstein_distance nümerik CDF entegrasyonu için kullanılır.
        - num_points parametresi yaklaşımın hassasiyetini belirler.
    """
    if min(a1, b1, a2, b2) <= 0:
        raise ValueError("Beta parametreleri (a, b) pozitif olmalıdır.")
        
    # [0, 1] aralığında ortak grid oluştur
    x = np.linspace(0, 1, num_points)
    
    # Olasılık yoğunluk fonksiyonlarını (PDF) hesapla
    pdf1 = stats.beta.pdf(x, a1, b1)
    pdf2 = stats.beta.pdf(x, a2, b2)
    
    # PDF'leri normalize et (sayısal kararlılık için toplamları 1 olsun)
    pdf1 /= np.sum(pdf1)
    pdf2 /= np.sum(pdf2)
    
    # Wasserstein mesafesini hesapla
    w_dist = stats.wasserstein_distance(x, x, pdf1, pdf2)
    
    return float(w_dist)