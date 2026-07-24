"""
Beta Dağılımları Arası Jensen-Shannon (JS) Iraksama Modülü
"""
import numpy as np
import scipy.stats as stats
from scipy.spatial.distance import jensenshannon

def js_divergence_beta(a1: float, b1: float, a2: float, b2: float, num_points: int = 1000) -> float:
    """
    Beta(a1, b1) ve Beta(a2, b2) dağılımları arasındaki Jensen-Shannon ıraksamasını hesaplar.
    
    Design Notes:
        - JS ıraksaması, simetrik ve sınırlandırılmış (0 ile ln(2) arası) bir metriktir.
        - scipy.spatial.distance.jensenshannon fonksiyonu mesafe (karekök) döndürdüğü için, 
          ıraksamayı bulmak adına sonucun karesi alınır.
    """
    if min(a1, b1, a2, b2) <= 0:
        raise ValueError("Beta parametreleri (a, b) pozitif olmalıdır.")
        
    x = np.linspace(0, 1, num_points)
    
    pdf1 = stats.beta.pdf(x, a1, b1)
    pdf2 = stats.beta.pdf(x, a2, b2)
    
    # PDF'leri normalize ederek gerçek birer olasılık vektörüne dönüştürüyoruz
    pdf1 /= np.sum(pdf1)
    pdf2 /= np.sum(pdf2)
    
    # jensenshannon() karekök(JS) döndürür, bu yüzden karesini alıyoruz
    js_dist = jensenshannon(pdf1, pdf2)
    
    return float(js_dist ** 2)