# color_cs/color_utils.py
import numpy as np
import math
from config.config import _PALETTE

try:
    from skimage import color as skcolor
    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False

def _lab_from_rgb(rgb_array):
    if HAS_SKIMAGE:
        rgb_f = np.array(rgb_array, dtype=np.float32) / 255.0
        return skcolor.rgb2lab(rgb_f.reshape(1, 1, 3)).reshape(3)
    rgb_f = np.array(rgb_array, dtype=np.float64) / 255.0
    lin = np.where(rgb_f <= 0.04045, rgb_f / 12.92, ((rgb_f + 0.055) / 1.055) ** 2.4)
    M = np.array([[0.4124564, 0.3575761, 0.1804375], [0.2126729, 0.7151522, 0.0721750], [0.0193339, 0.1191920, 0.9503041]])
    xyz = (M @ lin) / np.array([0.95047, 1.00000, 1.08883])
    f = np.where(xyz > 0.008856, xyz ** (1/3), 7.787 * xyz + 16/116)
    return np.array([116*f[1]-16, 500*(f[0]-f[1]), 200*(f[1]-f[2])])

def delta_e_2000(lab1, lab2):
    L1, a1, b1 = lab1; L2, a2, b2 = lab2
    C1 = math.sqrt(a1**2 + b1**2); C2 = math.sqrt(a2**2 + b2**2)
    C_avg = (C1 + C2) / 2; C7 = C_avg**7
    G = 0.5 * (1 - math.sqrt(C7 / (C7 + 25**7)))
    a1p, a2p = a1*(1+G), a2*(1+G); C1p = math.sqrt(a1p**2 + b1**2); C2p = math.sqrt(a2p**2 + b2**2)
    h1p = math.degrees(math.atan2(b1, a1p)) % 360; h2p = math.degrees(math.atan2(b2, a2p)) % 360
    dLp = L2 - L1; dCp = C2p - C1p; dhp = h2p - h1p
    if abs(dhp) > 180: dhp += 360 if dhp < 0 else -360
    dHp = 2 * math.sqrt(C1p * C2p) * math.sin(math.radians(dhp / 2))
    Lp_avg = (L1 + L2) / 2; Cp_avg = (C1p + C2p) / 2; hp_avg = (h1p + h2p) / 2
    if abs(h1p - h2p) > 180: hp_avg += 180
    T = (1 - 0.17*math.cos(math.radians(hp_avg-30)) + 0.24*math.cos(math.radians(2*hp_avg)) + 0.32*math.cos(math.radians(3*hp_avg+6)) - 0.20*math.cos(math.radians(4*hp_avg-63)))
    SL = 1 + 0.015*(Lp_avg-50)**2 / math.sqrt(20+(Lp_avg-50)**2); SC = 1 + 0.045*Cp_avg; SH = 1 + 0.015*Cp_avg*T
    Cp7 = Cp_avg**7; RC = 2 * math.sqrt(Cp7 / (Cp7 + 25**7)); d_theta = 30 * math.exp(-((hp_avg-275)/25)**2); RT = -math.sin(math.radians(2*d_theta)) * RC
    return math.sqrt((dLp/SL)**2 + (dCp/SC)**2 + (dHp/SH)**2 + RT*(dCp/SC)*(dHp/SH))

# Pre-compute LAB
_PALETTE_LAB = {name: _lab_from_rgb(rgb) for name, rgb in _PALETTE.items()}

def get_color_name(bgr_color):
    lab_in = _lab_from_rgb(bgr_color[::-1])
    best_name, best_dist, best_rgb = "Unknown", float('inf'), [150, 150, 150]
    for name, lab_ref in _PALETTE_LAB.items():
        dist = delta_e_2000(lab_in, lab_ref)
        if dist < best_dist:
            best_dist, best_name, best_rgb = dist, name, _PALETTE[name]
    return best_name, '#%02x%02x%02x' % tuple(best_rgb), best_dist