# process/processor.py
import cv2, numpy as np
from color_cs.color_utils import get_color_name, HAS_SKIMAGE
from config.config import _PALETTE

try:
    from skimage import color as skcolor
except ImportError: pass

try:
    from sklearn.cluster import KMeans
    HAS_KM = True
except ImportError:
    HAS_KM = False

def get_body_zone(roi_bgr):
    h, w = roi_bgr.shape[:2]
    gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    row_grad = np.abs(cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)).mean(axis=1)
    y0_search, y1_search = int(h * 0.25), int(h * 0.85)
    search = row_grad[y0_search:y1_search]
    win = max(1, int(len(search) * 0.35))
    best_y, best_score = y0_search, float('inf')
    for i in range(len(search) - win):
        score = search[i:i+win].mean()
        if score < best_score: best_score, best_y = score, y0_search + i
    zone = roi_bgr[best_y: best_y+win, int(w*0.15): int(w*0.85)]
    return zone if zone.size > 0 else roi_bgr[int(h*0.3):int(h*0.75), int(w*0.15):int(w*0.85)]

def specular_mask(hsv_img):
    S, V = hsv_img[:,:,1], hsv_img[:,:,2]
    mask = ~((S < 25) & (V > 210)) & ~(V < 35)
    if mask.sum() < mask.size * 0.10: mask = ~((S < 10) & (V > 230)) & ~(V < 35)
    return mask

def kmeans_color(roi_bgr, k=5):
    if roi_bgr is None or roi_bgr.size == 0: return [(100.0, "Unknown", "#999999")]
    body = get_body_zone(roi_bgr)
    scale = min(1.0, 80.0 / max(body.shape[:2]))
    small = cv2.resize(body, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV); mask = specular_mask(hsv)
    filtered_bgr = small.reshape(-1, 3)[mask.flatten()]
    if len(filtered_bgr) < k * 2: filtered_bgr = small.reshape(-1, 3)

    if HAS_SKIMAGE:
        rgb_f = filtered_bgr[:, ::-1].astype(np.float32) / 255.0
        lab_px = skcolor.rgb2lab(rgb_f.reshape(-1, 1, 3)).reshape(-1, 3).astype(np.float32)
    else: lab_px = filtered_bgr.astype(np.float32)

    if HAS_KM and len(lab_px) >= k:
        km = KMeans(n_clusters=k, n_init=8, max_iter=150, random_state=42)
        labels = km.fit_predict(lab_px); ctrs = km.cluster_centers_; cnts = np.bincount(labels, minlength=k)
        inertia = np.array([np.mean(np.sum((lab_px[labels==ci] - ctrs[ci])**2, axis=1)) if cnts[ci] > 0 else 1.0 for ci in range(k)])
    else: ctrs, cnts, inertia = lab_px.mean(axis=0, keepdims=True), np.array([len(lab_px)]), np.array([1.0])

    rows = []
    for ci, (ctr, cnt) in enumerate(zip(ctrs, cnts)):
        if cnt == 0: continue
        if HAS_SKIMAGE:
            bgr = (skcolor.lab2rgb(ctr.reshape(1,1,3).astype(np.float64)).reshape(3) * 255).clip(0,255).astype(np.uint8)[::-1]
        else: bgr = ctr.clip(0,255).astype(np.uint8)
        name, hexc, de = get_color_name(bgr)
        weight = cnt / max(inertia[ci], 0.1); rows.append((weight, name, hexc))

    total_w = sum(r[0] for r in rows); merged = {}
    for w, name, hexc in rows:
        pct = w / total_w * 100
        if name in merged: merged[name] = (merged[name][0] + pct, merged[name][1])
        else: merged[name] = (pct, hexc)
    result = sorted([(pct, name, hexc) for name, (pct, hexc) in merged.items()], reverse=True)
    total = sum(r[0] for r in result)
    return [(pct/total*100, name, hexc) for pct, name, hexc in result]

def roi_from_xyxy(img, xyxy):
    x0, y0, x1, y1 = [int(v) for v in xyxy]
    return img[max(0,y0):min(y1,img.shape[0]), max(0,x0):min(x1,img.shape[1])]