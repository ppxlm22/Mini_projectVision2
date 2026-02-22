import tkinter as tk
from tkinter import ttk, filedialog
import cv2, numpy as np, threading, os, math
from PIL import Image, ImageTk

try:
    from skimage import color as skcolor
    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False

try:
    from sklearn.cluster import KMeans
    HAS_KM = True
except ImportError:
    HAS_KM = False

try:
    from ultralytics import YOLO
    HAS_YOLO = True
except ImportError:
    HAS_YOLO = False

# ════════════════════════════════════════════
#  CONFIG
# ════════════════════════════════════════════
MODEL_PATHS = ["runs/best.pt", "DetectCars.pt", "yolov11n.pt"]
CAR_CLASSES  = {"car", "cars", "automobile", "vehicle", "truck", "van", "bus"}

# Representative RGB values for each car color (mid-tone, not over-bright or dark)
_PALETTE = {
    "Pearl White":      [250, 248, 245],
    "White":            [240, 240, 240],
    "Silver":           [190, 190, 195],    
    "Light Gray":       [160, 162, 165],
    "Gray":             [110, 112, 115],
    "Dark Gray":        [70,  72,  75],
    "Black":            [25,  25,  28],
    "Red":              [185, 15,  15],
    "Dark Red":         [120, 10,  10],
    "Metallic Red":     [160, 30,  40],
    "Bordeaux":         [100, 0,   30],
    "Dark Blue":        [10,  30,  100],
    "Blue":             [20,  60,  160],
    "Midnight Blue":    [15,  20,  70],
    "Sky Blue":         [100, 170, 220],
    "Dark Green":       [10,  60,  30],
    "Green":            [30,  120, 50],
    "Olive/Military":   [60,  80,  50],
    "Mint Green":       [120, 200, 170],
    "Yellow":           [255, 215, 0],
    "Gold":             [200, 155, 20],
    "Orange":           [230, 95,  10],
    "Brown":            [100, 55,  25],
    "Tan":              [175, 130, 85],
    "Cream":            [240, 235, 200],
    "Beige":            [210, 195, 165],
    "Purple":           [90,  20,  110],
    "Metallic Purple":  [120, 60,  140],
    "Pink":             [220, 120, 150],
}

# ════════════════════════════════════════════
#  COLOR MATCHING — Delta E 2000
#  Industry-standard perceptual color difference formula.
#  Accounts for non-linear human perception, especially in blue and high-lightness zones.
# ════════════════════════════════════════════

def _lab_from_rgb(rgb_array):
    if HAS_SKIMAGE:
        rgb_f = np.array(rgb_array, dtype=np.float32) / 255.0
        return skcolor.rgb2lab(rgb_f.reshape(1, 1, 3)).reshape(3)
    # Manual fallback: sRGB → linear → XYZ (D65) → LAB
    rgb_f = np.array(rgb_array, dtype=np.float64) / 255.0
    lin = np.where(rgb_f <= 0.04045, rgb_f / 12.92, ((rgb_f + 0.055) / 1.055) ** 2.4)
    M = np.array([[0.4124564, 0.3575761, 0.1804375],
                  [0.2126729, 0.7151522, 0.0721750],
                  [0.0193339, 0.1191920, 0.9503041]])
    xyz = (M @ lin) / np.array([0.95047, 1.00000, 1.08883])
    f = np.where(xyz > 0.008856, xyz ** (1/3), 7.787 * xyz + 16/116)
    return np.array([116*f[1]-16, 500*(f[0]-f[1]), 200*(f[1]-f[2])])


def delta_e_2000(lab1, lab2):
    """CIE Delta E 2000 — weighted distance in L*a*b* with Hue/Chroma/Lightness corrections."""
    L1, a1, b1 = lab1
    L2, a2, b2 = lab2
    C1 = math.sqrt(a1**2 + b1**2)
    C2 = math.sqrt(a2**2 + b2**2)
    C_avg = (C1 + C2) / 2
    C7 = C_avg**7
    G = 0.5 * (1 - math.sqrt(C7 / (C7 + 25**7)))
    a1p, a2p = a1*(1+G), a2*(1+G)
    C1p = math.sqrt(a1p**2 + b1**2)
    C2p = math.sqrt(a2p**2 + b2**2)
    h1p = math.degrees(math.atan2(b1, a1p)) % 360
    h2p = math.degrees(math.atan2(b2, a2p)) % 360
    dLp = L2 - L1
    dCp = C2p - C1p
    dhp = h2p - h1p
    if abs(dhp) > 180: dhp += 360 if dhp < 0 else -360
    dHp = 2 * math.sqrt(C1p * C2p) * math.sin(math.radians(dhp / 2))
    Lp_avg = (L1 + L2) / 2
    Cp_avg = (C1p + C2p) / 2
    hp_avg = (h1p + h2p) / 2
    if abs(h1p - h2p) > 180: hp_avg += 180
    T = (1 - 0.17*math.cos(math.radians(hp_avg-30))
           + 0.24*math.cos(math.radians(2*hp_avg))
           + 0.32*math.cos(math.radians(3*hp_avg+6))
           - 0.20*math.cos(math.radians(4*hp_avg-63)))
    SL = 1 + 0.015*(Lp_avg-50)**2 / math.sqrt(20+(Lp_avg-50)**2)
    SC = 1 + 0.045*Cp_avg
    SH = 1 + 0.015*Cp_avg*T
    Cp7 = Cp_avg**7
    RC = 2 * math.sqrt(Cp7 / (Cp7 + 25**7))
    d_theta = 30 * math.exp(-((hp_avg-275)/25)**2)
    RT = -math.sin(math.radians(2*d_theta)) * RC
    return math.sqrt((dLp/SL)**2 + (dCp/SC)**2 + (dHp/SH)**2 + RT*(dCp/SC)*(dHp/SH))


# Pre-compute LAB values for each palette entry
_PALETTE_LAB = {name: _lab_from_rgb(rgb) for name, rgb in _PALETTE.items()}


def get_color_name(bgr_color):
    """Match a BGR pixel to the nearest palette color using Delta E 2000."""
    lab_in = _lab_from_rgb(bgr_color[::-1])  # BGR → RGB → LAB
    best_name, best_dist, best_rgb = "Unknown", float('inf'), [150, 150, 150]
    for name, lab_ref in _PALETTE_LAB.items():
        dist = delta_e_2000(lab_in, lab_ref)
        if dist < best_dist:
            best_dist, best_name, best_rgb = dist, name, _PALETTE[name]
    return best_name, '#%02x%02x%02x' % tuple(best_rgb), best_dist


def get_body_zone(roi_bgr):
    """Return the sub-region most likely to be the car body panel (low-gradient band)."""
    h, w = roi_bgr.shape[:2]
    gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    row_grad = np.abs(cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)).mean(axis=1)

    y0_search, y1_search = int(h * 0.25), int(h * 0.85)
    search = row_grad[y0_search:y1_search]
    win = max(1, int(len(search) * 0.35))

    best_y, best_score = y0_search, float('inf')
    for i in range(len(search) - win):
        score = search[i:i+win].mean()
        if score < best_score:
            best_score, best_y = score, y0_search + i

    zone = roi_bgr[best_y: best_y+win, int(w*0.15): int(w*0.85)]
    return zone if zone.size > 0 else roi_bgr[int(h*0.3):int(h*0.75), int(w*0.15):int(w*0.85)]


def specular_mask(hsv_img):

    S, V = hsv_img[:,:,1], hsv_img[:,:,2]
    mask = ~((S < 25) & (V > 210)) & ~(V < 35)
    if mask.sum() < mask.size * 0.10:
        mask = ~((S < 10) & (V > 230)) & ~(V < 35)
    return mask


# ════════════════════════════════════════════
#  K-MEANS IN LAB SPACE
#  Cluster in perceptual LAB, then rank by compactness-weighted count.
# ════════════════════════════════════════════

def kmeans_color(roi_bgr, k=5):
    if roi_bgr is None or roi_bgr.size == 0:
        return [(100.0, "Unknown", "#999999")]

    body  = get_body_zone(roi_bgr)
    scale = min(1.0, 80.0 / max(body.shape[:2]))
    small = cv2.resize(body, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

    hsv  = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    mask = specular_mask(hsv)
    filtered_bgr = small.reshape(-1, 3)[mask.flatten()]
    if len(filtered_bgr) < k * 2:
        filtered_bgr = small.reshape(-1, 3)

    # Convert filtered pixels to LAB for clustering
    if HAS_SKIMAGE:
        rgb_f = filtered_bgr[:, ::-1].astype(np.float32) / 255.0
        lab_px = skcolor.rgb2lab(rgb_f.reshape(-1, 1, 3)).reshape(-1, 3).astype(np.float32)
    else:
        lab_px = filtered_bgr.astype(np.float32)

    if HAS_KM and len(lab_px) >= k:
        km = KMeans(n_clusters=k, n_init=8, max_iter=150, random_state=42)
        labels = km.fit_predict(lab_px)
        ctrs   = km.cluster_centers_
        cnts   = np.bincount(labels, minlength=k)
        # Per-cluster inertia (compactness): tighter clusters are more trustworthy
        inertia = np.array([
            np.mean(np.sum((lab_px[labels==ci] - ctrs[ci])**2, axis=1)) if cnts[ci] > 0 else 1.0
            for ci in range(k)
        ])
    else:
        ctrs    = lab_px.mean(axis=0, keepdims=True)
        cnts    = np.array([len(lab_px)])
        inertia = np.array([1.0])

    rows = []
    for ci, (ctr, cnt) in enumerate(zip(ctrs, cnts)):
        if cnt == 0:
            continue
        # Convert LAB cluster center back to BGR for name matching
        if HAS_SKIMAGE:
            bgr = (skcolor.lab2rgb(ctr.reshape(1,1,3).astype(np.float64)).reshape(3) * 255
                   ).clip(0,255).astype(np.uint8)[::-1]
        else:
            bgr = ctr.clip(0,255).astype(np.uint8)

        name, hexc, de = get_color_name(bgr)
        weight = cnt / max(inertia[ci], 0.1)  # large + compact cluster wins
        rows.append((weight, name, hexc))

    if not rows:
        return [(100.0, "Unknown", "#999999")]

    # Merge duplicate color names, then normalize to percentages
    total_w = sum(r[0] for r in rows)
    merged  = {}
    for w, name, hexc in rows:
        pct = w / total_w * 100
        if name in merged:
            merged[name] = (merged[name][0] + pct, merged[name][1])
        else:
            merged[name] = (pct, hexc)

    result = sorted([(pct, name, hexc) for name, (pct, hexc) in merged.items()], reverse=True)
    total  = sum(r[0] for r in result)
    return [(pct/total*100, name, hexc) for pct, name, hexc in result]


def roi_from_xyxy(img, xyxy):
    x0, y0, x1, y1 = [int(v) for v in xyxy]
    return img[max(0,y0):min(y1,img.shape[0]), max(0,x0):min(x1,img.shape[1])]


# ════════════════════════════════════════════
#  UI
# ════════════════════════════════════════════
C = dict(
    bg="#FEF6ED", bg2="#FDEBD4", card="#FFFFFF",
    border="#F5C99A", sidebar="#FF7A1F", side2="#E85D00",
    accent="#FF5722", amber="#FF9800",
    text="#3A1800", text2="#7A4010", text3="#C07030", white="#FFFFFF",
)


class RoundBtn(tk.Canvas):
    def __init__(self, parent, text, cmd=None, w=182, h=38,
                 bg="#FF5722", fg="#FFF", hover="#FF7043",
                 font=("Segoe UI", 10, "bold"), r=19, enabled=True):
        super().__init__(parent, width=w, height=h,
                         bg=parent.cget("bg"), highlightthickness=0, bd=0)
        self.cmd = cmd; self.cbg = bg; self.chover = hover; self.cfg = fg
        self.on = enabled; self.dis = "#D4B8A0"
        self.rect = self._rr(1, 1, w-1, h-1, r, fill=bg if enabled else self.dis)
        self.txt  = self.create_text(w//2, h//2, text=text,
                                     fill=fg if enabled else "#B09080",
                                     font=font, anchor="center")
        self.bind("<Enter>",           lambda _: self._c(self.chover) if self.on else None)
        self.bind("<Leave>",           lambda _: self._c(self.cbg)    if self.on else None)
        self.bind("<ButtonPress-1>",   self._dn)
        self.bind("<ButtonRelease-1>", self._up)

    def _rr(self, x1, y1, x2, y2, r, **kw):
        p = [x1+r,y1, x2-r,y1, x2,y1, x2,y1+r, x2,y2-r, x2,y2,
             x2-r,y2, x1+r,y2, x1,y2, x1,y2-r, x1,y1+r, x1,y1]
        return self.create_polygon(p, smooth=True, **kw)

    def _c(self, col): self.itemconfig(self.rect, fill=col)
    def _dn(self, _):
        if self.on: self._c(self.cfg); self.itemconfig(self.txt, fill=self.cbg)
    def _up(self, _):
        if self.on:
            self._c(self.chover); self.itemconfig(self.txt, fill=self.cfg)
            if self.cmd: self.cmd()

    def enable(self):
        self.on = True;  self._c(self.cbg); self.itemconfig(self.txt, fill=self.cfg)
    def disable(self):
        self.on = False; self._c(self.dis);  self.itemconfig(self.txt, fill="#B09080")


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Car Color Detector — Precision v2")
        self.geometry("1320x860")
        self.configure(bg=C["bg"])
        self.model = None; self.img_path = None
        self.result_pil = None; self.dets = []; self._tkimg = None; self._cdata = []
        self._ttk(); self._ui()
        threading.Thread(target=self._load_model, daemon=True).start()

    def _ttk(self):
        s = ttk.Style(self); s.theme_use("clam")
        s.configure("T.Treeview", background=C["card"], foreground=C["text"],
                    fieldbackground=C["card"], font=("Segoe UI", 10), rowheight=32, borderwidth=0)
        s.configure("T.Treeview.Heading", background=C["sidebar"], foreground=C["white"],
                    font=("Segoe UI", 9, "bold"), relief="flat", padding=(4, 6))
        s.map("T.Treeview", background=[("selected", C["amber"])],
              foreground=[("selected", C["white"])])

    def _ui(self):
        # Sidebar
        s = tk.Frame(self, bg=C["sidebar"], width=215)
        s.pack(side="left", fill="y"); s.pack_propagate(False)
        logo = tk.Frame(s, bg=C["side2"], height=124)
        logo.pack(fill="x"); logo.pack_propagate(False)
        tk.Label(logo, text="🍊", bg=C["side2"], font=("Segoe UI Emoji", 40)).pack(pady=(12, 0))
        tk.Label(logo, text="Car Detector Pro v2", bg=C["side2"], fg=C["white"],
                 font=("Segoe UI", 11, "bold")).pack()

        tk.Label(s, text="▸ Actions", bg=C["sidebar"], fg="#FFD8A0",
                 font=("Segoe UI", 8, "bold")).pack(anchor="w", padx=16, pady=(14, 3))
        self.b_open = RoundBtn(s, "📂 Open Image", cmd=self._browse, bg=C["white"], fg=C["accent"])
        self.b_open.pack(pady=3)
        self.b_det = RoundBtn(s, "🔍 Detect", cmd=self._detect, bg=C["accent"], fg=C["white"], enabled=False)
        self.b_det.pack(pady=3)

        tk.Label(s, text="▸ Method", bg=C["sidebar"], fg="#FFD8A0",
                 font=("Segoe UI", 8, "bold")).pack(anchor="w", padx=16, pady=(20, 3))
        for line in ["• Delta E 2000", "• Gradient Body Zone", "• Specular Removal",
                     "• LAB K-Means", "• Compactness Weight"]:
            tk.Label(s, text=line, bg=C["sidebar"], fg="#FFE8C8",
                     font=("Segoe UI", 8)).pack(anchor="w", padx=20)

        self._msv = tk.StringVar(value="⏳ Loading model...")
        tk.Label(s, textvariable=self._msv, bg=C["sidebar"], fg=C["white"],
                 font=("Segoe UI", 8)).pack(side="bottom", pady=20)

        # Main panel
        f = tk.Frame(self, bg=C["bg"]); f.pack(side="left", fill="both", expand=True, padx=10, pady=10)
        f.columnconfigure(0, weight=60); f.columnconfigure(1, weight=40); f.rowconfigure(0, weight=1)

        self._cv = tk.Canvas(f, bg=C["bg2"], highlightthickness=2, highlightbackground=C["border"])
        self._cv.grid(row=0, column=0, sticky="nsew")

        r = tk.Frame(f, bg=C["bg"]); r.grid(row=0, column=1, sticky="nsew", padx=(10, 0))
        r.rowconfigure(1, weight=1); r.rowconfigure(3, weight=1); r.columnconfigure(0, weight=1)

        self._tree = ttk.Treeview(r, columns=("id","cls","color","car"), show="headings", style="T.Treeview")
        for col, hdr, wid in [("id","#",30),("cls","Class",80),("color","Main Color",120),("car","Car?",50)]:
            self._tree.heading(col, text=hdr); self._tree.column(col, width=wid, anchor="center")
        self._tree.grid(row=1, column=0, sticky="nsew")
        self._tree.bind("<<TreeviewSelect>>", self._on_sel)

        tk.Label(r, text="▸ Color Distribution (click a row)", bg=C["bg"], fg=C["text3"],
                 font=("Segoe UI", 8, "bold")).grid(row=2, column=0, sticky="w", pady=(8, 2))
        self._cc = tk.Canvas(r, bg=C["card"], height=220, highlightthickness=1,
                             highlightbackground=C["border"])
        self._cc.grid(row=3, column=0, sticky="nsew")

    def _load_model(self):
        if not HAS_YOLO:
            self._msv.set("❌ Install ultralytics"); return
        for p in MODEL_PATHS:
            if os.path.exists(p):
                self.model = YOLO(p)
                self.after(0, lambda: self._msv.set(f"✅ Ready: {p}")); return
        self._msv.set("⚠️ No model found")

    def _browse(self):
        p = filedialog.askopenfilename()
        if p:
            self.img_path = p
            self._show(Image.open(p))
            self.b_det.enable()

    def _show(self, pil):
        self._cv.delete("all")
        cw, ch = self._cv.winfo_width(), self._cv.winfo_height()
        if cw < 10: cw, ch = 800, 600
        t = pil.copy(); t.thumbnail((cw, ch), Image.LANCZOS)
        self._tkimg = ImageTk.PhotoImage(t)
        self._cv.create_image(cw//2, ch//2, image=self._tkimg)

    def _detect(self):
        if not self.img_path or not self.model: return
        self.b_det.disable()

        def run():
            img_bgr = cv2.imread(self.img_path)
            res = self.model(img_bgr, verbose=False)[0]
            self.dets = []
            out = img_bgr.copy()
            for box in res.boxes:
                if float(box.conf[0]) < 0.4: continue
                cls  = res.names[int(box.cls[0])]
                xyxy = box.xyxy[0].cpu().numpy()
                x0, y0, x1, y1 = map(int, xyxy)
                is_car = cls.lower() in CAR_CLASSES

                color_info = [(0, "—", "#AAAAAA")]
                if is_car:
                    color_info = kmeans_color(roi_from_xyxy(img_bgr, xyxy))

                d = dict(id=len(self.dets)+1, cls=cls, color=color_info[0][1],
                         hex=color_info[0][2], clusters=color_info,
                         box=[x0,y0,x1,y1], is_car=is_car)
                self.dets.append(d)

                col_draw = (0, 140, 255) if is_car else (128, 128, 128)
                cv2.rectangle(out, (x0,y0), (x1,y1), col_draw, 2)
                cv2.putText(out, d["color"], (x0, y0-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, col_draw, 2)

            self.result_pil = Image.fromarray(cv2.cvtColor(out, cv2.COLOR_BGR2RGB))
            self.after(0, self._done)

        threading.Thread(target=run, daemon=True).start()

    def _done(self):
        self._show(self.result_pil)
        for i in self._tree.get_children(): self._tree.delete(i)
        for d in self.dets:
            self._tree.insert("", "end", iid=str(d["id"]),
                              values=(d["id"], d["cls"], d["color"], "✅" if d["is_car"] else "—"))
        self.b_det.enable()

    def _on_sel(self, _):
        sel = self._tree.selection()
        if not sel: return
        d = next(x for x in self.dets if str(x["id"]) == sel[0])
        self._cdata = d["clusters"]
        self._draw_bars()

    def _draw_bars(self):
        self._cc.delete("all")
        cw = self._cc.winfo_width() or 300
        y = 15
        for pct, name, hexc in self._cdata[:6]:
            bar_w = max(4, int((cw - 130) * pct / 100))
            self._cc.create_rectangle(10, y, 10+bar_w, y+22, fill=hexc, outline=C["border"])
            self._cc.create_text(10+bar_w+8, y+11, text=f"{name}  {pct:.1f}%",
                                  anchor="w", font=("Segoe UI", 9, "bold"), fill=C["text"])
            y += 30
        if not self._cdata:
            self._cc.create_text(cw//2, 80, text="Select a vehicle row above",
                                  font=("Segoe UI", 10), fill=C["text3"])


if __name__ == "__main__":
    App().mainloop()