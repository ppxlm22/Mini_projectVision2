"""
🍊 Car Color Detector
- Standard YOLO (result.boxes)
- K-Means HSV → หาสีหลักเฉพาะ class car
- Cute orange-peach theme
pip install ultralytics opencv-python pillow numpy scikit-learn
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2, numpy as np, threading, os, math
from PIL import Image, ImageTk

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
MODEL_PATHS = [
    "runs/best.pt", "best.pt",
    "runs/detect/train/weights/best.pt",
    "runs/detect/train2/weights/best.pt",
    "weights/best.pt",
]
CAR_CLASSES = {"car", "cars", "automobile", "vehicle", "truck", "van", "bus"}

# ════════════════════════════════════════════
#  HSV → ชื่อสีไทย
# ════════════════════════════════════════════
_COLORS = [
    # H(0-180)  S(0-255)  V(0-255)   ชื่อ               HEX
    (  0,   0, 220,  "ขาว",             "#F5F5F5"),
    (  0,   0, 160,  "เงิน",            "#CFD8DC"),
    (  0,   0, 105,  "เทาอ่อน",         "#9E9E9E"),
    (  0,   0,  55,  "เทาเข้ม",         "#424242"),
    (  0,   0,  15,  "ดำ",             "#111111"),
    (  0,  90,  70,  "แดง",            "#E53935"),
    (170,  90,  65,  "แดงเข้ม",         "#B71C1C"),
    ( 10,  90,  70,  "แดงส้ม",          "#FF3D00"),
    ( 18,  95,  85,  "ส้ม",            "#FF6D00"),
    ( 25,  90,  85,  "เหลืองส้ม",       "#FF9100"),
    ( 30,  90,  75,  "เหลือง",          "#FDD835"),
    ( 22,  30,  92,  "ครีม",            "#FFFDE7"),
    ( 15,  55,  70,  "ทอง",            "#FFD54F"),
    ( 12,  75,  55,  "น้ำตาล",          "#795548"),
    ( 10,  65,  32,  "น้ำตาลเข้ม",      "#4E342E"),
    ( 42,  60,  58,  "เขียวอมเหลือง",   "#AFB42B"),
    ( 55,  65,  58,  "เขียว",           "#43A047"),
    ( 65,  65,  40,  "เขียวเข้ม",       "#1B5E20"),
    ( 88,  70,  58,  "เขียวน้ำเงิน",    "#00897B"),
    ( 97,  75,  62,  "ฟ้าเขียว",        "#00BCD4"),
    (105,  75,  68,  "ฟ้า",            "#29B6F6"),
    (113,  70,  62,  "น้ำเงิน",         "#1565C0"),
    (119,  60,  52,  "น้ำเงินเข้ม",     "#0D47A1"),
    (130,  58,  58,  "ม่วงน้ำเงิน",     "#5C6BC0"),
    (140,  65,  58,  "ม่วง",           "#7B1FA2"),
    (150,  68,  58,  "ม่วงแดง",         "#AD1457"),
]

def _hdist(a, b):
    d = abs(a - b)
    return min(d, 180 - d)

def hsv_name(h, s, v):
    if v < 30:                  return "ดำ",      "#111111"
    if s < 28:
        if v > 200:             return "ขาว",     "#F5F5F5"
        if v > 148:             return "เงิน",    "#CFD8DC"
        if v > 92:              return "เทาอ่อน", "#9E9E9E"
        return                         "เทาเข้ม", "#424242"
    best, bd = _COLORS[0], 1e9
    for row in _COLORS:
        d = math.sqrt((_hdist(h, row[0])/90)**2 +
                      ((s-row[1])/255)**2 +
                      ((v-row[2])/255)**2)
        if d < bd:
            bd, best = d, row
    return best[3], best[4]

def kmeans_color(roi_bgr, k=4):
    """คืน list[(pct, name, hex)] เรียงจากมากสุด"""
    if roi_bgr is None or roi_bgr.size == 0:
        return [(100.0, "ไม่ทราบ", "#999")]
    h, w = roi_bgr.shape[:2]
    scale = min(1.0, 80.0 / max(h, w, 1))
    small = cv2.resize(roi_bgr, (max(1,int(w*scale)), max(1,int(h*scale))),
                       interpolation=cv2.INTER_AREA)
    px = cv2.cvtColor(small, cv2.COLOR_BGR2HSV).reshape(-1, 3).astype(np.float32)
    k  = min(k, len(px))
    if k < 1:
        return [(100.0, "ไม่ทราบ", "#999")]
    if HAS_KM:
        km     = KMeans(n_clusters=k, n_init=8, max_iter=200, random_state=42)
        labels = km.fit_predict(px)
        ctrs   = km.cluster_centers_
        cnts   = np.bincount(labels, minlength=k)
    else:
        ctrs = px.mean(axis=0, keepdims=True)
        cnts = np.array([len(px)])
    total = max(cnts.sum(), 1)
    rows  = []
    for c, n in zip(ctrs, cnts):
        name, hexc = hsv_name(float(c[0]), float(c[1]), float(c[2]))
        rows.append((n/total*100, name, hexc))
    rows.sort(key=lambda x: -x[0])
    return rows

def roi_from_xyxy(img, xyxy):
    x0,y0,x1,y1 = [int(v) for v in xyxy]
    x0,y0 = max(0,x0), max(0,y0)
    x1,y1 = min(x1,img.shape[1]), min(y1,img.shape[0])
    return img[y0:y1, x0:x1]

# ════════════════════════════════════════════
#  PALETTE
# ════════════════════════════════════════════
C = dict(
    bg="#FEF6ED", bg2="#FDEBD4", card="#FFFFFF",
    border="#F5C99A", sidebar="#FF7A1F", side2="#E85D00",
    accent="#FF5722", amber="#FF9800",
    text="#3A1800", text2="#7A4010", text3="#C07030", white="#FFFFFF",
)

# ════════════════════════════════════════════
#  ROUNDED BUTTON
# ════════════════════════════════════════════
class RoundBtn(tk.Canvas):
    def __init__(self, parent, text, cmd=None, w=182, h=38,
                 bg="#FF5722", fg="#FFF", hover="#FF7043",
                 font=("Segoe UI",10,"bold"), r=19, enabled=True):
        super().__init__(parent, width=w, height=h,
                         bg=parent.cget("bg"), highlightthickness=0, bd=0)
        self.cmd=cmd; self.cbg=bg; self.chover=hover; self.cfg=fg
        self.on=enabled; self.dis="#D4B8A0"
        self.rect = self._rr(1,1,w-1,h-1,r, fill=bg if enabled else self.dis)
        self.txt  = self.create_text(w//2,h//2, text=text, fill=fg if enabled else "#B09080",
                                     font=font, anchor="center")
        self.bind("<Enter>",           lambda _: self._c(self.chover) if self.on else None)
        self.bind("<Leave>",           lambda _: self._c(self.cbg)    if self.on else None)
        self.bind("<ButtonPress-1>",   self._dn)
        self.bind("<ButtonRelease-1>", self._up)

    def _rr(self,x1,y1,x2,y2,r,**kw):
        p=[x1+r,y1,x2-r,y1,x2,y1,x2,y1+r,x2,y2-r,x2,y2,
           x2-r,y2,x1+r,y2,x1,y2,x1,y2-r,x1,y1+r,x1,y1]
        return self.create_polygon(p, smooth=True, **kw)

    def _c(self,col): self.itemconfig(self.rect,fill=col)
    def _dn(self,_):
        if self.on: self._c(self.cfg); self.itemconfig(self.txt,fill=self.cbg)
    def _up(self,_):
        if self.on:
            self._c(self.chover); self.itemconfig(self.txt,fill=self.cfg)
            if self.cmd: self.cmd()

    def enable(self):
        self.on=True;  self._c(self.cbg); self.itemconfig(self.txt,fill=self.cfg)
    def disable(self):
        self.on=False; self._c(self.dis);  self.itemconfig(self.txt,fill="#B09080")

# ════════════════════════════════════════════
#  MAIN APP
# ════════════════════════════════════════════
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("🍊 Car Color Detector")
        self.geometry("1320x860")
        self.minsize(1050,720)
        self.configure(bg=C["bg"])
        self.model=None; self.img_path=None
        self.result_pil=None; self.dets=[]; self._tkimg=None; self._cdata=[]
        self._ttk(); self._ui()
        threading.Thread(target=self._load_model, daemon=True).start()

    # ── TTK styles ────────────────────────────────────
    def _ttk(self):
        s=ttk.Style(self); s.theme_use("clam")
        s.configure("T.Treeview",
                    background=C["card"],foreground=C["text"],
                    fieldbackground=C["card"],
                    font=("Segoe UI",10),rowheight=32,borderwidth=0)
        s.configure("T.Treeview.Heading",
                    background=C["sidebar"],foreground=C["white"],
                    font=("Segoe UI",9,"bold"),relief="flat",padding=(4,6))
        s.map("T.Treeview",
              background=[("selected",C["amber"])],
              foreground=[("selected",C["white"])])
        s.configure("TScrollbar",
                    background=C["border"],troughcolor=C["bg2"],
                    arrowcolor=C["sidebar"],relief="flat",borderwidth=0)

    # ── UI ────────────────────────────────────────────
    def _ui(self):
        self._sidebar()
        f=tk.Frame(self,bg=C["bg"])
        f.pack(side="left",fill="both",expand=True,padx=10,pady=10)
        f.columnconfigure(0,weight=55); f.columnconfigure(1,weight=45)
        f.rowconfigure(0,weight=1)
        self._img_panel(f); self._right_panel(f)
        sb=tk.Frame(self,bg=C["border"],height=28)
        sb.pack(side="bottom",fill="x"); sb.pack_propagate(False)
        self._sv=tk.StringVar(value="✨ พร้อมใช้งาน")
        tk.Label(sb,textvariable=self._sv,bg=C["border"],fg=C["text2"],
                 font=("Segoe UI",9)).pack(side="left",padx=12,pady=5)

    # ── Sidebar ───────────────────────────────────────
    def _sidebar(self):
        s=tk.Frame(self,bg=C["sidebar"],width=215)
        s.pack(side="left",fill="y"); s.pack_propagate(False)
        logo=tk.Frame(s,bg=C["side2"],height=124)
        logo.pack(fill="x"); logo.pack_propagate(False)
        tk.Label(logo,text="🍊",bg=C["side2"],font=("Segoe UI Emoji",40)).pack(pady=(12,0))
        tk.Label(logo,text="Car Color Detector",bg=C["side2"],fg=C["white"],
                 font=("Segoe UI",11,"bold")).pack()
        def sec(t):
            tk.Label(s,text=t,bg=C["sidebar"],fg="#FFD8A0",
                     font=("Segoe UI",8,"bold")).pack(anchor="w",padx=16,pady=(14,3))
        sec("▸  การทำงาน")
        self.b_open  = RoundBtn(s,"📂  เปิดภาพ",  cmd=self._browse,
                                 bg=C["white"],fg=C["accent"],hover=C["bg2"])
        self.b_open.pack(pady=3)
        self.b_det   = RoundBtn(s,"🔍  ตรวจจับ",  cmd=self._detect,
                                 bg=C["accent"],fg=C["white"],hover="#FF7043",enabled=False)
        self.b_det.pack(pady=3)
        self.b_save  = RoundBtn(s,"💾  บันทึกผล", cmd=self._save,
                                 bg=C["amber"],fg=C["white"],hover="#FFA000")
        self.b_save.pack(pady=3)
        self.b_clear = RoundBtn(s,"🔄  ล้างหน้าจอ",cmd=self._clear,
                                 bg=C["side2"],fg=C["white"],hover=C["accent"])
        self.b_clear.pack(pady=3)
        tk.Frame(s,bg=C["side2"],height=1).pack(fill="x",padx=16,pady=12)
        sec("▸  สถานะโมเดล")
        self._msv=tk.StringVar(value="⏳ กำลังโหลด...")
        self._mlb=tk.Label(s,textvariable=self._msv,bg=C["sidebar"],fg=C["white"],
                           font=("Segoe UI",8),wraplength=188,justify="left")
        self._mlb.pack(anchor="w",padx=16)
        tk.Frame(s,bg=C["side2"],height=1).pack(fill="x",padx=16,pady=12)
        sec("▸  ไฟล์ที่เลือก")
        self._fsv=tk.StringVar(value="—")
        tk.Label(s,textvariable=self._fsv,bg=C["sidebar"],fg=C["white"],
                 font=("Segoe UI",8),wraplength=188,justify="left").pack(anchor="w",padx=16)
        tk.Label(s,text="YOLO  •  K-Means HSV",bg=C["sidebar"],fg=C["side2"],
                 font=("Segoe UI",7)).pack(side="bottom",pady=10)

    # ── Image panel ───────────────────────────────────
    def _img_panel(self,parent):
        w=tk.Frame(parent,bg=C["bg"])
        w.grid(row=0,column=0,sticky="nsew",padx=(0,8))
        w.rowconfigure(1,weight=1); w.columnconfigure(0,weight=1)
        tk.Label(w,text="🖼  ภาพและผลการตรวจจับ",bg=C["bg"],fg=C["text"],
                 font=("Segoe UI",13,"bold")).grid(row=0,column=0,sticky="w",pady=(0,6))
        card=tk.Frame(w,bg=C["card"],highlightthickness=2,highlightbackground=C["border"])
        card.grid(row=1,column=0,sticky="nsew")
        card.rowconfigure(0,weight=1); card.columnconfigure(0,weight=1)
        self._cv=tk.Canvas(card,bg=C["bg2"],highlightthickness=0,cursor="crosshair")
        self._cv.grid(sticky="nsew")
        self._cv.bind("<Configure>",self._on_resize)
        self._placeholder()

    def _placeholder(self):
        self._cv.delete("all")
        cw=max(self._cv.winfo_width(),400); ch=max(self._cv.winfo_height(),300)
        self._cv.create_rectangle(22,22,cw-22,ch-22,outline=C["border"],width=2,dash=(8,5))
        self._cv.create_text(cw//2,ch//2-28,text="🚗",font=("Segoe UI Emoji",54),fill=C["border"])
        self._cv.create_text(cw//2,ch//2+36,text="คลิก  📂 เปิดภาพ  เพื่อเริ่มต้น",
                             font=("Segoe UI",12),fill=C["text3"])

    def _show(self,pil):
        self._cv.delete("all")
        self.update_idletasks()
        cw=max(self._cv.winfo_width(),100); ch=max(self._cv.winfo_height(),100)
        t=pil.copy(); t.thumbnail((cw,ch),Image.LANCZOS)
        self._tkimg=ImageTk.PhotoImage(t)
        self._cv.create_image(cw//2,ch//2,anchor="center",image=self._tkimg)

    # ── Right panel ───────────────────────────────────
    def _right_panel(self,parent):
        w=tk.Frame(parent,bg=C["bg"])
        w.grid(row=0,column=1,sticky="nsew")
        w.rowconfigure(1,weight=2); w.rowconfigure(3,weight=3); w.columnconfigure(0,weight=1)

        tk.Label(w,text="📋  รายการตรวจจับ (เฉพาะ Cars)",
                 bg=C["bg"],fg=C["text"],
                 font=("Segoe UI",12,"bold")).grid(row=0,column=0,sticky="w",pady=(0,4))

        tc=tk.Frame(w,bg=C["card"],highlightthickness=2,highlightbackground=C["border"])
        tc.grid(row=1,column=0,sticky="nsew",pady=(0,10))
        tc.rowconfigure(0,weight=1); tc.columnconfigure(0,weight=1)

        cols=("id","cls","conf","color","pct")
        self._tree=ttk.Treeview(tc,columns=cols,show="headings",
                                 style="T.Treeview",selectmode="browse")
        for col,head,wd,anc in [("id","#",32,"center"),("cls","คลาส",80,"center"),
                                  ("conf","Conf",56,"center"),("color","สีหลัก",118,"w"),
                                  ("pct","% พท.",65,"center")]:
            self._tree.heading(col,text=head,anchor="center")
            self._tree.column(col,width=wd,anchor=anc,stretch=(col=="color"))
        self._tree.grid(row=0,column=0,sticky="nsew")
        sb=ttk.Scrollbar(tc,orient="vertical",command=self._tree.yview)
        sb.grid(row=0,column=1,sticky="ns")
        self._tree.configure(yscrollcommand=sb.set)
        self._tree.bind("<<TreeviewSelect>>",self._on_sel)

        tk.Label(w,text="🎨  K-Means สีรถยนต์",bg=C["bg"],fg=C["text"],
                 font=("Segoe UI",12,"bold")).grid(row=2,column=0,sticky="w",pady=(0,4))

        cc=tk.Frame(w,bg=C["card"],highlightthickness=2,highlightbackground=C["border"])
        cc.grid(row=3,column=0,sticky="nsew")
        cc.rowconfigure(0,weight=1); cc.columnconfigure(0,weight=1)
        self._cc=tk.Canvas(cc,bg=C["card"],highlightthickness=0)
        self._cc.grid(sticky="nsew",padx=8,pady=8)
        self._cc.bind("<Configure>",lambda _:self._draw_bars())
        self._hint()

    def _hint(self):
        self._cc.delete("all")
        cw=max(self._cc.winfo_width(),200); ch=max(self._cc.winfo_height(),100)
        self._cc.create_text(cw//2,ch//2,text="🎨  เลือกรายการรถในตารางเพื่อดูสี",
                             font=("Segoe UI",11),fill=C["text3"])

    def _draw_bars(self):
        if not self._cdata: self._hint(); return
        self._cc.delete("all")
        cw=max(self._cc.winfo_width(),220)
        px,py=14,10; bh,gap=30,8
        self._cc.create_text(px,py,text="สัดส่วนสี K-Means (HSV)",
                             font=("Segoe UI",9,"bold"),fill=C["text2"],anchor="nw")
        mw=cw-px*2-96; y=py+28
        for pct,name,hexc in self._cdata:
            bw=max(6,int(pct/100*mw))
            r=bh//2-3; cx=px+r+2; cy=y+bh//2
            self._cc.create_oval(cx-r,cy-r,cx+r,cy+r,fill=hexc,outline=C["border"],width=1.5)
            bx=px+bh+8
            self._cc.create_rectangle(bx,y+7,bx+mw,y+bh-7,fill=C["bg2"],outline="")
            self._cc.create_rectangle(bx,y+7,bx+bw,y+bh-7,fill=hexc,outline="")
            self._cc.create_text(bx+bw+8,y+bh//2,
                                 text=f"{name}  {pct:.1f}%",
                                 font=("Segoe UI",9,"bold"),fill=C["text"],anchor="w")
            y+=bh+gap

    # ── Model ─────────────────────────────────────────
    def _load_model(self):
        if not HAS_YOLO:
            self.after(0,lambda:(self._msv.set("⚠️ pip install ultralytics"),
                                 self._mlb.config(fg="#FFD0A0"))); return
        for p in MODEL_PATHS:
            if os.path.exists(p):
                try:
                    self.model=YOLO(p)
                    self.after(0,lambda pp=p:(self._msv.set(f"✅ {pp}"),
                                              self._mlb.config(fg="#AAFFBB")))
                    return
                except Exception as e:
                    self.after(0,lambda e=e:(self._msv.set(f"❌ {e}"),
                                             self._mlb.config(fg="#FFAAAA"))); return
        self.after(0,lambda:(self._msv.set("⚠️ ไม่พบ best.pt"),
                             self._mlb.config(fg="#FFD0A0")))

    # ── Browse ────────────────────────────────────────
    def _browse(self):
        p=filedialog.askopenfilename(title="เลือกภาพ",
            filetypes=[("Image","*.jpg *.jpeg *.png *.bmp *.webp"),("All","*.*")])
        if p: self._load_img(p)

    def _load_img(self,path):
        self.img_path=path; self.result_pil=None; self.dets=[]; self._cdata=[]
        for i in self._tree.get_children(): self._tree.delete(i)
        self._hint()
        img=Image.open(path); w,h=img.size
        self._fsv.set(f"{os.path.basename(path)}\n{w}×{h} px")
        self._show(img); self.b_det.enable()
        self._sv.set(f"📂 โหลด: {os.path.basename(path)}")

    # ── Detect ────────────────────────────────────────
    def _detect(self):
        if not self.img_path: return
        if self.model is None:
            messagebox.showwarning("โมเดล","โมเดลยังไม่พร้อม"); return
        self.b_det.disable(); self._sv.set("🔄 กำลังประมวลผล...")
        threading.Thread(target=self._thread,daemon=True).start()

    def _thread(self):
        try:
            img_bgr=cv2.imread(self.img_path)
            results=self.model(self.img_path,verbose=False)
            dets=[]
            for result in results:
                names=result.names
                boxes=result.boxes
                if boxes is None or len(boxes)==0: continue
                for box in boxes:
                    conf    =float(box.conf[0])
                    cls_id  =int(box.cls[0])
                    cls_name=names.get(cls_id,str(cls_id))
                    is_car  =cls_name.lower() in CAR_CLASSES
                    xyxy    =box.xyxy[0].cpu().numpy()
                    x0,y0,x1,y1=[int(v) for v in xyxy]
                    roi=roi_from_xyxy(img_bgr,xyxy)
                    if is_car:
                        clusters=kmeans_color(roi,k=4)
                        dom_name=clusters[0][1]; dom_hex=clusters[0][2]
                    else:
                        clusters=[]; dom_name="—"; dom_hex="#AAA"
                    area=(x1-x0)*(y1-y0)
                    tot =img_bgr.shape[0]*img_bgr.shape[1]
                    pct =area/tot*100 if tot else 0
                    dets.append(dict(id=len(dets)+1,cls=cls_name,conf=conf,
                                     color=dom_name,hex=dom_hex,pct=pct,
                                     clusters=clusters,
                                     x0=x0,y0=y0,x1=x1,y1=y1,is_car=is_car))

            # วาดกรอบบนภาพ
            out=img_bgr.copy()
            for d in dets: self._draw(out,d)
            self.result_pil=Image.fromarray(cv2.cvtColor(out,cv2.COLOR_BGR2RGB))
            self.dets=dets
            self.after(0,self._done)
        except Exception as e:
            import traceback; traceback.print_exc()
            self.after(0,lambda:(self._sv.set(f"❌ {e}"),
                                 self.b_det.enable(),
                                 messagebox.showerror("Error",str(e))))

    def _draw(self,img,d):
        x0,y0,x1,y1=d["x0"],d["y0"],d["x1"],d["y1"]
        col=(20,130,255) if d["is_car"] else (170,170,170)
        label=f"{d['cls']} {d['conf']:.2f}"
        if d["is_car"]: label+=f"  {d['color']}"
        cv2.rectangle(img,(x0,y0),(x1,y1),col,2)
        (tw,th),_=cv2.getTextSize(label,cv2.FONT_HERSHEY_SIMPLEX,0.52,1)
        ly=max(y0,th+8)
        cv2.rectangle(img,(x0,ly-th-8),(x0+tw+8,ly),col,-1)
        cv2.putText(img,label,(x0+4,ly-4),cv2.FONT_HERSHEY_SIMPLEX,
                    0.52,(255,255,255),1,cv2.LINE_AA)
        # วงกลมสีที่มุม
        if d["is_car"] and d["hex"] not in ("#AAA","#999"):
            try:
                h=d["hex"].lstrip("#")
                r2,g2,b2=int(h[0:2],16),int(h[2:4],16),int(h[4:6],16)
                cv2.circle(img,(x1-12,y0+12),9,(b2,g2,r2),-1)
                cv2.circle(img,(x1-12,y0+12),9,(255,255,255),1)
            except: pass

    def _done(self):
        for i in self._tree.get_children(): self._tree.delete(i)
        for d in self.dets:
            tag="car" if d["is_car"] else "other"
            self._tree.insert("","end",iid=str(d["id"]),
                              values=(d["id"],d["cls"],f"{d['conf']:.2f}",
                                      d["color"],f"{d['pct']:.1f}%"),tags=(tag,))
        self._tree.tag_configure("car",  background="#FFF3E0")
        self._tree.tag_configure("other",background=C["card"])
        if self.result_pil: self._show(self.result_pil.copy())
        cars=sum(1 for d in self.dets if d["is_car"])
        self._sv.set(f"✅ พบ {len(self.dets)} วัตถุ — รถยนต์ {cars} คัน")
        self.b_det.enable()

    # ── Tree select ───────────────────────────────────
    def _on_sel(self,_):
        sel=self._tree.selection()
        if not sel: return
        iid=int(sel[0])
        d=next((x for x in self.dets if x["id"]==iid),None)
        if not d or not d["is_car"] or not d["clusters"]:
            self._cdata=[]; self._hint(); return
        self._cdata=list(d["clusters"])
        self._draw_bars()

    # ── Save / Clear ──────────────────────────────────
    def _save(self):
        if self.result_pil is None:
            messagebox.showinfo("บันทึก","ยังไม่มีผลลัพธ์"); return
        p=filedialog.asksaveasfilename(defaultextension=".jpg",
            filetypes=[("JPEG","*.jpg"),("PNG","*.png")])
        if p: self.result_pil.save(p); self._sv.set(f"💾 บันทึก: {os.path.basename(p)}")

    def _clear(self):
        self.img_path=None; self.result_pil=None
        self.dets=[]; self._cdata=[]
        for i in self._tree.get_children(): self._tree.delete(i)
        self._placeholder(); self._hint()
        self._fsv.set("—"); self.b_det.disable()
        self._sv.set("🔄 ล้างหน้าจอแล้ว")

    def _on_resize(self,_):
        if self.result_pil:    self._show(self.result_pil.copy())
        elif self.img_path:    self._show(Image.open(self.img_path))
        else:                  self._placeholder()

if __name__=="__main__":
    App().mainloop()