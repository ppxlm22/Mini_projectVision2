# main.py
import tkinter as tk
from tkinter import ttk, filedialog
import cv2, threading, os
from PIL import Image, ImageTk
from config.config import C, MODEL_PATHS, CAR_CLASSES
from process.processor import kmeans_color, roi_from_xyxy

try:
    from ultralytics import YOLO
    HAS_YOLO = True
except ImportError:
    HAS_YOLO = False

class RoundBtn(tk.Canvas):
    def __init__(self, parent, text, cmd=None, w=182, h=38, bg="#FF5722", fg="#FFF", hover="#FF7043", font=("Segoe UI", 10, "bold"), r=19, enabled=True):
        super().__init__(parent, width=w, height=h, bg=parent.cget("bg"), highlightthickness=0, bd=0)
        self.cmd = cmd; self.cbg = bg; self.chover = hover; self.cfg = fg; self.on = enabled; self.dis = "#D4B8A0"
        self.rect = self._rr(1, 1, w-1, h-1, r, fill=bg if enabled else self.dis)
        self.txt  = self.create_text(w//2, h//2, text=text, fill=fg if enabled else "#B09080", font=font, anchor="center")
        self.bind("<Enter>", lambda _: self.itemconfig(self.rect, fill=self.chover) if self.on else None)
        self.bind("<Leave>", lambda _: self.itemconfig(self.rect, fill=self.cbg) if self.on else None)
        self.bind("<ButtonPress-1>", self._dn); self.bind("<ButtonRelease-1>", self._up)

    def _rr(self, x1, y1, x2, y2, r, **kw):
        p = [x1+r,y1, x2-r,y1, x2,y1, x2,y1+r, x2,y2-r, x2,y2, x2-r,y2, x1+r,y2, x1,y2, x1,y2-r, x1,y1+r, x1,y1]
        return self.create_polygon(p, smooth=True, **kw)
    def _dn(self, _):
        if self.on: self.itemconfig(self.rect, fill=self.cfg); self.itemconfig(self.txt, fill=self.cbg)
    def _up(self, _):
        if self.on: 
            self.itemconfig(self.rect, fill=self.chover); self.itemconfig(self.txt, fill=self.cfg)
            if self.cmd: self.cmd()
    def enable(self): self.on = True; self.itemconfig(self.rect, fill=self.cbg); self.itemconfig(self.txt, fill=self.cfg)
    def disable(self): self.on = False; self.itemconfig(self.rect, fill=self.dis); self.itemconfig(self.txt, fill="#B09080")

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("🍊 Car Color Detector — Precision v2")
        self.geometry("1320x860"); self.configure(bg=C["bg"])
        self.model = None; self.img_path = None; self.dets = []; self._tkimg = None; self._cdata = []
        self._ttk(); self._ui()
        threading.Thread(target=self._load_model, daemon=True).start()

    def _ttk(self):
        s = ttk.Style(self); s.theme_use("clam")
        s.configure("T.Treeview", background=C["card"], foreground=C["text"], fieldbackground=C["card"], rowheight=32)
        s.configure("T.Treeview.Heading", background=C["sidebar"], foreground=C["white"], font=("Segoe UI", 9, "bold"))

    def _ui(self):
        sidebar = tk.Frame(self, bg=C["sidebar"], width=215); sidebar.pack(side="left", fill="y"); sidebar.pack_propagate(False)
        logo = tk.Frame(sidebar, bg=C["side2"], height=124); logo.pack(fill="x")
        tk.Label(logo, text="🍊", bg=C["side2"], font=("Segoe UI Emoji", 40)).pack(pady=(12,0))
        tk.Label(logo, text="Car Detector Pro v2", bg=C["side2"], fg="white", font=("Segoe UI", 11, "bold")).pack()
        
        self.b_open = RoundBtn(sidebar, "📂 Open Image", cmd=self._browse, bg="white", fg=C["accent"]); self.b_open.pack(pady=10)
        self.b_det = RoundBtn(sidebar, "🔍 Detect", cmd=self._detect, bg=C["accent"], fg="white", enabled=False); self.b_det.pack()

        main = tk.Frame(self, bg=C["bg"]); main.pack(side="left", fill="both", expand=True, padx=10, pady=10)
        main.columnconfigure(0, weight=60); main.columnconfigure(1, weight=40); main.rowconfigure(0, weight=1)
        self._cv = tk.Canvas(main, bg=C["bg2"], highlightthickness=2, highlightbackground=C["border"]); self._cv.grid(row=0, column=0, sticky="nsew")

        right = tk.Frame(main, bg=C["bg"]); right.grid(row=0, column=1, sticky="nsew", padx=(10,0))
        self._tree = ttk.Treeview(right, columns=("id","cls","color","car"), show="headings")
        for col, hdr in [("id","#"),("cls","Class"),("color","Color"),("car","Car?")]:
            self._tree.heading(col, text=hdr); self._tree.column(col, width=50, anchor="center")
        self._tree.pack(fill="both", expand=True); self._tree.bind("<<TreeviewSelect>>", self._on_sel)

        self._cc = tk.Canvas(right, bg=C["card"], height=220, highlightthickness=1); self._cc.pack(fill="x", pady=10)
        self._msv = tk.StringVar(value="⏳ Loading..."); tk.Label(sidebar, textvariable=self._msv, bg=C["sidebar"], fg="white").pack(side="bottom", pady=20)

    def _load_model(self):
        for p in MODEL_PATHS:
            if os.path.exists(p): self.model = YOLO(p); self.after(0, lambda: self._msv.set(f"✅ Ready: {p}")); return
        self._msv.set("⚠️ No model found")

    def _browse(self):
        p = filedialog.askopenfilename(initialdir=os.path.join(os.getcwd(), "images"))
        if p: self.img_path = p; self._show(Image.open(p)); self.b_det.enable()

    def _show(self, pil):
        self._cv.delete("all"); cw, ch = self._cv.winfo_width() or 800, self._cv.winfo_height() or 600
        t = pil.copy(); t.thumbnail((cw, ch), Image.LANCZOS); self._tkimg = ImageTk.PhotoImage(t)
        self._cv.create_image(cw//2, ch//2, image=self._tkimg)

    def _detect(self):
        if not self.model: return
        self.b_det.disable()
        
        def run():
            img = cv2.imread(self.img_path)
            res = self.model(img, verbose=False)[0]
            self.dets = []
            out = img.copy()
            
            for box in res.boxes:
                conf = float(box.conf[0])
                if conf < 0.65: continue
                
                xyxy = box.xyxy[0].cpu().numpy()
                cls = res.names[int(box.cls[0])]
                is_car = cls.lower() in CAR_CLASSES
                
                color_info = kmeans_color(roi_from_xyxy(img, xyxy)) if is_car else [(0, "—", "#AAAAAA")]
                
                d = dict(
                    id=len(self.dets)+1, 
                    cls=cls, 
                    conf=f"{conf:.2f}", 
                    color=color_info[0][1], 
                    hex=color_info[0][2], 
                    clusters=color_info, 
                    box=xyxy, 
                    is_car=is_car
                )
                self.dets.append(d)
                
                col_draw = (0, 140, 255) if is_car else (100, 255, 128)
                x0, y0, x1, y1 = map(int, xyxy)
                
                cv2.rectangle(out, (x0, y0), (x1, y1), col_draw, 2)
                
                label = f"#{d['id']} {cls} {conf:.2f}"
                
                (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(out, (x0, y0 - h - 15), (x0 + w, y0), col_draw, -1) 
                
                cv2.putText(out, label, (x0, y0 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

            self.result_pil = Image.fromarray(cv2.cvtColor(out, cv2.COLOR_BGR2RGB))
            self.after(0, self._done)
            
        threading.Thread(target=run, daemon=True).start()

    def _done(self):
        self._show(self.result_pil)
        for i in self._tree.get_children(): self._tree.delete(i)
        for d in self.dets: self._tree.insert("", "end", iid=str(d["id"]), values=(d["id"], d["cls"], d["color"], "✅" if d["is_car"] else "—"))
        self.b_det.enable()

    def _on_sel(self, _):
        sel = self._tree.selection()
        if sel: d = next(x for x in self.dets if str(x["id"]) == sel[0]); self._cdata = d["clusters"]; self._draw_bars()

    def _draw_bars(self):
        self._cc.delete("all"); y = 15; cw = self._cc.winfo_width() or 300
        for pct, name, hexc in self._cdata[:6]:
            bw = max(4, int((cw - 130) * pct / 100))
            self._cc.create_rectangle(10, y, 10+bw, y+22, fill=hexc, outline=C["border"])
            self._cc.create_text(10+bw+8, y+11, text=f"{name} {pct:.1f}%", anchor="w", font=("Segoe UI", 9, "bold"))
            y += 30

if __name__ == "__main__":
    App().mainloop()