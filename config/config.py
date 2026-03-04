# config/config.py
MODEL_PATHS = ["runs/best.pt", "best.pt"]
CAR_CLASSES  = {"car", "cars", "automobile", "vehicle", "truck", "van", "bus"}

# สีสำหรับ UI
C = dict(
    bg="#FEF6ED", bg2="#FDEBD4", card="#FFFFFF",
    border="#F5C99A", sidebar="#FF7A1F", side2="#E85D00",
    accent="#FF5722", amber="#FF9800",
    text="#3A1800", text2="#7A4010", text3="#C07030", white="#FFFFFF",
)

_PALETTE = {
    "Pearl White": [250, 248, 245], "White": [240, 240, 240],
    "Silver": [190, 190, 195], "Light Gray": [160, 162, 165],
    "Gray": [110, 112, 115], "Dark Gray": [70, 72, 75],
    "Black": [25, 25, 28], "Red": [185, 15, 15],
    "Dark Red": [120, 10, 10], "Metallic Red": [160, 30, 40],
    "Bordeaux": [100, 0, 30], "Dark Blue": [10, 30, 100],
    "Blue": [20, 60, 160], "Midnight Blue": [15, 20, 70],
    "Sky Blue": [100, 170, 220], "Dark Green": [10, 60, 30],
    "Green": [30, 120, 50], "Olive/Military": [60, 80, 50],
    "Mint Green": [120, 200, 170], "Yellow": [255, 215, 0],
    "Gold": [200, 155, 20], "Orange": [230, 95, 10],
    "Brown": [100, 55, 25], "Tan": [175, 130, 85],
    "Cream": [240, 235, 200], "Beige": [210, 195, 165],
    "Purple": [90, 20, 110], "Metallic Purple": [120, 60, 140],
    "Pink": [220, 120, 150],
}