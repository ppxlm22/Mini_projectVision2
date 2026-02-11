from bing_image_downloader import downloader

# รายชื่อที่เจาะจงรุ่นรถในไทย เพื่อความแม่นยำ
brands = {
    "Toyota": "Toyota Fortuner Thailand road", # ระบุรุ่นยอดฮิต
    "Honda": "Honda Civic Thailand",
    "Isuzu": "Isuzu D-Max Thailand", # ใส่ Thailand เพื่อให้ได้ฉากถนนไทย
    "Mitsubishi": "Mitsubishi Pajero Sport",
    "Mazda": "Mazda 2 Thailand",
    "Nissan": "Nissan Almera Thailand",
    "Ford": "Ford Ranger Raptor",
    "MG": "MG 5 Thailand",
    "Suzuki": "Suzuki Swift Thailand",
    "BMW": "BMW Series 3 on road"
}

for brand, query in brands.items():
    print(f"Downloading {brand}...")
    downloader.download(
        query, 
        limit=300,  # จำนวนรูปที่ต้องการ
        output_dir='dataset_cars', 
        adult_filter_off=True, 
        force_replace=False, 
        timeout=60
    )