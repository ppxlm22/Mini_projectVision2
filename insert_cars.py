from bing_image_downloader import downloader

brands = {
    "Toyota": "Toyota Fortuner Thailand road",
    "Honda": "Honda Civic Thailand",
    "Isuzu": "Isuzu D-Max Thailand", 
    "Mitsubishi": "Mitsubishi Pajero Sport",
    "Ford": "Ford Ranger Raptor",
    "MG": "MG 5 Thailand",
}
for brand, query in brands.items():
    print(f"Downloading {brand}...")
    downloader.download(
        query, 
        limit=300,  
        output_dir='dataset_cars', 
        adult_filter_off=True, 
        force_replace=False, 
        timeout=60
    )