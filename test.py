import rasterio
pmtiles_filename = 'nsw_roads.pmtiles'
with rasterio.open(pmtiles_filename) as src:
    # Read specific bands or attributes that might contain road information
    road_data = src.read()  # Example: Read all data (adjust as per your specific needs)
    metadata = src.meta
    print(road_data)