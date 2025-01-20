import os
root_dir = r"C:\Users\saman\Desktop\Disaster project\WG_LS_ Data"
raster_files = {
    'aspect': os.path.join(root_dir, "Aspect", "Aspect_Final.tif"),
    'curvature': os.path.join(root_dir, "Curvature", "Curvature_Final.tif"),
    'dem': os.path.join(root_dir, "DEM", "DEM_Final.tif"),
    'distance_to_road': os.path.join(root_dir, "Distance from road", "Distance_To_Road.tif"),
    'distance_to_settlements': os.path.join(root_dir, "Distance from Settlements", "Dist_To_Settlements_Final.tif"),
    'distance_to_stream': os.path.join(root_dir, "Distance from Stream", "Distance_To_Stream.tif"),
    'lulc': os.path.join(root_dir, "LULC", "LULC_Final.tif"),
    'ndvi': os.path.join(root_dir, "NDVI", "NDVI_final.tif"),
    'rainfall': os.path.join(root_dir, "Rainfall", "Max_Rainfall.tif"),
    'slope': os.path.join(root_dir, "Slope", "Slope_Final.tif"),
    'soil': os.path.join(root_dir, "Soil", "Soil_Final.tif"),
    'twi': os.path.join(root_dir, "TWI", "TWI_Final.tif")
}

for key in raster_files.keys():
    print(key, end=", ")