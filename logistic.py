import os
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.transform import from_origin
from shapely.geometry import Point
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
from rasterio.enums import Resampling
from rasterio.windows import Window

# Set the root directory
root_dir = r"C:\Users\saman\Desktop\Disaster project\WG_LS_ Data"

# Function to read and align rasters with improved resampling
def read_and_align_raster(file_path, reference_profile, resampling_method=Resampling.cubic_spline):
    with rasterio.open(file_path) as src:
        transform, width, height = calculate_default_transform(
            src.crs, reference_profile['crs'], src.width, src.height, *src.bounds)
        kwargs = src.meta.copy()
        kwargs.update({
            'crs': reference_profile['crs'],
            'transform': transform,
            'width': width,
            'height': height
        })
        
        destination = np.zeros((height, width), dtype=np.float32)
        reproject(
            source=rasterio.band(src, 1),
            destination=destination,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=transform,
            dst_crs=reference_profile['crs'],
            resampling=resampling_method)
    return destination, transform

# Read the landslide occurrence data
landslides = gpd.read_file(os.path.join(root_dir, "LS_WG1 Data", "WG_combined.shp"))

# Create a binary target variable (1 for landslide, 0 for no landslide)
landslides['landslide'] = 1

# Read and align raster data
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

# Use DEM as the reference raster for alignment
with rasterio.open(raster_files['dem']) as reference_raster:
    reference_profile = reference_raster.profile

aligned_rasters = {}
for name, path in raster_files.items():
    resampling_method = Resampling.nearest if name in ['lulc', 'soil'] else Resampling.cubic_spline
    try:
        aligned_rasters[name], _ = read_and_align_raster(path, reference_profile, resampling_method)
    except Exception as e:
        print(f"Error processing {name}: {str(e)}. Skipping this layer.")

# Ensure all rasters have the same shape without resizing
target_shape = aligned_rasters['dem'].shape
aligned_rasters_filtered = {}
for name, raster in aligned_rasters.items():
    if raster.shape == target_shape:
        aligned_rasters_filtered[name] = raster
    else:
        print(f"Warning: {name} raster shape mismatch. Skipping this layer.")

aligned_rasters = aligned_rasters_filtered

# Transform landslide coordinates to pixel coordinates
def world_to_pixel(x, y, transform):
    return ~transform * (x, y)

pixel_coords = [world_to_pixel(x, y, reference_profile['transform']) for x, y in zip(landslides.geometry.x, landslides.geometry.y)]

# Extract raster values at landslide locations
X = pd.DataFrame()
for name, raster in aligned_rasters.items():
    X[name] = [raster[int(y), int(x)] if 0 <= int(y) < raster.shape[0] and 0 <= int(x) < raster.shape[1] else np.nan for x, y in pixel_coords]

# Add landslide occurrence data
X['landslide'] = 1

# Generate non-landslide points with increased sample size for better balance
n_samples = len(landslides) * 2  # Increase the number of non-landslide points
non_landslide_points = gpd.GeoDataFrame(
    geometry=[Point(x, y) for x, y in zip(
        np.random.uniform(reference_profile['transform'][2], reference_profile['transform'][2] + reference_profile['width'] * reference_profile['transform'][0], n_samples),
        np.random.uniform(reference_profile['transform'][5] + reference_profile['height'] * reference_profile['transform'][4], reference_profile['transform'][5], n_samples)
    )],
    crs=reference_profile['crs']
)

# Transform non-landslide coordinates to pixel coordinates
non_landslide_pixel_coords = [world_to_pixel(x, y, reference_profile['transform']) for x, y in zip(non_landslide_points.geometry.x, non_landslide_points.geometry.y)]

# Extract raster values at non-landslide locations
non_landslide_X = pd.DataFrame()
for name, raster in aligned_rasters.items():
    non_landslide_X[name] = [raster[int(y), int(x)] if 0 <= int(y) < raster.shape[0] and 0 <= int(x) < raster.shape[1] else np.nan for x, y in non_landslide_pixel_coords]

# Add non-landslide occurrence data
non_landslide_X['landslide'] = 0

# Combine landslide and non-landslide data
X = pd.concat([X, non_landslide_X], ignore_index=True)

# Remove rows with NaN values
X = X.dropna()

# Split features and target
y = X['landslide']
X = X.drop('landslide', axis=1)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train Logistic Regression model
# Using balanced class weights and increased max_iter
lr_model = LogisticRegression(
    class_weight='balanced',
    max_iter=1000,
    random_state=42,
    n_jobs=-1  # Use all available CPU cores
)
lr_model.fit(X_train_scaled, y_train)

# Make predictions
y_pred = lr_model.predict(X_test_scaled)

# Print classification report and confusion matrix
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Get feature importance from logistic regression coefficients
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': np.abs(lr_model.coef_[0])  # Take absolute values for importance
})
feature_importance = feature_importance.sort_values('importance', ascending=False)
print("\nFeature Importance:")
print(feature_importance)

# Plot feature importance
plt.figure(figsize=(10, 6))
plt.bar(feature_importance['feature'], feature_importance['importance'])
plt.xticks(rotation=90)
plt.title('Feature Importance (Logistic Regression Coefficients)')
plt.tight_layout()
plt.savefig(os.path.join(root_dir, 'feature_importance_lr.png'))

# Function to process raster chunks to avoid memory issues
def process_raster_chunks(rasters, model, scaler, chunk_size=1000):
    height, width = rasters[list(rasters.keys())[0]].shape
    susceptibility_map = np.full((height, width), np.nan)

    for row in range(0, height, chunk_size):
        row_end = min(row + chunk_size, height)
        chunk_data = np.stack([rasters[name][row:row_end, :].reshape(-1) for name in X.columns]).T
        
        # Remove rows with NaN values
        valid_indices = ~np.isnan(chunk_data).any(axis=1)
        chunk_data = chunk_data[valid_indices]
        
        if chunk_data.size > 0:
            # Scale the input data
            chunk_data_scaled = scaler.transform(chunk_data)
            
            # Make predictions
            chunk_susceptibility = model.predict_proba(chunk_data_scaled)[:, 1]
            
            # Reshape the chunk susceptibility
            chunk_result = np.full((row_end - row, width), np.nan)
            chunk_result.flat[valid_indices] = chunk_susceptibility
            
            # Update the susceptibility map
            susceptibility_map[row:row_end, :] = chunk_result

    return susceptibility_map

# Process the raster in chunks
print("\nGenerating susceptibility map...")
susceptibility_map = process_raster_chunks(aligned_rasters, lr_model, scaler)

# Save the susceptibility map
output_file = os.path.join(root_dir, 'landslide_susceptibility_lr.tif')
profile = reference_profile.copy()
profile.update(dtype=rasterio.float32, count=1, compress='lzw')

with rasterio.open(output_file, 'w', **profile) as dst:
    dst.write(susceptibility_map.astype(rasterio.float32), 1)

print(f"Landslide susceptibility map saved to: {output_file}")

# Visualize the susceptibility map
plt.figure(figsize=(12, 8))
plt.imshow(susceptibility_map, cmap='YlOrRd', vmin=0, vmax=1)
plt.colorbar(label='Susceptibility')
plt.title('Landslide Susceptibility Map (Logistic Regression)')
plt.axis('off')
plt.tight_layout()
plt.savefig(os.path.join(root_dir, 'landslide_susceptibility_map_lr.png'), dpi=300, bbox_inches='tight')
plt.close()

print("Landslide susceptibility map visualization saved.")