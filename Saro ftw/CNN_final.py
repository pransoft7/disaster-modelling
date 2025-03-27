import os
import numpy as np
import geopandas as gpd
import rasterio
import tensorflow as tf
import tifffile
from rasterio.warp import Resampling, reproject
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, Flatten, Dense, Dropout, MaxPooling1D
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

# Set root directory
root_dir = r"E:\DATASET"

def read_and_align_raster(reference_raster, file_path):
    """Align raster to match reference dimensions and CRS"""
    try:
        with rasterio.open(file_path) as src:
            transform, width, height = rasterio.warp.calculate_default_transform(
                src.crs,
                reference_raster.crs,
                reference_raster.width,
                reference_raster.height,
                *reference_raster.bounds
            )
            data = np.empty((height, width), dtype=np.float32)
            reproject(
                source=rasterio.band(src, 1),
                destination=data,
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=transform,
                dst_crs=reference_raster.crs,
                resampling=Resampling.bilinear
            )
            return data
    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")
        return None

# Load DEM as reference raster
with rasterio.open(os.path.join(root_dir, "DEM", "DEM_Final.tif")) as dem_src:
    dem_data = dem_src.read(1)
    aligned_rasters = {'dem': dem_data}

# Define all raster components
raster_components = {
    'aspect': ("Aspect", "Aspect_Final.tif"),
    'curvature': ("Curvature", "Curvature_Final.tif"),
    'distance_to_road': ("Distance from road", "Distance_To_Road.tif"),
    'distance_to_settlements': ("Distance from Settlements", "Dist_To_Settlements_Final.tif"),
    'distance_to_stream': ("Distance from Stream", "Distance_To_Stream.tif"),
    'lulc': ("LULC", "LULC_Final.tif"),
    'ndvi': ("NDVI", "NDVI_final.tif"),
    'rainfall': ("Rainfall", "Max_Rainfall.tif"),
    'slope': ("Slope", "Slope_Final.tif"),
    'soil': ("Soil", "Soil_Final.tif"),
    'twi': ("TWI", "TWI_Final.tif")
}

# Load and align other rasters
for key, (folder, filename) in raster_components.items():
    file_path = os.path.join(root_dir, folder, filename)
    aligned_data = read_and_align_raster(dem_src, file_path)
    if aligned_data is not None:
        if aligned_data.shape == dem_data.shape:
            aligned_rasters[key] = aligned_data
        else:
            print(f"Shape mismatch in {filename} even after alignment")
    else:
        print(f"Skipped {filename} due to processing errors")

# Verify loaded rasters
print(f"Successfully loaded {len(aligned_rasters)} raster components")
assert len(aligned_rasters) >= 3, "Insufficient valid rasters for analysis"

# Load landslide labels raster
ls_nls_path = os.path.join(root_dir, "LS_WG1 Data", "WG_LS_NLS.tiff")
with rasterio.open(ls_nls_path) as ls_nls_src:
    ls_nls_data = ls_nls_src.read(1)  # Read the first band

# Extract feature-label pairs for all pixels
X, y = [], []
height, width = dem_data.shape

for y_coord in range(height):
    for x in range(width):
        label = ls_nls_data[y_coord, x]
        if label in [0, 1]:  # Ensure it's a valid class
            try:
                features = [aligned_rasters[key][y_coord, x] for key in aligned_rasters]
                X.append(features)
                y.append(label)
            except (IndexError, KeyError):
                continue



# Convert to numpy arrays
X = np.array(X)
y = np.array(y)

# Normalize features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Reshape for Conv1D (samples, features, channels)
X = X.reshape(-1, X.shape[1], 1)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build 1D CNN model
model = Sequential([
    Conv1D(64, 2, activation='relu', input_shape=(X.shape[1], 1)),
    MaxPooling1D(1),
    Conv1D(128, 2, activation='relu'),
    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

model.compile(
    optimizer=Adam(learning_rate=0.0005),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Train model
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=30,
    batch_size=64,
    verbose=1
)

# Save model
model.save(os.path.join(root_dir, 'landslide_cnn_model_v2.h5'))

# ================== ADDITIONAL OUTPUTS ================== #
def compute_feature_importance(model, X_sample):
    """Compute gradient-based feature importance"""
    input_tensor = tf.convert_to_tensor(X_sample)
    with tf.GradientTape() as tape:
        tape.watch(input_tensor)
        outputs = model(input_tensor)
    grads = tape.gradient(outputs, input_tensor)
    return np.mean(np.abs(grads.numpy()), axis=0)[:, 0]

# Feature importance visualization
sample_indices = np.random.choice(len(X_test), 100, replace=False)
feature_importance = compute_feature_importance(model, X_test[sample_indices])

plt.figure(figsize=(10, 6))
plt.barh(list(aligned_rasters.keys()), feature_importance)
plt.title('Feature Importance Scores')
plt.xlabel('Mean Absolute Gradient')
plt.tight_layout()
plt.savefig(os.path.join(root_dir, 'feature_importance.png'), dpi=300)
plt.close()

# Susceptibility map generation
def generate_susceptibility_map():
    susceptibility = np.zeros(dem_data.shape, dtype=np.float32)
    batch_size = 1000
    height, width = dem_data.shape
    total_pixels = height * width
    
    for i in range(0, total_pixels, batch_size):
        batch = []
        coords = []
        
        for j in range(i, min(i+batch_size, total_pixels)):
            y, x = divmod(j, width)
            try:
                features = [aligned_rasters[key][y, x] for key in aligned_rasters]
                batch.append(features)
                coords.append((y, x))
            except IndexError:
                continue
        
        if not batch:
            continue
            
        X_batch = scaler.transform(np.array(batch))
        X_batch = X_batch.reshape(-1, X_batch.shape[1], 1)
        preds = model.predict(X_batch, verbose=0)[:, 0]
        
        for (y, x), prob in zip(coords, preds):
            susceptibility[y, x] = prob
            
    return susceptibility

susceptibility_map = generate_susceptibility_map()

# Save susceptibility map as TIFF
tifffile.imwrite(
    os.path.join(root_dir, 'landslide_susceptibility.tif'),
    susceptibility_map,
    metadata={'axes': 'YX'},
    dtype=np.float32
)

# Create PNG visualization
plt.figure(figsize=(15, 10))
plt.imshow(susceptibility_map, cmap='viridis', norm=Normalize(vmin=0, vmax=1))
plt.colorbar(label='Susceptibility Probability')
plt.axis('off')
plt.title('Landslide Susceptibility Map')
plt.savefig(os.path.join(root_dir, 'landslide_susceptibility_map.png'), bbox_inches='tight', dpi=300)
plt.close()

# ================== ORIGINAL PLOTTING CODE ================== #
# Plot training history
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend()

plt.tight_layout()
plt.savefig(os.path.join(root_dir, 'training_history.png'), dpi=300)
plt.show()
