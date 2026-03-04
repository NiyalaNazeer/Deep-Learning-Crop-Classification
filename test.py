import os
import spectral  # type: ignore
import numpy as np  # type: ignore
import tensorflow as tf  # type: ignore
from tensorflow.keras.models import load_model  # type: ignore
import joblib  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
import matplotlib.patches as mpatches  # type: ignore
import gc
import random
import rasterio  # type: ignore
from rasterio.transform import from_origin  # type: ignore
from matplotlib.colors import ListedColormap  # type: ignore


model = load_model("./models/crop_classification_model.h5")
encoder = joblib.load("./models/label_encoder.pkl")
scaler = joblib.load("./models/feature_scaler.pkl")


print("Model loaded successfully. 🌍")
print("please wait 🌅")


hyperspectral_hdr_path = "Reflectance_Hyperspectral_Data/Jhagdia_Ref_Hyperspectral_Data.hdr"


classification_folder = "Crop_Location_Data"
hyperspectral_name = os.path.basename(hyperspectral_hdr_path).replace("_Ref_Hyperspectral_Data.hdr", "")
classification_hdr_path = None

for file in os.listdir(classification_folder):
    if file.startswith(hyperspectral_name) and file.endswith(".hdr"):
        classification_hdr_path = os.path.join(classification_folder, file)
        break

if not classification_hdr_path:
    raise FileNotFoundError(f"No classification HDR file found for {hyperspectral_name} in {classification_folder}")

print(f"Using classification file: {classification_hdr_path}")


def load_hyperspectral_image(hdr_path):
    img = spectral.open_image(hdr_path)
    return np.array(img.load(), dtype=np.float32), img.metadata


def load_classification_metadata(hdr_path):
    img = spectral.open_image(hdr_path)
    return img.metadata


image, hyperspectral_metadata = load_hyperspectral_image(hyperspectral_hdr_path)
height, width, num_bands = image.shape


classification_metadata = load_classification_metadata(classification_hdr_path)


map_info = hyperspectral_metadata.get("map info") or classification_metadata.get("map info")

if map_info:
    if isinstance(map_info, str):
        map_info = map_info.split(",")
    try:
        x_start, y_start = float(map_info[3]), float(map_info[4])
        pixel_size_x, pixel_size_y = float(map_info[5]), float(map_info[6])
        projection = "EPSG:32643" 
    except (IndexError, ValueError):
        print("Error: Invalid map_info format.")
        x_start, y_start, pixel_size_x, pixel_size_y = 0, 0, 1, 1
        projection = "EPSG:32643"
else:
    print("Warning: No geospatial data found in metadata.")
    x_start, y_start, pixel_size_x, pixel_size_y = 0, 0, 1, 1
    projection = "EPSG:32643"


class_names = classification_metadata.get("class names", [])

if not class_names:
    class_names = hyperspectral_metadata.get("class names", [])

if isinstance(class_names, str):
    class_names = [name.strip() for name in class_names.split(",") if name.strip()]

if not class_names:
    class_names = [] 


image_reshaped = image.reshape(-1, num_bands)
image_reshaped = scaler.transform(image_reshaped)


predictions = model.predict(image_reshaped)
predicted_classes = np.argmax(predictions, axis=1)
predicted_map = predicted_classes.reshape(height, width)


if not class_names:
    class_names = [f"Class {i}" for i in range(np.max(predicted_classes) + 1)]


def generate_colors(n):
    random.seed(42)
    return [(random.random(), random.random(), random.random()) for _ in range(n)]

unique_classes = len(class_names)
class_colors = generate_colors(unique_classes)
cmap = ListedColormap(class_colors[:unique_classes])


fig, ax = plt.subplots(figsize=(12, 8))
im = ax.imshow(predicted_map, cmap=cmap)


patches = [mpatches.Patch(color=class_colors[i], label=class_names[i]) for i in range(unique_classes)]
legend = ax.legend(handles=patches, loc="upper right", bbox_to_anchor=(2, 1), title="Crop Types", fontsize=10, frameon=True)


cbar = plt.colorbar(im, ax=ax, shrink=0.7, pad=0.02)
cbar.set_label("Crop Type", fontsize=12)


plt.title("Predicted Crop Classification Map", fontsize=14, fontweight="bold")
plt.xlabel("Width", fontsize=12)
plt.ylabel("Height", fontsize=12)


plt.tight_layout(rect=[0, 0, 0.85, 1])
plt.savefig("./output/final_crop_classification.png", bbox_inches="tight")
plt.show()


transform = from_origin(x_start, y_start, pixel_size_x, -pixel_size_y)
output_tif_path = "./output/crop_classification_map.tif"

with rasterio.open(
    output_tif_path, "w", driver="GTiff",
    height=height, width=width,
    count=1, dtype=np.uint8,
    crs=projection, transform=transform
) as dst:
    dst.write(predicted_map.astype(np.uint8), 1)

print(f"GeoTIFF saved with UTM Zone 43N, WGS-84 projection at: {output_tif_path}")

del image, image_reshaped, predictions
gc.collect()
