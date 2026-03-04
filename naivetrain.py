import spectral # type: ignore
import numpy as np # type: ignore
import joblib # type: ignore
import gc
import matplotlib.pyplot as plt # type: ignore
import seaborn as sns # type: ignore

from sklearn.model_selection import train_test_split # type: ignore
from sklearn.preprocessing import LabelEncoder, StandardScaler # type: ignore
from sklearn.metrics import classification_report, confusion_matrix # type: ignore
from sklearn.naive_bayes import GaussianNB # type: ignore

print("Loading libraries completed. 🚀")


np.random.seed(42)


hyperspectral_paths = [
    "Reflectance_Hyperspectral_Data/Anand_Ref_Hyperspectral_Data.hdr",
    "Reflectance_Hyperspectral_Data/Jhagdia_Ref_Hyperspectral_Data.hdr",
    "Reflectance_Hyperspectral_Data/Kota_Ref_Hyperspectral_Data.hdr",
    "Reflectance_Hyperspectral_Data/Maddur_Ref_Hyperspectral_Data.hdr",
    "Reflectance_Hyperspectral_Data/Talala_Ref_Hyperspectral_Data.hdr",
]

crop_location_paths = [
    "Crop_Location_Data/Anand_Cls_Data.hdr",
    "Crop_Location_Data/Jhagdia_Cls_Data.hdr",
    "Crop_Location_Data/Kota_Cls_Data.hdr",
    "Crop_Location_Data/Maddur_Cls_Data.hdr",
    "Crop_Location_Data/Talala_Cls_Data.hdr",
]


def load_hyperspectral_image(hdr_path):
    print(f"Loading hyperspectral data from {hdr_path} 🐞")
    img = spectral.open_image(hdr_path).load()
    return np.array(img, dtype=np.float32)


def load_crop_labels(label_path):
    print(f"Loading crop labels from {label_path} 🌵")
    labels = spectral.open_image(label_path).load()
    return np.array(labels, dtype=np.int16)

hyperspectral_data, crop_labels, locations = [], [], []


for i, (img_path, lbl_path) in enumerate(zip(hyperspectral_paths, crop_location_paths)):
    try:
        img = load_hyperspectral_image(img_path)
        lbl = load_crop_labels(lbl_path)
        print(f"Location {i+1}: Image shape {img.shape}, Label shape {lbl.shape}")
        hyperspectral_data.append(img)
        crop_labels.append(lbl)
        locations.append(np.full((img.shape[0], img.shape[1]), i))
    except Exception as e:
        print(f"Error loading data from {img_path} or {lbl_path}: {e}")

if not hyperspectral_data:
    raise ValueError("No valid data was loaded. Check file paths and formats.")


X_pixels_list, y_pixels_list, loc_pixels_list = [], [], []

for i, (img, lbl, loc) in enumerate(zip(hyperspectral_data, crop_labels, locations)):
    print(f"Processing location {i+1} ♨️")

    num_pixels = img.shape[0] * img.shape[1]
    
    lbl = lbl.reshape(-1)
    img_pixels = img.reshape(num_pixels, -1)
    loc_pixels = loc.reshape(-1)

    valid_mask = (lbl > 0) & ~np.any(np.isnan(img_pixels), axis=1) & ~np.any(np.isinf(img_pixels), axis=1)
    valid_count = np.sum(valid_mask)
    print(f"Location {i+1}: Found {valid_count} valid pixels out of {num_pixels}")
    
    if valid_count > 0:
        X_pixels_list.append(img_pixels[valid_mask])
        y_pixels_list.append(lbl[valid_mask])
        loc_pixels_list.append(loc_pixels[valid_mask])


X_pixels = np.vstack(X_pixels_list)
y_pixels = np.concatenate(y_pixels_list)
loc_pixels = np.concatenate(loc_pixels_list)


del hyperspectral_data, crop_labels, X_pixels_list, y_pixels_list, loc_pixels_list
gc.collect()

print(f"Total dataset: {X_pixels.shape[0]} pixels with {X_pixels.shape[1]} bands")


print("Normalizing features...🕛🕧🕐")
scaler = StandardScaler()
X_pixels = scaler.fit_transform(X_pixels)
joblib.dump(scaler, "./models/feature_scaler.pkl")


encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y_pixels)
joblib.dump(encoder, "./models/label_encoder.pkl")


X_train, X_test, y_train, y_test, loc_train, loc_test = train_test_split(
    X_pixels, y_encoded, loc_pixels, test_size=0.2, random_state=42, stratify=y_encoded
)

del X_pixels, y_pixels, y_encoded, loc_pixels
gc.collect()


print("\nTraining Gaussian Naïve Bayes model... 🧠")
gnb = GaussianNB()
gnb.fit(X_train, y_train)


joblib.dump(gnb, "./models/naive_bayes_model.pkl")
print("Naïve Bayes model saved as 'naive_bayes_model.pkl'")

test_acc = gnb.score(X_test, y_test)
print(f"\nTest Accuracy (Naïve Bayes): {test_acc:.4f}")


y_pred_nb = gnb.predict(X_test)
num_classes = len(np.unique(y_train))
model = create_model(X_train.shape[1], num_classes)
model.summary()


history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=50,
    batch_size=64,
    callbacks=callbacks,
    class_weight=class_weights,
    verbose=1
)

print("\nNaïve Bayes Classification Report:\n", classification_report(y_test, y_pred_nb))

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.tight_layout()
plt.savefig('./output/training_history.png')
print("Training history plot saved as 'training_history.png'")


cm = confusion_matrix(y_test, y_pred_nb)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix (Naïve Bayes)')
plt.tight_layout()
plt.savefig('./output/naive_bayes_confusion_matrix.png')
print("Confusion matrix saved as 'naive_bayes_confusion_matrix.png'")


location_accuracy = {}
for location in np.unique(loc_test):
    loc_mask = (loc_test == location)
    if np.sum(loc_mask) > 0:
        loc_acc = np.mean(y_pred_nb[loc_mask] == y_test[loc_mask])
        location_accuracy[int(location)] = loc_acc

print("\nAccuracy by location (Naïve Bayes):")
for loc, acc in location_accuracy.items():
    print(f"Location {loc+1}: {acc:.4f}")

print("\nNaïve Bayes Model training and evaluation completed successfully! 🚀")
