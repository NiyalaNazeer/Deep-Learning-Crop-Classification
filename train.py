import spectral # type: ignore
import numpy as np # type: ignore
import tensorflow as tf # type: ignore
from tensorflow.keras import layers, models # type: ignore
from sklearn.model_selection import train_test_split # type: ignore
from sklearn.preprocessing import LabelEncoder, StandardScaler # type: ignore
from sklearn.metrics import classification_report, confusion_matrix # type: ignore
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau # type: ignore
import matplotlib.pyplot as plt # type: ignore
import seaborn as sns # type: ignore
import joblib    # type: ignore
import gc 
from sklearn.ensemble import RandomForestClassifier  # type: ignore
print("TensorFlow version:", tf.__version__)



np.random.seed(42)
tf.random.set_seed(42)

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
    print(f"Loading hyperspectral data from {hdr_path}  🐞")
    img = spectral.open_image(hdr_path).load()
    return np.array(img, dtype=np.float32)

def load_crop_labels(label_path):
    print(f"Loading crop labels from {label_path}   🌵")
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
    print(f"Processing location  {i+1}  ♨️")

    num_pixels = img.shape[0] * img.shape[1]
    
    lbl = lbl.reshape(-1) if len(lbl.shape) == 2 else lbl.reshape(-1)
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

print("\nCalculating feature importance...🚀")
rf = RandomForestClassifier(n_estimators=50, max_depth=10, n_jobs=-1, random_state=42)
rf.fit(X_train[:10000] if len(X_train) > 10000 else X_train, 
       y_train[:10000] if len(y_train) > 10000 else y_train)

plt.figure(figsize=(10, 6))
importances = rf.feature_importances_
indices = np.argsort(importances)[-20:] 
plt.barh(range(20), importances[indices])
plt.yticks(range(20), [f'Band {i}' for i in indices])
plt.xlabel('Feature Importance')
plt.title('Top 20 Important Spectral Bands')
plt.tight_layout()
plt.savefig('./output/feature_importance.png')
print("Feature importance plot saved as 'feature_importance.png'")

def create_model(input_shape, num_classes, dropout_rate=0.3):
    model = models.Sequential([
        layers.Input(shape=(input_shape,)),
        layers.BatchNormalization(),
        layers.Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        layers.Dropout(dropout_rate),
        layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        layers.Dropout(dropout_rate),
        layers.Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    
    return model

num_classes = len(np.unique(y_train))
model = create_model(X_train.shape[1], num_classes)
model.summary()


unique_classes, class_counts = np.unique(y_train, return_counts=True)
total_samples = len(y_train)
class_weights = {cls: total_samples / (len(unique_classes) * count) for cls, count in zip(unique_classes, class_counts)}
print("\nClass weights:", class_weights)

callbacks = [
    EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=1)
]


history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=50,
    batch_size=64,
    callbacks=callbacks,
    class_weight=class_weights,
    verbose=1
)


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


print("\nEvaluating model...")
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=1)
print(f"Test Accuracy: {test_acc:.4f}")


y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)


class_names = [str(cls) for cls in encoder.classes_]
report = classification_report(y_test, y_pred_classes, target_names=class_names, zero_division=0)
print("\nClassification Report:\n", report)


cm = confusion_matrix(y_test, y_pred_classes)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.tight_layout()
plt.savefig('./output/confusion_matrix.png')
print("Confusion matrix saved as 'confusion_matrix.png'")


model.save("./models/crop_classification_model.h5")
model.save("./models/crop_classification_model.keras")
print("Model saved in both .h5 and .keras formats")


location_accuracy = {}
for location in np.unique(loc_test):
    loc_mask = (loc_test == location)
    if np.sum(loc_mask) > 0:
        loc_acc = np.mean(y_pred_classes[loc_mask] == y_test[loc_mask])
        location_accuracy[int(location)] = loc_acc

print("\nAccuracy by location:")
for loc, acc in location_accuracy.items():
    print(f"Location {loc+1}: {acc:.4f}")

print("\nTraining completed successfully! 🚀")
