import os
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tqdm import tqdm

# Parameters
img_height, img_width = 150, 150
data_dir = 'Apple_Folder/apple_dataset'
categories = os.listdir(os.path.join(data_dir, 'train'))

# 1. Load and preprocess image data
def load_images_and_labels(folder):
    images = []
    labels = []
    for label, category in enumerate(categories):
        path = os.path.join(folder, category)
        for img_name in tqdm(os.listdir(path), desc=f"Loading {category}"):
            img_path = os.path.join(path, img_name)
            try:
                img = load_img(img_path, target_size=(img_height, img_width))
                img_array = img_to_array(img)
                img_array = preprocess_input(img_array)  # VGG16-style preprocessing
                images.append(img_array)
                labels.append(label)
            except:
                continue
    return np.array(images), np.array(labels)

X_train_raw, y_train = load_images_and_labels(os.path.join(data_dir, 'train'))
X_val_raw, y_val = load_images_and_labels(os.path.join(data_dir, 'validation'))

# 2. Feature extraction using VGG16 (no top layer)
vgg = VGG16(weights='imagenet', include_top=False, input_shape=(img_height, img_width, 3))
X_train_features = vgg.predict(X_train_raw)
X_val_features = vgg.predict(X_val_raw)

# Flatten the features for Random Forest input
X_train_flat = X_train_features.reshape(X_train_features.shape[0], -1)
X_val_flat = X_val_features.reshape(X_val_features.shape[0], -1)

# 3. Train Random Forest classifier
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train_flat, y_train)

# 4. Predict and evaluate
y_pred = rf_model.predict(X_val_flat)

# Accuracy
acc = accuracy_score(y_val, y_pred)
print(f"Random Forest Accuracy: {acc:.4f}")

# Confusion Matrix
cm = confusion_matrix(y_val, y_pred)
print("Confusion Matrix:")
print(cm)

# Classification Report
print("Classification Report:")
print(classification_report(y_val, y_pred, target_names=categories))
