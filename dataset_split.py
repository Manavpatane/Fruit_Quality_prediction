import os
import shutil
import random

# Set your paths
original_data_dir = 'apple'
base_dir = 'apple_dataset'
train_dir = os.path.join(base_dir, 'train')
val_dir = os.path.join(base_dir, 'validation')

# Create target directories
for split_dir in [train_dir, val_dir]:
    for category in ['apple_bad', 'apple_good']:
        os.makedirs(os.path.join(split_dir, category), exist_ok=True)

# Define split ratio
split_ratio = 0.8  # 80% training, 20% validation

# Function to split and move files
def split_and_copy(category):
    src_dir = os.path.join(original_data_dir, category)
    all_files = os.listdir(src_dir)
    random.shuffle(all_files)

    split_idx = int(len(all_files) * split_ratio)
    train_files = all_files[:split_idx]
    val_files = all_files[split_idx:]

    for fname in train_files:
        shutil.copy(os.path.join(src_dir, fname), os.path.join(train_dir, category, fname))
    for fname in val_files:
        shutil.copy(os.path.join(src_dir, fname), os.path.join(val_dir, category, fname))

# Run for each category
split_and_copy('apple_bad')
split_and_copy('apple_good')

print("✅ Data successfully split into train and validation folders.")
