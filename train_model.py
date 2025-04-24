from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint

# Data directories
train_dir = 'Apple_Folder/apple_dataset/train'
val_dir = 'Apple_Folder/apple_dataset/validation'

# Image settings
img_height, img_width = 150, 150
batch_size = 32

# Data augmentation
train_datagen = ImageDataGenerator(rescale=1./255)
val_datagen = ImageDataGenerator(rescale=1./255)

train_gen = train_datagen.flow_from_directory(train_dir,
                                              target_size=(img_height, img_width),
                                              batch_size=batch_size,
                                              class_mode='categorical')

val_gen = val_datagen.flow_from_directory(val_dir,
                                          target_size=(img_height, img_width),
                                          batch_size=batch_size,
                                          class_mode='categorical')

# Build the model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(img_height, img_width, 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(train_gen.num_classes, activation='softmax')
])

# Compile
model.compile(optimizer=Adam(learning_rate=0.0001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Save model
checkpoint = ModelCheckpoint('apple_keras_model.h5', save_best_only=True)

# Train
model.fit(train_gen,
          validation_data=val_gen,
          epochs=10,
          callbacks=[checkpoint])
