import os
import tensorflow as tf
import shutil
import numpy as np
from tensorflow.keras import layers, models
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint

script_dir = os.path.dirname(os.path.abspath(__file__))

parent_dir = os.path.dirname(script_dir)

train_dir = os.path.join(script_dir, 'train')
output_dir = os.path.join(script_dir, 'output')


input_directories = [
    os.path.join(parent_dir, 'two_players_top'),
    os.path.join(parent_dir, 'two_players_bot')
]

datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
)

train_generator = datagen.flow_from_directory(
    train_dir,
    target_size=(150, 150),
    batch_size=8,
    class_mode='categorical',
    shuffle=True
)

model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    layers.MaxPooling2D(pool_size=(2, 2)),

    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(pool_size=(2, 2)),

    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D(pool_size=(2, 2)),

    layers.Conv2D(256, (3, 3), activation='relu'),
    layers.MaxPooling2D(pool_size=(2, 2)),

    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    layers.Dense(4, activation='softmax')
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model_checkpoint = ModelCheckpoint(
    os.path.join(script_dir, 'best_model.keras'),
    monitor='accuracy',
    save_best_only=True,
    mode='max',
    verbose=1
)

history = model.fit(
    train_generator,
    steps_per_epoch=len(train_generator),
    epochs=50,
    callbacks=[model_checkpoint],
    verbose=1
)

class_labels = ['player1', 'player2', 'player3', 'player4']
model = load_model(os.path.join(script_dir, 'best_model.keras'))

def create_directories(base_output_dir):
    for class_name in class_labels:
        class_dir = os.path.join(base_output_dir, class_name)
        os.makedirs(class_dir, exist_ok=True)

def move_image(image_path, class_index, output_base_dir):
    class_dir = os.path.join(output_base_dir, class_labels[class_index])
    shutil.copy(image_path, class_dir)

def predict_and_move_images(input_directories, output_base_dir):
    create_directories(output_base_dir)

    for directory_path in input_directories:
        for filename in os.listdir(directory_path):
            file_path = os.path.join(directory_path, filename)

            if os.path.isfile(file_path) and not filename.startswith('.'):
                class_index = predict_image(file_path)
                print(f"Image: {filename}, Predicted Class: {class_labels[class_index[0]]}")

                move_image(file_path, class_index[0], output_base_dir)

def predict_image(image_path):
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(150, 150))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    predictions = model.predict(img_array)
    class_index = np.argmax(predictions, axis=1)
    return class_index

predict_and_move_images(input_directories, output_dir)

print("Player folders Created.")
