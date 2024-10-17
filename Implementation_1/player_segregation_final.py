import os
import tensorflow as tf
import shutil
import numpy as np
from tensorflow.keras import layers, models
from tensorflow.keras.models import load_model

script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)

output_dir = os.path.join(script_dir, 'output')
input_directories = [
    os.path.join(parent_dir, 'two_players_top'),
    os.path.join(parent_dir, 'two_players_bot')
]

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