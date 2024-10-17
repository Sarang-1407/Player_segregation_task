import cv2
import os
import numpy as np
from sklearn.cluster import KMeans

script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)

output_dirs = [os.path.join(script_dir, 'output/player1'),
               os.path.join(script_dir, 'output/player2'),
               os.path.join(script_dir, 'output/player3'),
               os.path.join(script_dir, 'output/player4')]

for directory in output_dirs:
    if not os.path.exists(directory):
        os.makedirs(directory)

court_img = cv2.imread(os.path.join(parent_dir, 'output.jpg'))
gray_court = cv2.cvtColor(court_img, cv2.COLOR_BGR2GRAY)

top_dir = os.path.join(parent_dir, 'two_players_top')
bot_dir = os.path.join(parent_dir, 'two_players_bot')

top_images = [cv2.imread(os.path.join(top_dir, img)) for img in os.listdir(top_dir)]
bot_images = [cv2.imread(os.path.join(bot_dir, img)) for img in os.listdir(bot_dir)]

def segment_players(image, lower_bound, upper_bound):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_bound, upper_bound)
    player_mask = cv2.bitwise_not(mask)
    kernel = np.ones((5, 5), np.uint8)
    cleaned_mask = cv2.morphologyEx(player_mask, cv2.MORPH_CLOSE, kernel)
    players_segmented = cv2.bitwise_and(image, image, mask=cleaned_mask)
    return players_segmented, cleaned_mask


lower_red = np.array([-6, 82, 67])
upper_red = np.array([14, 162, 147])


segmented_top_images = [segment_players(img, lower_red, upper_red) for img in top_images]
segmented_bot_images = [segment_players(img, lower_red, upper_red) for img in bot_images]


def extract_hog_features(image, resize_dim=(128, 128)):
    image_resized = cv2.resize(image, resize_dim)
    gray_img = cv2.cvtColor(image_resized, cv2.COLOR_BGR2GRAY)
    hog = cv2.HOGDescriptor()
    hog_features = hog.compute(gray_img)
    hog_features = hog_features.flatten()
    return hog_features

top_features = [extract_hog_features(img) for img, _ in segmented_top_images]
bot_features = [extract_hog_features(img) for img, _ in segmented_bot_images]

top_features = np.array(top_features)
bot_features = np.array(bot_features)
all_features = np.vstack((top_features, bot_features))

kmeans = KMeans(n_clusters=4, random_state=42)
kmeans.fit(all_features)

labels = kmeans.labels_

top_player1 = [top_images[i] for i, label in enumerate(labels[:len(top_images)]) if label == 0]
top_player2 = [top_images[i] for i, label in enumerate(labels[:len(top_images)]) if label == 1]
bot_player1 = [bot_images[i] for i, label in enumerate(labels[len(top_images):]) if label == 2]
bot_player2 = [bot_images[i] for i, label in enumerate(labels[len(top_images):]) if label == 3]


def save_images(images, directory, filenames):
    for img, filename in zip(images, filenames):
        img_path = os.path.join(directory, filename)  
        cv2.imwrite(img_path, img)

top_filenames = os.listdir(top_dir)
bot_filenames = os.listdir(bot_dir)

save_images(top_player1, os.path.join(script_dir, 'output/player1'), top_filenames)
save_images(top_player2, os.path.join(script_dir, 'output/player2'), top_filenames)
save_images(bot_player1, os.path.join(script_dir, 'output/player3'), bot_filenames)
save_images(bot_player2, os.path.join(script_dir, 'output/player4'), bot_filenames)

print("Player folders created successfully.")
