# Player_segregation_task
The objective of our task is to classify images of players into separate classes from a badminton game data, consisting of two folders and an image of the court.

## Overview
This repository contains implementations for classifying images of badminton players into separate classes. The task is accomplished using three distinct approaches: a Convolutional Neural Network (CNN), OpenCV techniques for image segmentation-clustering, and a YOLO-based detection method.

## Directory Structure
```
/Implementation_1 
    ├── player_segregation.py 
    ├── player_segregation_final.py 
    └── best_model.keras
    └── train
/Implementation_2 
    ├── hsv_range.py 
    └── player_segregation.py
/Implementation_3 
    ├── yolo_train.ipynb 
    ├── dataset
    ├── dataset.zip
    └── yolov5 
/two_players_top
    └── (Images of players from the top half) 
/two_players_bot 
    └── (Images of players from the bottom half)
/execute.sh 
/execute_.sh
/execute2.sh
/Output.jpg
```

## Requirements
- Python 3.x
- TensorFlow or Keras for the CNN implementation
- OpenCV for image processing
- YOLOv5 repository for the YOLO implementation

You can install the required Python packages using:

```bash
pip install -r requirements.txt
```

## Implementations

### 1. CNN-Based Approach
This implementation uses a CNN for image classification.

#### Running the Implementation
1. Run the training and prediction script:
   ```bash
   bash execute.sh
   ```
   or use the following to only run predictions:
   ```bash
   bash execute_.sh
   ```

### 2. OpenCV-Based Approach
This implementation utilizes OpenCV for color segmentation and clustering.

#### Running the Implementation
1. Run the script:
   ```bash
   bash execute2.sh
   ```

### 3. YOLO-Based Approach
This implementation fine-tunes a YOLOv5 model to detect players.

#### Running the Implementation
1. Navigate to the `Implementation_3` directory.
2. Run the training script:
   ```bash
   jupyter notebook yolo_train.ipynb
   ```
   Follow the instructions within the notebook to complete the training and prediction tasks.

## Notes
- You may consider adjusting hyperparameters like the number of epochs and batch size based on your dataset and available computational resources.

