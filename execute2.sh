#!/bin/bash

echo "Starting the player segregation script..."

start_time=$SECONDS

echo "Installing required packages..."
pip install numpy opencv-python scikit-learn

cd "$(dirname "$0")" || exit 1  

python 'Implementation_2/player_segregation.py'

echo "Script execution complete."

elapsed_time=$(($SECONDS - $start_time))
echo "Total execution time: $elapsed_time seconds"

read -p "Press [Enter] key to close the terminal..."