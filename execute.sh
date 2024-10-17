#!/bin/bash

echo "Starting the player segregation script..."

start_time=$SECONDS

echo "Installing required packages..."
pip install --upgrade pip
pip install tensorflow numpy keras shutil

cd "$(dirname "$0")" || exit 1  

python 'Implementation_1/player_segregation.py'

echo "Script execution complete."

elapsed_time=$(($SECONDS - $start_time))
echo "Total execution time: $elapsed_time seconds"

read -p "Press [Enter] key to close the terminal..."