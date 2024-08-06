#!/bin/bash

# Command to check if the container is running
CHECK_CONTAINER_CMD="docker ps -q -f name=$michael-isaac-lab-base"

# Create a new session named "michael-isaac-lab-docker-session"
tmux new-session -d -s michael-isaac-lab-docker-session

# Split the window vertically
tmux split-window -v

# Start Docker container in the first pane
tmux send-keys -t michael-isaac-lab-docker-session:0.0 './container_haku.sh start' C-m

# Wait for the docker container to start, then enter in the second pane
tmux send-keys -t michael-isaac-lab-docker-session "while [ -z \"\$($CHECK_CONTAINER_CMD)\" ]; do sleep 1; done" C-m
tmux send-keys -t michael-isaac-lab-docker-session:0.1 './container_haku.sh enter' C-m

# Attach to the session
tmux attach-session -t michael-isaac-lab-docker-session