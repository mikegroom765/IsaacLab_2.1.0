#!/bin/bash

# Command to check if the container is running
CHECK_CONTAINER_CMD="docker ps -q -f name=$michael-isaac-lab-base"
CHECK_CONTAINER_CMD_2="docker ps -q -f name=$michael-isaac-lab-base-2"

# Create a new session named "michael-isaac-lab-docker-session"
tmux new-session -d -s michael-isaac-lab-docker-session

# Split the window vertically
tmux split-window -v

# Start Docker container in the first pane
tmux send-keys -t michael-isaac-lab-docker-session:0.0 './container_haku.sh start' C-m

# Wait for the docker container to start, then enter in the second pane
tmux send-keys -t michael-isaac-lab-docker-session "while [ -z \"\$($CHECK_CONTAINER_CMD)\" ]; do sleep 1; done" C-m
tmux send-keys -t michael-isaac-lab-docker-session:0.1 'docker exec -it --tty michael-isaac-lab-base bash' C-m

# Create a new window named michael-isaac-lab-base-2
tmux new-window -n michael-isaac-lab-base-2

# Split the window vertically
tmux split-window -v

# Wait for the docker container to start, then enter in the second pane
tmux send-keys -t michael-isaac-lab-docker-session:1.0 "while [ -z \"\$($CHECK_CONTAINER_CMD_2)\" ]; do sleep 1; done" C-m
tmux send-keys -t michael-isaac-lab-docker-session:1.0 'docker exec -it --tty michael-isaac-lab-base-2 bash' C-m

# Attach to the session
tmux attach-session -t michael-isaac-lab-docker-session